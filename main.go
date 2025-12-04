package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/milvus"
)

func main() {
	ctx := context.Background()
	topic := "The rise of AI"

	llm, err := googleai.New(ctx, googleai.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		log.Fatal(err)
	}

	embedLLM, err := googleai.New(ctx,
		googleai.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
		googleai.WithDefaultEmbeddingModel("gemini-embedding-001"),
	)
	if err != nil {
		log.Fatal(err)
	}

	embedder, err := embeddings.NewEmbedder(embedLLM)
	if err != nil {
		log.Fatal(err)
	}

	index, _ := entity.NewIndexAUTOINDEX(entity.L2)
	store, err := milvus.New(
		ctx,
		client.Config{
			Address: os.Getenv("MILVUS_URL"),
			APIKey:  os.Getenv("MILVUS_API_KEY"),
		},
		milvus.WithCollectionName("web_docs"),
		milvus.WithIndex(index),
		milvus.WithEmbedder(embedder),
	)
	if err != nil {
		log.Fatal(err)
	}

	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "searchMilvus",
				Description: "Search the Milvus vector database for documents relevant to a topic.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{
							"type":        "string",
							"description": "The topic or search query.",
						},
					},
					"required": []string{"query"},
				},
			},
		},
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "crawlTopic",
				Description: "Crawl the web for the latest articles or text about a topic.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"topic": map[string]any{
							"type":        "string",
							"description": "The topic to crawl.",
						},
					},
					"required": []string{"topic"},
				},
			},
		},
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "postToFacebook",
				Description: "Post the generated Vietnamese Facebook post onto a Facebook page.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"content": map[string]any{
							"type":        "string",
							"description": "The text content to post.",
						},
					},
					"required": []string{"content"},
				},
			},
		},
	}

	task := fmt.Sprintf(`You are a Vietnamese social media writer.
	Your goal: Write an engaging Facebook post about "%s".
	You can use tools to gather and post information.
	If there's not enough info in Milvus, call "crawlTopic".
	After generating the post, call "postToFacebook".`, topic)

	messageHistory := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are an autonomous assistant that can call tools."),
		llms.TextParts(llms.ChatMessageTypeHuman, task),
	}

	resp, err := llm.GenerateContent(ctx, messageHistory, llms.WithTools(tools))
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) == 0 {
		log.Fatal("No response from LLM.")
	}

	choice := resp.Choices[0]
	assistantResponse := llms.TextParts(llms.ChatMessageTypeAI, choice.Content)
	for _, tc := range choice.ToolCalls {
		assistantResponse.Parts = append(assistantResponse.Parts, tc)
	}
	messageHistory = append(messageHistory, assistantResponse)

	// Handle tool calls
	for _, tc := range choice.ToolCalls {
		switch tc.FunctionCall.Name {
		case "searchMilvus":
			var args struct {
				Query string `json:"query"`
			}
			_ = json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args)
			results, _ := store.SimilaritySearch(ctx, args.Query, 20, vectorstores.WithScoreThreshold(0.9))
			fmt.Println("🔍 searchMilvus results:", len(results))

			toolResponse := llms.MessageContent{
				Role: llms.ChatMessageTypeTool,
				Parts: []llms.ContentPart{
					llms.ToolCallResponse{
						Name:    tc.FunctionCall.Name,
						Content: formatDocs(results),
					},
				},
			}
			messageHistory = append(messageHistory, toolResponse)

		case "crawlTopic":
			var args struct {
				Topic string `json:"topic"`
			}
			_ = json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args)
			fmt.Println("🌐 Crawling topic:", args.Topic)
			texts := crawlTopic(args.Topic)
			fmt.Println("Crawled docs:", len(texts))
			if len(texts) > 0 {
				var docs []schema.Document
				for _, t := range texts {
					docs = append(docs, schema.Document{PageContent: t})
				}
				_, err := store.AddDocuments(ctx, docs)
				if err != nil {
					log.Println("Error adding docs:", err)
				}
			}
			messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeTool,
				fmt.Sprintf("Crawled %d docs for %s", len(texts), args.Topic)))

		case "postToFacebook":
			var args struct {
				Content string `json:"content"`
			}
			_ = json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args)
			fmt.Println("📣 Posting to Facebook:", args.Content[:min(120, len(args.Content))], "...")
			postOntoFacebookPage(args.Content)

			messageHistory = append(messageHistory, llms.TextParts(llms.ChatMessageTypeTool,
				"Posted successfully!"))

		default:
			fmt.Println("⚠️ Unknown tool call:", tc.FunctionCall.Name)
		}
	}

	resp, err = llm.GenerateContent(ctx, messageHistory, llms.WithTools(tools))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\n🤖 Final Gemini Output:")
	b, _ := json.MarshalIndent(resp.Choices[0], "", "  ")
	fmt.Println(string(b))
}

func formatDocs(docs []schema.Document) string {
	var sb strings.Builder
	for _, d := range docs {
		sb.WriteString(d.PageContent)
		sb.WriteString("\n---\n")
	}
	return sb.String()
}

func crawlTopic(topic string) []string {
	time.Sleep(1 * time.Second)
	return []string{fmt.Sprintf("Example article content about %s.", topic)}
}

func postOntoFacebookPage(content string) {
	fmt.Println("✅ (Simulated) Posted to Facebook:", content[:min(100, len(content))])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
