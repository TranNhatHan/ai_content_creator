package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	milvus "github.com/tmc/langchaingo/vectorstores/milvus/v2"
	"github.com/tmc/langchaingo/textsplitter"
)

const (
	cseIDEnv = "GOOGLE_CSE_ID"
	apiKeyEnv = "GOOGLE_API_KEY"
)

type SearchResponse struct {
	Items []struct {
		Title   string `json:"title"`
		Link    string `json:"link"`
		Snippet string `json:"snippet"`
	} `json:"items"`
}

func main() {
	ctx := context.Background()
	topic := "Go and Python which programming language is better"

	llm, err := googleai.New(
		ctx,
		googleai.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
				 googleai.WithDefaultEmbeddingModel("gemini-embedding-001"),
	)
	if err != nil {
		log.Fatal(err)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatal(err)
	}

	config := milvusclient.ClientConfig{
		Address: os.Getenv("MILVUS_URL"),
		APIKey:  os.Getenv("MILVUS_API_KEY"),
	}

	idx := index.NewAutoIndex(entity.L2)

	store, err := milvus.New(
		ctx,
		config,
		milvus.WithEmbedder(embedder),
				 milvus.WithCollectionName("web_docs"),
				 milvus.WithIndex(idx),
	)
	if err != nil {
		log.Fatal(err)
	}

	getAnswer := func() (string, []schema.Document, error) {
		results, err := store.SimilaritySearch(
			ctx,
			topic,
			7,
			vectorstores.WithScoreThreshold(0.97),
		)
		if err != nil {
			return "", nil, err
		}

		info := ""
		for _, r := range results {
			info += r.PageContent + "\n"
		}

		prompt := fmt.Sprintf(`
		You are a Vietnamese social media writer.
		Write a short, engaging Facebook post using the information below.
		If the information is unrelated to the topic, reply "Không có thông tin liên quan."

		Topic: %s
		Information:
		%s
		`, topic, info)

		answer, err := llms.GenerateFromSinglePrompt(
			ctx,
			llm,
			prompt,
			llms.WithModel("gemini-2.0-flash"),
		)
		return answer, results, err
	}

	answer, _, err := getAnswer()
	if err != nil {
		log.Fatal(err)
	}

	if strings.Contains(strings.ToLower(answer), "không có thông tin liên quan") {
		fmt.Println("No related info in Milvus, crawling the web...")

		crawledTexts := crawlTopic(topic)
		if len(crawledTexts) == 0 {
			log.Fatal("No content found online.")
		}

		var docs []schema.Document
		for _, htmlText := range crawledTexts {
			chunks := splitTextToChunks(htmlText)
			for _, chunk := range chunks {
				docs = append(docs, schema.Document{
					PageContent: chunk,
				})
			}
		}

		fmt.Println("Adding new docs to Milvus...")
		_, err := store.AddDocuments(ctx, docs)
		if err != nil {
			log.Fatal("AddDocuments error:", err)
		}
		answer, _, err = getAnswer()
		if err != nil {
			log.Fatal(err)
		}
	}

	fmt.Println("Gemini Output:")
	fmt.Println(answer)


	if askYesNo("Do you want to post this onto your Facebook Page?") {
		postOntoFacebookPage(answer)
	}
}

func crawlTopic(topic string) []string {
	apiKey := strings.TrimSpace(os.Getenv(apiKeyEnv))
	cseID := strings.TrimSpace(os.Getenv(cseIDEnv))

	params := url.Values{}
	params.Add("key", apiKey)
	params.Add("cx", cseID)
	params.Add("q", url.QueryEscape(topic))
	params.Add("num", "3")

	searchURL := fmt.Sprintf("https://customsearch.googleapis.com/customsearch/v1?%s", params.Encode())

	resp, err := http.Get(searchURL)
	if err != nil {
		log.Printf("Search request failed: %v", err)
		return nil
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var results SearchResponse
	if err := json.Unmarshal(body, &results); err != nil {
		log.Printf("JSON parse error: %v", err)
		return nil
	}

	var texts []string
	for _, item := range results.Items {
		fmt.Println("Crawling:", item.Link)
		text := extractText(item.Link)
		if len(text) > 100 {
			texts = append(texts, text)
			time.Sleep(1 * time.Second)
		}
	}

	return texts
}

func extractText(pageURL string) string {
	resp, err := http.Get(pageURL)
	if err != nil {
		log.Printf("Fetch failed %s: %v", pageURL, err)
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return ""
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return ""
	}

	var b strings.Builder
	doc.Find("p, h1, h2, h3, li").Each(func(_ int, s *goquery.Selection) {
		txt := strings.TrimSpace(s.Text())
		if txt != "" {
			b.WriteString(txt + "\n")
		}
	})

	return b.String()
}

func splitTextToChunks(text string) []string {
	splitter := textsplitter.NewMarkdownTextSplitter(
		textsplitter.WithChunkSize(500),
		textsplitter.WithChunkOverlap(50),
	)
	chunks, err := splitter.SplitText(text)
	if err != nil {
		return []string{text}
	}
	return chunks
}

func postOntoFacebookPage(content string) {
	url := fmt.Sprintf(
		"https://graph.facebook.com/v23.0/%s/feed",
		os.Getenv("FACEBOOK_PAGE_ID"),
	)

	data := map[string]string{
		"message":      content,
		"access_token": os.Getenv("FACEBOOK_PAGE_ACCESS_TOKEN"),
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		fmt.Println("Post successfully created!")
	} else {
		fmt.Println("Failed to post:", resp.Status)
	}
}

func askYesNo(prompt string) bool {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(prompt, " (y/n): ")
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintln(os.Stderr, "read error:", err)
			return false
		}
		input = strings.TrimSpace(strings.ToLower(input))

		switch input {
			case "y", "yes":
				return true
			case "n", "no":
				return false
		}

		fmt.Println("Please type 'y' or 'n' and press Enter.")
	}
}
