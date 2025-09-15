package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/milvus"
)

func main() {
	ctx := context.Background()
	cfg := client.Config{
		Address: os.Getenv("MILVUS_URL"),
		APIKey:  os.Getenv("MILVUS_API_KEY"),
	}
	llm, err := googleai.New(
		ctx,
		googleai.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
		googleai.WithDefaultEmbeddingModel("gemini-embedding-001"),
	)
	if err != nil {
		log.Fatal(err)
	}
	embeddingModel, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatal(err)
	}

	index, err := entity.NewIndexAUTOINDEX(entity.L2)
	if err != nil {
		log.Fatal("failed to build index:", err)
	}

	store, err := milvus.New(
		ctx,
		cfg,
		milvus.WithEmbedder(embeddingModel),
		milvus.WithCollectionName("books"),
		milvus.WithIndex(index),
	)
	if err != nil {
		log.Fatal(err, store)
	}
	// file, err := os.Open("philosophy_simply_explained.txt")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// loadedDoc := documentloaders.NewText(file)
	// spliter := textsplitter.NewRecursiveCharacter(
	// 	textsplitter.WithChunkSize(512),
	// )
	// docs, err := loadedDoc.LoadAndSplit(ctx, spliter)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// store.AddDocuments(ctx, docs)

	// file, err := os.Open("philosophy_markdown_sections/Chinese_philosophy.md")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// loadedDoc := documentloaders.NewText(file)
	// spliter := textsplitter.NewMarkdownTextSplitter(
	// 	textsplitter.WithHeadingHierarchy(true),
	// )
	// docs, err := loadedDoc.LoadAndSplit(ctx, spliter)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// store.AddDocuments(ctx, docs)

	query := "Thoughts of Plato, Saussure and Saussure about Semiotics"
	results, err := store.SimilaritySearch(ctx, query, 30, vectorstores.WithScoreThreshold(0.85))
	if err != nil {
		log.Fatal(err)
	}
	providedInfomation := ""
	for _, result := range results {
		providedInfomation += result.PageContent
	}
	prompt := fmt.Sprintf("You are a content creator writting social media (Facebook) posts with a specific topic. "+
		"Using the provided infomation to write the content for a given topic, if the information is not related to the topic just say 'No related information'. "+
		"Make sure the format is suitable with Facebook posts (without bold, italic, bullet-points, ...). "+
		"And then, the final output must be translated that into Vietnamese and the whole output is post dicrecly into Facebook so don't generate some sentence like 'This is your information:' (Don't need to genarate the English content).\n"+
		"Topic: %v\nInfomation: %v\n", query, providedInfomation)
	out, err := llms.GenerateFromSinglePrompt(
		ctx,
		llm,
		prompt,
		llms.WithModel("gemini-2.0-flash"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(out)

	if askYesNo("Do you want to post this onto your Facebook Page?") {
		postOntoFacebookPage(out)
	}
}

func postOntoFacebookPage(content string) {
	url := fmt.Sprintf("https://graph.facebook.com/v23.0/%s/feed", os.Getenv("FACEBOOK_PAGE_ID"))

	data := map[string]string{
		"message":      content,
		"access_token": "EAALB93J4c70BPfHSZByno9bpDVUiCLDFXM0L2T5QWTszniQPbWqt4Gch8JRGmh5pCK4sOhpj9ZA9ZAhDMkA7zcZCCBzkCwLGYhmiWUJ3CSyhoHO7GIJde12VcwYP70aZCtlnyL3xjvaKFqpAxtx1ZCZB0siB84pf3ZB0ArXJVBARpDKZCE9iu94bdk5eKKAp51MlITXPf",
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
		if input == "y" || input == "yes" {
			return true
		}
		if input == "n" || input == "no" {
			return false
		}
		fmt.Println("Please type 'y' or 'n' and press Enter.")
	}
}
