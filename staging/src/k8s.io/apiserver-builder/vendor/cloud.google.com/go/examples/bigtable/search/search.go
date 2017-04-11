/*
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Search is a sample web server that uses Cloud Bigtable as the storage layer
// for a simple document-storage and full-text-search service.
// It has four functions:
// - Initialize and clear the table.
// - Add a document.  This adds the content of a user-supplied document to the
//   Bigtable, and adds references to the document to an index in the Bigtable.
//   The document is indexed under each unique word in the document.
// - Search the index.  This returns documents containing each word in a user
//   query, with snippets and links to view the whole document.
// - Copy table.  This copies the documents and index from another table and
//   adds them to the current one.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"cloud.google.com/go/bigtable"
	"golang.org/x/net/context"
)

var (
	addTemplate = template.Must(template.New("").Parse(`<html><body>
Added {{.Title}}
</body></html>`))

	contentTemplate = template.Must(template.New("").Parse(`<html><body>
<b>{{.Title}}</b><br><br>
{{.Content}}
</body></html>`))

	searchTemplate = template.Must(template.New("").Parse(`<html><body>
Results for <b>{{.Query}}</b>:<br><br>
{{range .Results}}
<a href="/content?name={{.Title}}">{{.Title}}</a><br>
<i>{{.Snippet}}</i><br><br>
{{end}}
</body></html>`))
)

const (
	indexColumnFamily   = "i"
	contentColumnFamily = "c"
	mainPage            = `
	<html>
		<head>
			<title>Document Search</title>
		</head>
		<body>
			Initialize and clear table:
			<form action="/reset" method="post">
				<div><input type="submit" value="Init"></div>
			</form>

			Search for documents:
			<form action="/search" method="post">
				<div><input type="text" name="q" size=80></div>
				<div><input type="submit" value="Search"></div>
			</form>

			Add a document:
			<form action="/add" method="post">
				Document name:
				<div><textarea name="name" rows="1" cols="80"></textarea></div>
				Document text:
				<div><textarea name="content" rows="20" cols="80"></textarea></div>
				<div><input type="submit" value="Submit"></div>
			</form>

			Copy data from another table:
			<form action="/copy" method="post">
				Source table name:
				<div><input type="text" name="name" size=80></div>
				<div><input type="submit" value="Copy"></div>
			</form>
		</body>
	</html>
	`
)

func main() {
	var (
		project   = flag.String("project", "", "The name of the project.")
		instance  = flag.String("instance", "", "The name of the Cloud Bigtable instance.")
		tableName = flag.String("table", "docindex", "The name of the table containing the documents and index.")
		port      = flag.Int("port", 8080, "TCP port for server.")
	)
	flag.Parse()

	// Make an admin client.
	adminClient, err := bigtable.NewAdminClient(context.Background(), *project, *instance)
	if err != nil {
		log.Fatal("Bigtable NewAdminClient:", err)
	}

	// Make a regular client.
	client, err := bigtable.NewClient(context.Background(), *project, *instance)
	if err != nil {
		log.Fatal("Bigtable NewClient:", err)
	}

	// Open the table.
	table := client.Open(*tableName)

	// Set up HTML handlers, and start the web server.
	http.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) { handleSearch(w, r, table) })
	http.HandleFunc("/content", func(w http.ResponseWriter, r *http.Request) { handleContent(w, r, table) })
	http.HandleFunc("/add", func(w http.ResponseWriter, r *http.Request) { handleAddDoc(w, r, table) })
	http.HandleFunc("/reset", func(w http.ResponseWriter, r *http.Request) { handleReset(w, r, *tableName, adminClient) })
	http.HandleFunc("/copy", func(w http.ResponseWriter, r *http.Request) { handleCopy(w, r, *tableName, client, adminClient) })
	http.HandleFunc("/", handleMain)
	log.Fatal(http.ListenAndServe(":"+strconv.Itoa(*port), nil))
}

// handleMain outputs the home page.
func handleMain(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, mainPage)
}

// tokenize splits a string into tokens.
// This is very simple, it's not a good tokenization function.
func tokenize(s string) []string {
	wordMap := make(map[string]bool)
	f := strings.FieldsFunc(s, func(r rune) bool { return !unicode.IsLetter(r) })
	for _, word := range f {
		word = strings.ToLower(word)
		wordMap[word] = true
	}
	words := make([]string, 0, len(wordMap))
	for word := range wordMap {
		words = append(words, word)
	}
	return words
}

// handleContent fetches the content of a document from the Bigtable and returns it.
func handleContent(w http.ResponseWriter, r *http.Request, table *bigtable.Table) {
	ctx, _ := context.WithTimeout(context.Background(), 10*time.Second)
	name := r.FormValue("name")
	if len(name) == 0 {
		http.Error(w, "No document name supplied.", http.StatusBadRequest)
		return
	}

	row, err := table.ReadRow(ctx, name)
	if err != nil {
		http.Error(w, "Error reading content: "+err.Error(), http.StatusInternalServerError)
		return
	}
	content := row[contentColumnFamily]
	if len(content) == 0 {
		http.Error(w, "Document not found.", http.StatusNotFound)
		return
	}
	var buf bytes.Buffer
	if err := contentTemplate.ExecuteTemplate(&buf, "", struct{ Title, Content string }{name, string(content[0].Value)}); err != nil {
		http.Error(w, "Error executing HTML template: "+err.Error(), http.StatusInternalServerError)
		return
	}
	io.Copy(w, &buf)
}

// handleSearch responds to search queries, returning links and snippets for matching documents.
func handleSearch(w http.ResponseWriter, r *http.Request, table *bigtable.Table) {
	ctx, _ := context.WithTimeout(context.Background(), 10*time.Second)
	query := r.FormValue("q")
	// Split the query into words.
	words := tokenize(query)
	if len(words) == 0 {
		http.Error(w, "Empty query.", http.StatusBadRequest)
		return
	}

	// readRows reads from many rows concurrently.
	readRows := func(rows []string) ([]bigtable.Row, error) {
		results := make([]bigtable.Row, len(rows))
		errors := make([]error, len(rows))
		var wg sync.WaitGroup
		for i, row := range rows {
			wg.Add(1)
			go func(i int, row string) {
				defer wg.Done()
				results[i], errors[i] = table.ReadRow(ctx, row, bigtable.RowFilter(bigtable.LatestNFilter(1)))
			}(i, row)
		}
		wg.Wait()
		for _, err := range errors {
			if err != nil {
				return nil, err
			}
		}
		return results, nil
	}

	// For each query word, get the list of documents containing it.
	results, err := readRows(words)
	if err != nil {
		http.Error(w, "Error reading index: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Count how many of the query words each result contained.
	hits := make(map[string]int)
	for _, r := range results {
		for _, r := range r[indexColumnFamily] {
			hits[r.Column]++
		}
	}

	// Build a slice of all the documents that matched every query word.
	var matches []string
	for doc, count := range hits {
		if count == len(words) {
			matches = append(matches, doc[len(indexColumnFamily+":"):])
		}
	}

	// Fetch the content of those documents from the Bigtable.
	content, err := readRows(matches)
	if err != nil {
		http.Error(w, "Error reading results: "+err.Error(), http.StatusInternalServerError)
		return
	}

	type result struct{ Title, Snippet string }
	data := struct {
		Query   string
		Results []result
	}{query, nil}

	// Output links and snippets.
	for i, doc := range matches {
		var text string
		c := content[i][contentColumnFamily]
		if len(c) > 0 {
			text = string(c[0].Value)
		}
		if len(text) > 100 {
			text = text[:100] + "..."
		}
		data.Results = append(data.Results, result{doc, text})
	}
	var buf bytes.Buffer
	if err := searchTemplate.ExecuteTemplate(&buf, "", data); err != nil {
		http.Error(w, "Error executing HTML template: "+err.Error(), http.StatusInternalServerError)
		return
	}
	io.Copy(w, &buf)
}

// handleAddDoc adds a document to the index.
func handleAddDoc(w http.ResponseWriter, r *http.Request, table *bigtable.Table) {
	if r.Method != "POST" {
		http.Error(w, "POST requests only", http.StatusMethodNotAllowed)
		return
	}

	ctx, _ := context.WithTimeout(context.Background(), time.Minute)

	name := r.FormValue("name")
	if len(name) == 0 {
		http.Error(w, "Empty document name!", http.StatusBadRequest)
		return
	}

	content := r.FormValue("content")
	if len(content) == 0 {
		http.Error(w, "Empty document content!", http.StatusBadRequest)
		return
	}

	var (
		writeErr error          // Set if any write fails.
		mu       sync.Mutex     // Protects writeErr
		wg       sync.WaitGroup // Used to wait for all writes to finish.
	)

	// writeOneColumn writes one column in one row, updates err if there is an error,
	// and signals wg that one operation has finished.
	writeOneColumn := func(row, family, column, value string, ts bigtable.Timestamp) {
		mut := bigtable.NewMutation()
		mut.Set(family, column, ts, []byte(value))
		err := table.Apply(ctx, row, mut)
		if err != nil {
			mu.Lock()
			writeErr = err
			mu.Unlock()
		}
	}

	// Start a write to store the document content.
	wg.Add(1)
	go func() {
		writeOneColumn(name, contentColumnFamily, "", content, bigtable.Now())
		wg.Done()
	}()

	// Start writes to store the document name in the index for each word in the document.
	words := tokenize(content)
	for _, word := range words {
		var (
			row    = word
			family = indexColumnFamily
			column = name
			value  = ""
			ts     = bigtable.Now()
		)
		wg.Add(1)
		go func() {
			// TODO: should use a semaphore to limit the number of concurrent writes.
			writeOneColumn(row, family, column, value, ts)
			wg.Done()
		}()
	}
	wg.Wait()
	if writeErr != nil {
		http.Error(w, "Error writing to Bigtable: "+writeErr.Error(), http.StatusInternalServerError)
		return
	}
	var buf bytes.Buffer
	if err := addTemplate.ExecuteTemplate(&buf, "", struct{ Title string }{name}); err != nil {
		http.Error(w, "Error executing HTML template: "+err.Error(), http.StatusInternalServerError)
		return
	}
	io.Copy(w, &buf)
}

// handleReset deletes the table if it exists, creates it again, and creates its column families.
func handleReset(w http.ResponseWriter, r *http.Request, table string, adminClient *bigtable.AdminClient) {
	if r.Method != "POST" {
		http.Error(w, "POST requests only", http.StatusMethodNotAllowed)
		return
	}
	ctx, _ := context.WithTimeout(context.Background(), 5*time.Minute)
	adminClient.DeleteTable(ctx, table)
	if err := adminClient.CreateTable(ctx, table); err != nil {
		http.Error(w, "Error creating Bigtable: "+err.Error(), http.StatusInternalServerError)
		return
	}
	time.Sleep(20 * time.Second)
	// Create two column families, and set the GC policy for each one to keep one version.
	for _, family := range []string{indexColumnFamily, contentColumnFamily} {
		if err := adminClient.CreateColumnFamily(ctx, table, family); err != nil {
			http.Error(w, "Error creating column family: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if err := adminClient.SetGCPolicy(ctx, table, family, bigtable.MaxVersionsPolicy(1)); err != nil {
			http.Error(w, "Error setting GC policy: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}
	w.Write([]byte("<html><body>Done.</body></html>"))
	return
}

// copyTable copies data from one table to another.
func copyTable(src, dst string, client *bigtable.Client, adminClient *bigtable.AdminClient) error {
	if src == "" || src == dst {
		return nil
	}
	ctx, _ := context.WithTimeout(context.Background(), time.Minute)

	// Open the source and destination tables.
	srcTable := client.Open(src)
	dstTable := client.Open(dst)

	var (
		writeErr error          // Set if any write fails.
		mu       sync.Mutex     // Protects writeErr
		wg       sync.WaitGroup // Used to wait for all writes to finish.
	)
	copyRowToTable := func(row bigtable.Row) bool {
		mu.Lock()
		failed := writeErr != nil
		mu.Unlock()
		if failed {
			return false
		}
		mut := bigtable.NewMutation()
		for family, items := range row {
			for _, item := range items {
				// Get the column name, excluding the column family name and ':' character.
				columnWithoutFamily := item.Column[len(family)+1:]
				mut.Set(family, columnWithoutFamily, bigtable.Now(), item.Value)
			}
		}
		wg.Add(1)
		go func() {
			// TODO: should use a semaphore to limit the number of concurrent writes.
			if err := dstTable.Apply(ctx, row.Key(), mut); err != nil {
				mu.Lock()
				writeErr = err
				mu.Unlock()
			}
			wg.Done()
		}()
		return true
	}

	// Create a filter that only accepts the column families we're interested in.
	filter := bigtable.FamilyFilter(indexColumnFamily + "|" + contentColumnFamily)
	// Read every row from srcTable, and call copyRowToTable to copy it to our table.
	err := srcTable.ReadRows(ctx, bigtable.InfiniteRange(""), copyRowToTable, bigtable.RowFilter(filter))
	wg.Wait()
	if err != nil {
		return err
	}
	return writeErr
}

// handleCopy copies data from one table to another.
func handleCopy(w http.ResponseWriter, r *http.Request, dst string, client *bigtable.Client, adminClient *bigtable.AdminClient) {
	if r.Method != "POST" {
		http.Error(w, "POST requests only", http.StatusMethodNotAllowed)
		return
	}
	src := r.FormValue("name")
	if src == "" {
		http.Error(w, "No source table specified.", http.StatusBadRequest)
		return
	}
	if err := copyTable(src, dst, client, adminClient); err != nil {
		http.Error(w, "Failed to rebuild index: "+err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Fprint(w, "Copied table.\n")
}
