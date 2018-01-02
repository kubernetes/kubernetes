// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package logging_test

import (
	"bytes"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"

	"cloud.google.com/go/preview/logging"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
)

var (
	client    *logging.Client
	projectID = flag.String("project-id", "", "ID of the project to use")
)

func ExampleClient_Entries_pagination() {
	// This example demonstrates how to iterate through items a page at a time
	// even if each successive page is fetched by a different process. It is a
	// complete web server that displays pages of log entries. To run it as a
	// standalone program, rename both the package and this function to "main".
	ctx := context.Background()
	flag.Parse()
	if *projectID == "" {
		log.Fatal("-project-id missing")
	}
	var err error
	client, err = logging.NewClient(ctx, *projectID)
	if err != nil {
		log.Fatalf("creating logging client: %v", err)
	}

	http.HandleFunc("/entries", handleEntries)
	log.Print("listening on 8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

var pageTemplate = template.Must(template.New("").Parse(`
<table>
  {{range .Entries}}
    <tr><td>{{.}}</td></tr>
  {{end}}
</table>
{{if .Next}}
  <a href="/entries?pageToken={{.Next}}">Next Page</a>
{{end}}
`))

func handleEntries(w http.ResponseWriter, r *http.Request) {
	ctx := context.Background()
	filter := fmt.Sprintf(`logName = "projects/%s/logs/testlog"`, *projectID)
	it := client.Entries(ctx, logging.Filter(filter))
	var entries []*logging.Entry
	nextTok, err := iterator.NewPager(it, 5, r.URL.Query().Get("pageToken")).NextPage(&entries)
	if err != nil {
		http.Error(w, fmt.Sprintf("problem getting the next page: %v", err), http.StatusInternalServerError)
		return
	}
	data := struct {
		Entries []*logging.Entry
		Next    string
	}{
		entries,
		nextTok,
	}
	var buf bytes.Buffer
	if err := pageTemplate.Execute(&buf, data); err != nil {
		http.Error(w, fmt.Sprintf("problem executing page template: %v", err), http.StatusInternalServerError)
	}
	if _, err := buf.WriteTo(w); err != nil {
		log.Printf("writing response: %v", err)
	}
}
