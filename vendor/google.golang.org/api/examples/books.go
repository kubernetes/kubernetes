// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	books "google.golang.org/api/books/v1"
	"google.golang.org/api/googleapi"
)

func init() {
	registerDemo("books", books.BooksScope, booksMain)
}

// booksMain is an example that demonstrates calling the Books API.
//
// Example usage:
//   go build -o go-api-demo *.go
//   go-api-demo -clientid="my-clientid" -secret="my-secret" books
func booksMain(client *http.Client, argv []string) {
	if len(argv) != 0 {
		fmt.Fprintln(os.Stderr, "Usage: books")
		return
	}

	svc, err := books.New(client)
	if err != nil {
		log.Fatalf("Unable to create Books service: %v", err)
	}

	bs, err := svc.Mylibrary.Bookshelves.List().Do()
	if err != nil {
		log.Fatalf("Unable to retrieve mylibrary bookshelves: %v", err)
	}

	if len(bs.Items) == 0 {
		log.Fatal("You have no bookshelves to explore.")
	}
	for _, b := range bs.Items {
		// Note that sometimes VolumeCount is not populated, so it may erroneously say '0'.
		log.Printf("You have %v books on bookshelf %q:", b.VolumeCount, b.Title)

		// List the volumes on this shelf.
		vol, err := svc.Mylibrary.Bookshelves.Volumes.List(strconv.FormatInt(b.Id, 10)).Do()
		if err != nil {
			log.Fatalf("Unable to retrieve mylibrary bookshelf volumes: %v", err)
		}
		for _, v := range vol.Items {
			var info struct {
				Text  bool `json:"text"`
				Image bool `json:"image"`
			}
			// Parse the additional fields, but ignore errors, as the information is not critical.
			googleapi.ConvertVariant(v.VolumeInfo.ReadingModes.(map[string]interface{}), &info)
			var s []string
			if info.Text {
				s = append(s, "text")
			}
			if info.Image {
				s = append(s, "image")
			}
			extra := fmt.Sprintf("; formats: %v", s)
			if v.VolumeInfo.ImageLinks != nil {
				extra += fmt.Sprintf("; thumbnail: %v", v.VolumeInfo.ImageLinks.Thumbnail)
			}
			log.Printf("  %q by %v%v", v.VolumeInfo.Title, strings.Join(v.VolumeInfo.Authors, ", "), extra)
		}
	}
}
