// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	youtube "google.golang.org/api/youtube/v3"
)

func init() {
	registerDemo("youtube", youtube.YoutubeUploadScope, youtubeMain)
}

// youtubeMain is an example that demonstrates calling the YouTube API.
// It is similar to the sample found on the Google Developers website:
// https://developers.google.com/youtube/v3/docs/videos/insert
// but has been modified slightly to fit into the examples framework.
//
// Example usage:
//   go build -o go-api-demo
//   go-api-demo -clientid="my-clientid" -secret="my-secret" youtube filename
func youtubeMain(client *http.Client, argv []string) {
	if len(argv) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: youtube filename")
		return
	}
	filename := argv[0]

	service, err := youtube.New(client)
	if err != nil {
		log.Fatalf("Unable to create YouTube service: %v", err)
	}

	upload := &youtube.Video{
		Snippet: &youtube.VideoSnippet{
			Title:       "Test Title",
			Description: "Test Description", // can not use non-alpha-numeric characters
			CategoryId:  "22",
		},
		Status: &youtube.VideoStatus{PrivacyStatus: "unlisted"},
	}

	// The API returns a 400 Bad Request response if tags is an empty string.
	upload.Snippet.Tags = []string{"test", "upload", "api"}

	call := service.Videos.Insert("snippet,status", upload)

	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Error opening %v: %v", filename, err)
	}
	defer file.Close()

	response, err := call.Media(file).Do()
	if err != nil {
		log.Fatalf("Error making YouTube API call: %v", err)
	}
	fmt.Printf("Upload successful! Video ID: %v\n", response.Id)
}
