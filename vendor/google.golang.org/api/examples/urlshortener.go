// Copyright 2017 Google Inc. All Rights Reserved.
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

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	urlshortener "google.golang.org/api/urlshortener/v1"
)

func init() {
	registerDemo("urlshortener", urlshortener.UrlshortenerScope, urlShortenerMain)
}

func urlShortenerMain(client *http.Client, argv []string) {
	if len(argv) != 1 {
		fmt.Fprintf(os.Stderr, "Usage: urlshortener http://goo.gl/xxxxx     (to look up details)\n")
		fmt.Fprintf(os.Stderr, "       urlshortener http://example.com/long (to shorten)\n")
		return
	}

	svc, err := urlshortener.New(client)
	if err != nil {
		log.Fatalf("Unable to create UrlShortener service: %v", err)
	}

	urlstr := argv[0]

	// short -> long
	if strings.HasPrefix(urlstr, "http://goo.gl/") || strings.HasPrefix(urlstr, "https://goo.gl/") {
		url, err := svc.Url.Get(urlstr).Do()
		if err != nil {
			log.Fatalf("URL Get: %v", err)
		}
		fmt.Printf("Lookup of %s: %s\n", urlstr, url.LongUrl)
		return
	}

	// long -> short
	url, err := svc.Url.Insert(&urlshortener.Url{
		Kind:    "urlshortener#url", // Not really needed
		LongUrl: urlstr,
	}).Do()
	if err != nil {
		log.Fatalf("URL Insert: %v", err)
	}
	fmt.Printf("Shortened %s => %s\n", urlstr, url.Id)
}
