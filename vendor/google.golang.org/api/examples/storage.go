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
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"google.golang.org/api/googleapi"
	storage "google.golang.org/api/storage/v1"
)

func init() {
	registerDemo("storage", storage.DevstorageReadWriteScope, storageMain)
}

func storageMain(client *http.Client, argv []string) {
	if len(argv) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: storage filename bucket (to upload an object)")
		return
	}

	service, err := storage.New(client)
	if err != nil {
		log.Fatalf("Unable to create Storage service: %v", err)
	}

	filename := argv[0]
	bucket := argv[1]

	goFile, err := os.Open(filename)
	if err != nil {
		log.Fatalf("error opening %q: %v", filename, err)
	}
	storageObject, err := service.Objects.Insert(bucket, &storage.Object{Name: filename}).Media(goFile).Do()
	log.Printf("Got storage.Object, err: %#v, %v", storageObject, err)
	if err != nil {
		return
	}

	resp, err := service.Objects.Get(bucket, filename).Download()
	if err != nil {
		log.Fatalf("error downloading %q: %v", filename, err)
	}
	defer resp.Body.Close()

	n, err := io.Copy(ioutil.Discard, resp.Body)
	if err != nil {
		log.Fatalf("error downloading %q: %v", filename, err)
	}

	log.Printf("Downloaded %d bytes", n)

	// Test If-None-Match - should get a "HTTP 304 Not Modified" response.
	obj, err := service.Objects.Get(bucket, filename).IfNoneMatch(storageObject.Etag).Do()
	log.Printf("Got obj, err: %#v, %v", obj, err)
	if googleapi.IsNotModified(err) {
		log.Printf("Success. Object not modified since upload.")
	} else {
		log.Printf("Error: expected object to not be modified since upload.")
	}
}
