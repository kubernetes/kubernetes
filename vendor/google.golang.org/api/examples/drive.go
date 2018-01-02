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

	drive "google.golang.org/api/drive/v2"
)

func init() {
	registerDemo("drive", drive.DriveScope, driveMain)
}

func driveMain(client *http.Client, argv []string) {
	if len(argv) != 1 {
		fmt.Fprintln(os.Stderr, "Usage: drive filename (to upload a file)")
		return
	}

	service, err := drive.New(client)
	if err != nil {
		log.Fatalf("Unable to create Drive service: %v", err)
	}

	filename := argv[0]

	goFile, err := os.Open(filename)
	if err != nil {
		log.Fatalf("error opening %q: %v", filename, err)
	}
	driveFile, err := service.Files.Insert(&drive.File{Title: filename}).Media(goFile).Do()
	log.Printf("Got drive.File, err: %#v, %v", driveFile, err)
}
