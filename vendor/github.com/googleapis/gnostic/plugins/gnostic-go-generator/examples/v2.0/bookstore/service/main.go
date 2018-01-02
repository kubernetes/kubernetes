// +build !appengine

// This file is omitted when the app is built for Google App Engine

/*
 Copyright 2017 Google Inc. All Rights Reserved.

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

package main

import (
	"log"

	"github.com/googleapis/gnostic/plugins/gnostic-go-generator/examples/v2.0/bookstore/bookstore"
)

func main() {
	err := bookstore.ServeHTTP(":8080")
	if err != nil {
		log.Printf("%v", err)
	}
}
