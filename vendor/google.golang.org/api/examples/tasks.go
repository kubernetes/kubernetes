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
	"log"
	"net/http"

	tasks "google.golang.org/api/tasks/v1"
)

func init() {
	registerDemo("tasks", tasks.TasksScope, tasksMain)
}

func tasksMain(client *http.Client, argv []string) {
	taskapi, err := tasks.New(client)
	if err != nil {
		log.Fatalf("Unable to create Tasks service: %v", err)
	}

	task, err := taskapi.Tasks.Insert("@default", &tasks.Task{
		Title: "finish this API code generator thing",
		Notes: "ummmm",
		Due:   "2011-10-15T12:00:00.000Z",
	}).Do()
	log.Printf("Got task, err: %#v, %v", task, err)
}
