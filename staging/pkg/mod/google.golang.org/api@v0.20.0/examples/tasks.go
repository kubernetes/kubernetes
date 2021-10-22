// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
