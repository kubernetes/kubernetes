package main

import (
	"log"
	"net/http"

	tasks "code.google.com/p/google-api-go-client/tasks/v1"
)

func init() {
	registerDemo("tasks", tasks.TasksScope, tasksMain)
}

func tasksMain(client *http.Client, argv []string) {
	taskapi, _ := tasks.New(client)
	task, err := taskapi.Tasks.Insert("@default", &tasks.Task{
		Title: "finish this API code generator thing",
		Notes: "ummmm",
		Due:   "2011-10-15T12:00:00.000Z",
	}).Do()
	log.Printf("Got task, err: %#v, %v", task, err)
}
