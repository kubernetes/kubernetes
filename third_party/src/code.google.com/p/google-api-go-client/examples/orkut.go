package main

import (
	"log"
	"net/http"

	orkut "code.google.com/p/google-api-go-client/orkut/v2"
)

func init() {
	registerDemo("orkut", orkut.OrkutScope, orkutMain)
}

func orkutMain(client *http.Client, argv []string) {
	orkutapi, _ := orkut.New(client)
	myBadges, err := orkutapi.Badges.List("me").Do()
	if err != nil {
		log.Println(err)
		return
	}

	log.Println("Listing my badges")
	for _, badge := range myBadges.Items {
		log.Printf("Got badge %v", badge.Caption)
	}

	myStream, err := orkutapi.Activities.List("me", "stream").Do()
	if err != nil {
		log.Println(err)
		return
	}
	log.Println("Showing my activity stream")
	for _, activity := range myStream.Items {
		log.Printf("Actor: %v; Content: %v", activity.Actor.DisplayName, activity.Object.Content)
	}
}
