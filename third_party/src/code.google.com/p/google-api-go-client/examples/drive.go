package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	drive "code.google.com/p/google-api-go-client/drive/v2"
)

func init() {
	registerDemo("drive", drive.DriveScope, driveMain)
}

func driveMain(client *http.Client, argv []string) {
	if len(argv) != 1 {
		fmt.Fprintln(os.Stderr, "Usage: drive filename (to upload a file)")
		return
	}

	service, _ := drive.New(client)
	filename := argv[0]

	goFile, err := os.Open(filename)
	if err != nil {
		log.Fatalf("error opening %q: %v", filename, err)
	}
	driveFile, err := service.Files.Insert(&drive.File{Title: filename}).Media(goFile).Do()
	log.Printf("Got drive.File, err: %#v, %v", driveFile, err)
}
