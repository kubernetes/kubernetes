package main

import (
	"log"
	"os"

	"github.com/mrunalp/fileutils"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatalf("usage: gocp <src> <dest>")
	}

	fi, err := os.Stat(os.Args[1])
	if err != nil {
		log.Fatalf("failed to stat file :%v", err)
	}

	if fi.IsDir() {
		if err := fileutils.CopyDirectory(os.Args[1], os.Args[2]); err != nil {
			log.Fatalf("error copying %s to %s: %v", os.Args[1], os.Args[2], err)
		}
	} else {
		if err := fileutils.CopyFile(os.Args[1], os.Args[2]); err != nil {
			log.Fatalf("error copying %s to %s: %v", os.Args[1], os.Args[2], err)
		}
	}
}
