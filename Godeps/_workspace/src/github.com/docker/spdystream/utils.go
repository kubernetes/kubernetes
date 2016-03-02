package spdystream

import (
	"log"
	"os"
)

var (
	DEBUG = os.Getenv("DEBUG")
)

func debugMessage(fmt string, args ...interface{}) {
	if DEBUG != "" {
		log.Printf(fmt, args...)
	}
}
