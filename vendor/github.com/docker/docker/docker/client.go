// +build !daemon

package main

import (
	"log" // see gh#8745, client needs to use go log pkg
)

func mainDaemon() {
	log.Fatal("This is a client-only binary - running the Docker daemon is not supported.")
}
