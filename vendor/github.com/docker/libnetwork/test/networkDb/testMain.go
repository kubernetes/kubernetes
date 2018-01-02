package main

import (
	"log"
	"os"

	"github.com/docker/libnetwork/test/networkDb/dbclient"
	"github.com/docker/libnetwork/test/networkDb/dbserver"
	"github.com/sirupsen/logrus"
)

func main() {
	logrus.Infof("Starting the image with these args: %v", os.Args)
	if len(os.Args) < 1 {
		log.Fatal("You need at least 1 argument [client/server]")
	}

	switch os.Args[1] {
	case "server":
		dbserver.Server(os.Args[2:])
	case "client":
		dbclient.Client(os.Args[2:])
	}
}
