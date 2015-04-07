package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/codegangsta/cli"
	"github.com/docker/libcontainer"
)

var statsCommand = cli.Command{
	Name:   "stats",
	Usage:  "display statistics for the container",
	Action: statsAction,
}

func statsAction(context *cli.Context) {
	container, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	state, err := libcontainer.GetState(dataPath)
	if err != nil {
		log.Fatal(err)
	}

	stats, err := libcontainer.GetStats(container, state)
	if err != nil {
		log.Fatal(err)
	}
	data, err := json.MarshalIndent(stats, "", "\t")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s", data)
}
