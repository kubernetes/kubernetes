package main

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/codegangsta/cli"
)

var configCommand = cli.Command{
	Name:   "config",
	Usage:  "display the container configuration",
	Action: configAction,
}

func configAction(context *cli.Context) {
	container, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	data, err := json.MarshalIndent(container, "", "\t")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s", data)
}
