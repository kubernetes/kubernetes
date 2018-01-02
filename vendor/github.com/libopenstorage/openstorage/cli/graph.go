package cli

import (
	"fmt"

	"github.com/codegangsta/cli"
)

type graphDriver struct {
	name string
}

func (g *graphDriver) status(context *cli.Context) {
	fmt.Printf("Graph Driver %s is OK\n", g.name)
}

// GraphDriverCommands exports CLI comamnds for a GraphDriver.
func GraphDriverCommands(name string) []cli.Command {
	g := &graphDriver{name: name}

	graphCommands := []cli.Command{
		{
			Name:    "status",
			Aliases: []string{"s"},
			Usage:   "Status on  the usage of a Graph Driver",
			Action:  g.status,
		},
	}

	return graphCommands
}
