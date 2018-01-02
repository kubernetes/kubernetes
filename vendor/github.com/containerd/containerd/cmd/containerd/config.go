package main

import (
	"os"

	"github.com/urfave/cli"
)

var configCommand = cli.Command{
	Name:  "config",
	Usage: "information on the containerd config",
	Subcommands: []cli.Command{
		{
			Name:  "default",
			Usage: "see the output of the default config",
			Action: func(context *cli.Context) error {
				_, err := defaultConfig().WriteTo(os.Stdout)
				return err
			},
		},
	},
}
