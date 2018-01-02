package cli

import (
	"github.com/codegangsta/cli"
)

func driverList(c *cli.Context) {
}

func driverAdd(c *cli.Context) {
}

// DriverCommands exports the list of CLI driver subcommands.
func DriverCommands() []cli.Command {
	commands := []cli.Command{
		{
			Name:    "add",
			Aliases: []string{"a"},
			Usage:   "add a new driver",
			Action:  driverAdd,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "name,n",
					Usage: "Driver Name",
				},
				cli.StringFlag{
					Name:  "options,o",
					Usage: "Comma separated name=value pairs, e.g disk=/dev/xvdg,mount=/var/openstorage/btrfs",
				},
			},
		},
		{
			Name:    "list",
			Aliases: []string{"l"},
			Usage:   "List drivers",
			Action:  driverList,
		},
	}
	return commands
}
