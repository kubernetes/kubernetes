package printdefaults

import (
	"fmt"

	"github.com/cloudflare/cfssl/cli"
)

var printDefaultsUsage = `cfssl print-defaults -- print default configurations that can be used as a template

Usage of print-defaults:
        cfssl print-defaults TYPE

If "list" is used as the TYPE, the list of supported types will be printed.
`

func printAvailable() {
	fmt.Println("Default configurations are available for:")
	for name := range defaults {
		fmt.Println("\t" + name)
	}
}

func printDefaults(args []string, c cli.Config) (err error) {
	arg, _, err := cli.PopFirstArgument(args)
	if err != nil {
		return
	}

	if arg == "list" {
		printAvailable()
	} else {
		if config, ok := defaults[arg]; !ok {
			printAvailable()
		} else {
			fmt.Println(config)
		}
	}

	return
}

// Command assembles the definition of Command 'print-defaults'
var Command = &cli.Command{
	UsageText: printDefaultsUsage,
	Flags:     []string{},
	Main:      printDefaults,
}
