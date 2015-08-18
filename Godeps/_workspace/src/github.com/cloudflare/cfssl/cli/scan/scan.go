package scan

import (
	"encoding/json"
	"fmt"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/scan"
)

var scanUsageText = `cfssl scan -- scan a host for issues
Usage of scan:
        cfssl scan [-family regexp] [-scanner regexp] [-timeout duration] HOST+
        cfssl scan -list

Arguments:
        HOST:    Host(s) to scan (including port)
Flags:
`
var scanFlags = []string{"list", "family", "scanner", "timeout"}

func printJSON(v interface{}) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("%s\n\n", b)
}

func scanMain(args []string, c cli.Config) (err error) {
	if c.List {
		printJSON(scan.Default)
	} else {
		// Execute for each HOST argument given
		for len(args) > 0 {
			var host string
			host, args, err = cli.PopFirstArgument(args)
			if err != nil {
				return
			}

			fmt.Printf("Scanning %s...\n", host)

			var results map[string]scan.FamilyResult
			results, err = scan.Default.RunScans(host, c.Family, c.Scanner, c.Timeout)
			if err != nil {
				return
			}
			if results != nil {
				printJSON(results)
			}
		}
	}
	return
}

// Command assembles the definition of Command 'scan'
var Command = &cli.Command{UsageText: scanUsageText, Flags: scanFlags, Main: scanMain}
