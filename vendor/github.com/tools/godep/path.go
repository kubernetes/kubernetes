package main

import (
	"fmt"
	"os"
)

var cmdPath = &Command{
	Name:  "path",
	Short: "print GOPATH for dependency code",
	Long: `
Command path prints a path for use in env var GOPATH
that makes available the specified version of each dependency.

The printed path does not include any GOPATH value from
the environment.

For more about how GOPATH works, see 'go help gopath'.
`,
	Run:          runPath,
	OnlyInGOPATH: true,
}

// Print the gopath that points to
// the included dependency code.
func runPath(cmd *Command, args []string) {
	if len(args) != 0 {
		cmd.UsageExit()
	}
	if VendorExperiment {
		fmt.Fprintln(os.Stderr, "Error: GO15VENDOREXPERIMENT is enabled and the vendor/ directory is not a valid Go workspace.")
		os.Exit(1)
	}
	gopath := prepareGopath()
	fmt.Println(gopath)
}
