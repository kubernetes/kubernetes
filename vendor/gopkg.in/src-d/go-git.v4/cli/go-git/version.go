package main

import "fmt"

var version string
var build string

type CmdVersion struct{}

func (c *CmdVersion) Execute(args []string) error {
	fmt.Printf("%s (%s) - build %s\n", bin, version, build)

	return nil
}
