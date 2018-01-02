package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/cli/genkey"
	"github.com/cloudflare/cfssl/config"
)

// main defines the newkey usage and registers all defined commands and flags.
func main() {

	var newkeyFlagSet = flag.NewFlagSet("newkey", flag.ExitOnError)
	var c cli.Config

	var usageText = `cfssl-newkey -- generate a new key and CSR

	Usage of genkey:
        	newkey CSRJSON

	Arguments:
        	CSRJSON:    JSON file containing the request, use '-' for reading JSON from stdin

	Flags:
	`

	registerFlags(&c, newkeyFlagSet)

	newkeyFlagSet.Usage = func() {
		fmt.Fprintf(os.Stderr, "\t%s", usageText)
		for _, name := range genkey.Command.Flags {
			if f := newkeyFlagSet.Lookup(name); f != nil {
				printDefaultValue(f)
			}
		}
	}
	args := os.Args[1:]
	newkeyFlagSet.Parse(args)
	args = newkeyFlagSet.Args()

	var err error
	c.CFG, err = config.LoadFile(c.ConfigFile)
	if c.ConfigFile != "" && err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config file: %v", err)
	}

	if err := genkey.Command.Main(args, c); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}

// printDefaultValue is a helper function to print out a user friendly
// usage message of a flag. It's useful since we want to write customized
// usage message on selected subsets of the global flag set. It is
// borrowed from standard library source code. Since flag value type is
// not exported, default string flag values are printed without
// quotes. The only exception is the empty string, which is printed as "".
func printDefaultValue(f *flag.Flag) {
	format := "  -%s=%s: %s\n"
	if f.DefValue == "" {
		format = "  -%s=%q: %s\n"
	}
	fmt.Fprintf(os.Stderr, format, f.Name, f.DefValue, f.Usage)
}

// registerFlags defines all cfssl command flags and associates their values with variables.
func registerFlags(c *cli.Config, f *flag.FlagSet) {
	f.BoolVar(&c.IsCA, "initca", false, "initialise new CA")
	f.StringVar(&c.ConfigFile, "config", "", "path to configuration file")
}
