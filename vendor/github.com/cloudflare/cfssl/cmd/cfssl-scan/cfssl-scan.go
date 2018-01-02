package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/cli/scan"
	"github.com/cloudflare/cfssl/config"
)

// main defines the scan usage and registers all defined commands and flags.
func main() {

	var scanFlagSet = flag.NewFlagSet("scan", flag.ExitOnError)
	var c cli.Config
	var usageText = `cfssl scan -- scan a host for issues
Usage of scan:
        cfssl scan [-family regexp] [-scanner regexp] [-timeout duration] [-ip IPAddr] [-num-workers num] [-max-hosts num] [-csv hosts.csv] HOST+
        cfssl scan -list

Arguments:
        HOST:    Host(s) to scan (including port)
Flags:
`
	registerFlags(&c, scanFlagSet)

	scanFlagSet.Usage = func() {
		fmt.Fprintf(os.Stderr, "\t%s", usageText)
		for _, name := range scan.Command.Flags {
			if f := scanFlagSet.Lookup(name); f != nil {
				printDefaultValue(f)
			}
		}
	}
	args := os.Args[1:]
	scanFlagSet.Parse(args)
	args = scanFlagSet.Args()

	var err error
	c.CFG, err = config.LoadFile(c.ConfigFile)
	if c.ConfigFile != "" && err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config file: %v", err)
	}

	if err := scan.Command.Main(args, c); err != nil {
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
	f.BoolVar(&c.List, "list", false, "list possible scanners")
	f.StringVar(&c.Family, "family", "", "scanner family regular expression")
	f.StringVar(&c.Scanner, "scanner", "", "scanner regular expression")
	f.DurationVar(&c.Timeout, "timeout", 5*time.Minute, "duration (ns, us, ms, s, m, h) to scan each host before timing out")
	f.StringVar(&c.CSVFile, "csv", "", "file containing CSV of hosts")
	f.IntVar(&c.NumWorkers, "num-workers", 10, "number of workers to use for scan")
	f.IntVar(&c.MaxHosts, "max-hosts", 100, "maximum number of hosts to scan")
	f.StringVar(&c.IP, "ip", "", "remote server ip")
	f.StringVar(&c.CABundleFile, "ca-bundle", "", "path to root certificate store")
}
