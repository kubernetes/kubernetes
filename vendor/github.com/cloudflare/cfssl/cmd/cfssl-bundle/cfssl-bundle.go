package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/cli/bundle"
	"github.com/cloudflare/cfssl/config"
)

// main defines the bundle usage and registers all defined commands and flags.
func main() {

	var bundleFlagSet = flag.NewFlagSet("bundle", flag.ExitOnError)
	var c cli.Config
	var usageText = `cfssl-bundle -- create a certificate bundle that contains the client cert

	Usage of bundle:
		- Bundle local certificate files
        	bundle -cert file [-ca-bundle file] [-int-bundle file] [-int-dir dir] [-metadata file] [-key keyfile] [-flavor optimal|ubiquitous|force] [-password password]
		- Bundle certificate from remote server.
        	bundle -domain domain_name [-ip ip_address] [-ca-bundle file] [-int-bundle file] [-int-dir dir] [-metadata file]

	Flags:
	`

	registerFlags(&c, bundleFlagSet)

	bundleFlagSet.Usage = func() {
		fmt.Fprintf(os.Stderr, "\t%s", usageText)
		for _, name := range bundle.Command.Flags {
			if f := bundleFlagSet.Lookup(name); f != nil {
				printDefaultValue(f)
			}
		}
	}
	args := os.Args[1:]
	bundleFlagSet.Parse(args)
	args = bundleFlagSet.Args()

	var err error
	c.CFG, err = config.LoadFile(c.ConfigFile)
	if c.ConfigFile != "" && err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load config file: %v", err)
	}

	if err := bundle.Command.Main(args, c); err != nil {
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
	f.StringVar(&c.CertFile, "cert", "", "Client certificate that contains the public key")
	f.StringVar(&c.KeyFile, "key", "", "private key for the certificate")
	f.StringVar(&c.CABundleFile, "ca-bundle", "", "path to root certificate store")
	f.StringVar(&c.IntBundleFile, "int-bundle", "", "path to intermediate certificate store")
	f.StringVar(&c.Flavor, "flavor", "ubiquitous", "Bundle Flavor: ubiquitous, optimal and force.")
	f.StringVar(&c.IntDir, "int-dir", "", "specify intermediates directory")
	f.StringVar(&c.Metadata, "metadata", "", "Metadata file for root certificate presence. The content of the file is a json dictionary (k,v): each key k is SHA-1 digest of a root certificate while value v is a list of key store filenames.")
	f.StringVar(&c.Domain, "domain", "", "remote server domain name")
	f.StringVar(&c.IP, "ip", "", "remote server ip")
	f.StringVar(&c.Password, "password", "0", "Password for accessing PKCS #12 data passed to bundler")
}
