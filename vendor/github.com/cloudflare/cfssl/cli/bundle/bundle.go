// Package bundle implements the bundle command.
package bundle

import (
	"errors"
	"fmt"

	"github.com/cloudflare/cfssl/bundler"
	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/ubiquity"
)

// Usage text of 'cfssl bundle'
var bundlerUsageText = `cfssl bundle -- create a certificate bundle that contains the client cert

Usage of bundle:
	- Bundle local certificate files
        cfssl bundle -cert file [-ca-bundle file] [-int-bundle file] [-int-dir dir] [-metadata file] [-key keyfile] [-flavor optimal|ubiquitous|force] [-password password]
	- Bundle certificate from remote server.
        cfssl bundle -domain domain_name [-ip ip_address] [-ca-bundle file] [-int-bundle file] [-int-dir dir] [-metadata file]

Flags:
`

// flags used by 'cfssl bundle'
var bundlerFlags = []string{"cert", "key", "ca-bundle", "int-bundle", "flavor", "int-dir", "metadata", "domain", "ip", "password"}

// bundlerMain is the main CLI of bundler functionality.
func bundlerMain(args []string, c cli.Config) (err error) {
	bundler.IntermediateStash = c.IntDir
	ubiquity.LoadPlatforms(c.Metadata)
	flavor := bundler.BundleFlavor(c.Flavor)
	var b *bundler.Bundler
	// If it is a force bundle, don't require ca bundle and intermediate bundle
	// Otherwise, initialize a bundler with CA bundle and intermediate bundle.
	if flavor == bundler.Force {
		b = &bundler.Bundler{}
	} else {
		b, err = bundler.NewBundler(c.CABundleFile, c.IntBundleFile)
		if err != nil {
			return
		}
	}

	var bundle *bundler.Bundle
	if c.CertFile != "" {
		if c.CertFile == "-" {
			var certPEM, keyPEM []byte
			certPEM, err = cli.ReadStdin(c.CertFile)
			if err != nil {
				return
			}
			if c.KeyFile != "" {
				keyPEM, err = cli.ReadStdin(c.KeyFile)
				if err != nil {
					return
				}
			}
			bundle, err = b.BundleFromPEMorDER(certPEM, keyPEM, flavor, "")
			if err != nil {
				return
			}
		} else {
			// Bundle the client cert
			bundle, err = b.BundleFromFile(c.CertFile, c.KeyFile, flavor, c.Password)
			if err != nil {
				return
			}
		}
	} else if c.Domain != "" {
		bundle, err = b.BundleFromRemote(c.Domain, c.IP, flavor)
		if err != nil {
			return
		}
	} else {
		return errors.New("Must specify bundle target through -cert or -domain")
	}

	marshaled, err := bundle.MarshalJSON()
	if err != nil {
		return
	}
	fmt.Printf("%s", marshaled)
	return
}

// Command assembles the definition of Command 'bundle'
var Command = &cli.Command{UsageText: bundlerUsageText, Flags: bundlerFlags, Main: bundlerMain}
