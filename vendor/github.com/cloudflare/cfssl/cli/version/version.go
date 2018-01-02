// Package version implements the version command.
package version

import (
	"fmt"
	"runtime"

	"github.com/cloudflare/cfssl/cli"
)

// Version stores the semantic versioning information for CFSSL.
var version = struct {
	Major    int
	Minor    int
	Patch    int
	Revision string
}{1, 2, 0, "release"}

func versionString() string {
	return fmt.Sprintf("%d.%d.%d", version.Major, version.Minor, version.Patch)
}

// Usage text for 'cfssl version'
var versionUsageText = `cfssl version -- print out the version of CF SSL

Usage of version:
	cfssl version
`

// The main functionality of 'cfssl version' is to print out the version info.
func versionMain(args []string, c cli.Config) (err error) {
	fmt.Printf("Version: %s\nRevision: %s\nRuntime: %s\n", versionString(), version.Revision, runtime.Version())
	return nil
}

// Command assembles the definition of Command 'version'
var Command = &cli.Command{UsageText: versionUsageText, Flags: nil, Main: versionMain}
