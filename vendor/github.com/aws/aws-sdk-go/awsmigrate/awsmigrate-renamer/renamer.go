// +build go1.5,deprecated

package main

//go:generate go run -tags deprecated gen/gen.go

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/aws/aws-sdk-go/awsmigrate/awsmigrate-renamer/rename"
)

var safeTag = "4e554f77f00d527b452c68a46f2e68595284121b"

func main() {
	gopath := os.Getenv("GOPATH")
	if gopath == "" {
		panic("GOPATH not set!")
	}
	gopath = strings.Split(gopath, ":")[0]

	// change directory to SDK
	err := os.Chdir(filepath.Join(gopath, "src", "github.com", "aws", "aws-sdk-go"))
	if err != nil {
		panic("Cannot find SDK repository")
	}

	// store orig HEAD
	head, err := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD").Output()
	if err != nil {
		panic("Cannot find SDK repository")
	}
	origHEAD := strings.Trim(string(head), " \r\n")

	// checkout to safe tag and run conversion
	exec.Command("git", "checkout", safeTag).Run()
	defer func() {
		exec.Command("git", "checkout", origHEAD).Run()
	}()

	rename.ParsePathsFromArgs()
}
