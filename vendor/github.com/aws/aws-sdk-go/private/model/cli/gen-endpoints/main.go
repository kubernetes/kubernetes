// +build codegen

// Command gen-endpoints parses a JSON description of the AWS endpoint
// discovery logic and generates a Go file which returns an endpoint.
//
//     aws-gen-goendpoints apis/_endpoints.json aws/endpoints_map.go
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/aws/aws-sdk-go/aws/endpoints"
)

// Generates the endpoints from json description
//
// Args:
//  -model The definition file to use
//  -out The output file to generate
func main() {
	var modelName, outName string
	flag.StringVar(&modelName, "model", "", "Endpoints definition model")
	flag.StringVar(&outName, "out", "", "File to write generated endpoints to.")
	flag.Parse()

	if len(modelName) == 0 || len(outName) == 0 {
		exitErrorf("model and out both required.")
	}

	modelFile, err := os.Open(modelName)
	if err != nil {
		exitErrorf("failed to open model definition, %v.", err)
	}
	defer modelFile.Close()

	outFile, err := os.Create(outName)
	if err != nil {
		exitErrorf("failed to open out file, %v.", err)
	}
	defer func() {
		if err := outFile.Close(); err != nil {
			exitErrorf("failed to successfully write %q file, %v", outName, err)
		}
	}()

	if err := endpoints.CodeGenModel(modelFile, outFile); err != nil {
		exitErrorf("failed to codegen model, %v", err)
	}
}

func exitErrorf(msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, msg+"\n", args...)
	os.Exit(1)
}
