// Command aws-gen-goendpoints parses a JSON description of the AWS endpoint
// discovery logic and generates a Go file which returns an endpoint.
//
//     aws-gen-goendpoints apis/_endpoints.json aws/endpoints_map.go
package main

import (
	"encoding/json"
	"os"

	"github.com/aws/aws-sdk-go/private/model"
)

// Generates the endpoints from json description
//
// CLI Args:
//  [0] This file's execution path
//  [1] The definition file to use
//  [2] The output file to generate
func main() {
	in, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer in.Close()

	var endpoints struct {
		Version   int
		Endpoints map[string]struct {
			Endpoint      string
			SigningRegion string
		}
	}
	if err = json.NewDecoder(in).Decode(&endpoints); err != nil {
		panic(err)
	}

	out, err := os.Create(os.Args[2])
	if err != nil {
		panic(err)
	}
	defer out.Close()

	if err := model.GenerateEndpoints(endpoints, out); err != nil {
		panic(err)
	}
}
