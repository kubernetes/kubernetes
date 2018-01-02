// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:generate encode-templates

// gnostic_go_generator is a sample Gnostic plugin that generates Go
// code that supports an API.
package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/golang/protobuf/proto"

	openapi "github.com/googleapis/gnostic/OpenAPIv2"
	plugins "github.com/googleapis/gnostic/plugins"
)

// Helper: if error is not nil, record it, serializes and returns the response and exits
func sendAndExitIfError(err error, response *plugins.Response) {
	if err != nil {
		response.Errors = append(response.Errors, err.Error())
		sendAndExit(response)
	}
}

// Helper: serializes and returns the response
func sendAndExit(response *plugins.Response) {
	responseBytes, _ := proto.Marshal(response)
	os.Stdout.Write(responseBytes)
	os.Exit(0)
}

// This is the main function for the code generation plugin.
func main() {

	// Use the name used to run the plugin to decide which files to generate.
	var files []string
	switch os.Args[0] {
	case "gnostic_go_client":
		files = []string{"client.go", "types.go"}
	case "gnostic_go_server":
		files = []string{"server.go", "provider.go", "types.go"}
	default:
		files = []string{"client.go", "server.go", "provider.go", "types.go"}
	}

	// Initialize the plugin response.
	response := &plugins.Response{}

	// Read the plugin input.
	data, err := ioutil.ReadAll(os.Stdin)
	sendAndExitIfError(err, response)
	if len(data) == 0 {
		sendAndExitIfError(fmt.Errorf("no input data"), response)
	}

	// Deserialize the input.
	request := &plugins.Request{}
	err = proto.Unmarshal(data, request)
	sendAndExitIfError(err, response)

	// Collect parameters passed to the plugin.
	invocation := os.Args[0]
	parameters := request.Parameters
	packageName := request.OutputPath // the default package name is the output directory
	for _, parameter := range parameters {
		invocation += " " + parameter.Name + "=" + parameter.Value
		if parameter.Name == "package" {
			packageName = parameter.Value
		}
	}

	// Log the invocation.
	log.Printf("Running %s(input:%s)", invocation, request.Wrapper.Version)

	// Read the document sent by the plugin and use it to generate client/server code.
	if request.Wrapper.Version != "v2" {
		err = fmt.Errorf("Unsupported OpenAPI version %s", request.Wrapper.Version)
		sendAndExitIfError(err, response)
	}
	document := &openapi.Document{}
	err = proto.Unmarshal(request.Wrapper.Value, document)
	sendAndExitIfError(err, response)

	// Create the renderer.
	renderer, err := NewServiceRenderer(document, packageName)
	sendAndExitIfError(err, response)

	// Run the renderer to generate files and add them to the response object.
	err = renderer.Generate(response, files)
	sendAndExitIfError(err, response)

	// Return with success.
	sendAndExit(response)
}
