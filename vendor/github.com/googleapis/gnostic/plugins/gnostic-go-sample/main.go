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

// gnostic_go_sample is a sample Gnostic plugin written in Go.
package main

import (
	"io/ioutil"
	"os"

	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/printer"

	openapi "github.com/googleapis/gnostic/OpenAPIv2"
	plugins "github.com/googleapis/gnostic/plugins"
)

// generate a simple report of an OpenAPI document's contents
func printDocument(code *printer.Code, document *openapi.Document) {
	code.Print("Swagger: %+v", document.Swagger)
	code.Print("Host: %+v", document.Host)
	code.Print("BasePath: %+v", document.BasePath)
	if document.Info != nil {
		code.Print("Info:")
		code.Indent()
		if document.Info.Title != "" {
			code.Print("Title: %s", document.Info.Title)
		}
		if document.Info.Description != "" {
			code.Print("Description: %s", document.Info.Description)
		}
		if document.Info.Version != "" {
			code.Print("Version: %s", document.Info.Version)
		}
		code.Outdent()
	}
	code.Print("Paths:")
	code.Indent()
	for _, pair := range document.Paths.Path {
		v := pair.Value
		if v.Get != nil {
			code.Print("GET %+v", pair.Name)
		}
		if v.Post != nil {
			code.Print("POST %+v", pair.Name)
		}
	}
	code.Outdent()
}

// record an error, then serialize and return the response
func sendAndExitIfError(err error, response *plugins.Response) {
	if err != nil {
		response.Errors = append(response.Errors, err.Error())
		sendAndExit(response)
	}
}

// serialize and return the response
func sendAndExit(response *plugins.Response) {
	responseBytes, _ := proto.Marshal(response)
	os.Stdout.Write(responseBytes)
	os.Exit(0)
}

func main() {
	// initialize the response
	response := &plugins.Response{}

	// read and deserialize the request
	data, err := ioutil.ReadAll(os.Stdin)
	sendAndExitIfError(err, response)

	request := &plugins.Request{}
	err = proto.Unmarshal(data, request)
	sendAndExitIfError(err, response)

	wrapper := request.Wrapper
	document := &openapi.Document{}
	err = proto.Unmarshal(wrapper.Value, document)
	sendAndExitIfError(err, response)

	// generate report
	code := &printer.Code{}
	code.Print("READING %s (%s)", wrapper.Name, wrapper.Version)
	printDocument(code, document)
	file := &plugins.File{}
	file.Name = "report.txt"
	file.Data = []byte(code.String())
	response.Files = append(response.Files, file)

	// send with success
	sendAndExit(response)
}
