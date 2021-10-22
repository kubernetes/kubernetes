// Copyright 2017 Google LLC. All Rights Reserved.
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

// gnostic_analyze is a tool for analyzing OpenAPI descriptions.
//
// It scans an API description and evaluates properties
// that influence the ease and quality of code generation.
//  - The number of HTTP operations of each method (GET, POST, etc).
//  - The number of HTTP operations with no OperationId value.
//  - The parameter types used and their frequencies.
//  - The response types used and their frequencies.
//  - The types used in definition objects and arrays and their frequencies.
// Results are returned in a JSON structure.
package main

import (
	"encoding/json"
	"os"
	"path"
	"strings"

	"github.com/googleapis/gnostic/plugins/gnostic-analyze/statistics"

	"github.com/golang/protobuf/proto"
	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
	plugins "github.com/googleapis/gnostic/plugins"
)

// Record an error, then serialize and return a response.
func sendAndExitIfError(err error, response *plugins.Response) {
	if err != nil {
		response.Errors = append(response.Errors, err.Error())
		sendAndExit(response)
	}
}

// Serialize and return a response.
func sendAndExit(response *plugins.Response) {
	responseBytes, _ := proto.Marshal(response)
	os.Stdout.Write(responseBytes)
	os.Exit(0)
}

// This is the main function for the plugin.
func main() {
	env, err := plugins.NewEnvironment()
	env.RespondAndExitIfError(err)

	var stats *statistics.DocumentStatistics

	for _, model := range env.Request.Models {
		switch model.TypeUrl {
		case "openapi.v2.Document":
			documentv2 := &openapiv2.Document{}
			err = proto.Unmarshal(model.Value, documentv2)
			if err == nil {
				// Analyze the API document.
				stats = statistics.NewDocumentStatistics(env.Request.SourceName, documentv2)
			}
		case "openapi.v3.Document":
			documentv3 := &openapiv3.Document{}
			err = proto.Unmarshal(model.Value, documentv3)
			if err == nil {
				// Analyze the API document.
				stats = statistics.NewDocumentStatisticsV3(env.Request.SourceName, documentv3)
			}
		}
	}

	if stats != nil {
		// Return the analysis results with an appropriate filename.
		// Results are in files named "summary.json" in the same relative
		// locations as the description source files.
		file := &plugins.File{}
		file.Name = strings.Replace(stats.Name, path.Base(stats.Name), "summary.json", -1)
		file.Data, err = json.MarshalIndent(stats, "", "  ")
		file.Data = append(file.Data, []byte("\n")...)
		env.RespondAndExitIfError(err)
		env.Response.Files = append(env.Response.Files, file)
	}

	env.RespondAndExit()
}
