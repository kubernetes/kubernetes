// Copyright 2020 Google LLC. All Rights Reserved.
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
package main

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/golang/protobuf/proto"
	lint "github.com/googleapis/gnostic/metrics/lint"
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

	var linter *lint.Linter

	for _, model := range env.Request.Models {
		switch model.TypeUrl {
		case "openapi.v2.Document":
			documentv2 := &openapiv2.Document{}
			err = proto.Unmarshal(model.Value, documentv2)
			if err == nil {
				// Analyze the API v2 document.
				linter, _ = lint.AIPLintV2(documentv2)
			}
		case "openapi.v3.Document":
			documentv3 := &openapiv3.Document{}
			err = proto.Unmarshal(model.Value, documentv3)
			if err == nil {
				// Analyze the API v3 document.
				linter, _ = lint.AIPLintV3(documentv3)
			}
		}
	}

	if linter != nil {
		file := &plugins.File{}
		file.Name = filepath.Join(
			filepath.Dir(env.Request.SourceName), "linter.json")
		file.Data, err = json.MarshalIndent(linter, "", "  ")
		env.RespondAndExitIfError(err)
		file.Data = append(file.Data, []byte("\n")...)
		env.Response.Files = append(env.Response.Files, file)

		file2 := &plugins.File{}
		file2.Name = filepath.Join(
			filepath.Dir(env.Request.SourceName), "linter.pb")
		file2.Data, err = proto.Marshal(linter)
		env.RespondAndExitIfError(err)
		env.Response.Files = append(env.Response.Files, file2)

	}

	env.RespondAndExit()
}
