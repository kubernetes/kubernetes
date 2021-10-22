// Copyright 2018 Google LLC. All Rights Reserved.
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

// gnostic-lint-paths is a tool for analyzing paths in OpenAPI descriptions.
//
// It scans an API description and checks it against a set of coding style guidelines.
package main

import (
	"github.com/golang/protobuf/proto"
	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
	plugins "github.com/googleapis/gnostic/plugins"
)

func checkPathsV2(document *openapiv2.Document, messages []*plugins.Message) []*plugins.Message {
	for _, pair := range document.Paths.Path {
		messages = append(messages,
			&plugins.Message{
				Level: plugins.Message_INFO,
				Code:  "PATH",
				Text:  pair.Name,
				Keys:  []string{"paths", pair.Name}})
	}
	return messages
}

func checkPathsV3(document *openapiv3.Document, messages []*plugins.Message) []*plugins.Message {
	for _, pair := range document.Paths.Path {
		messages = append(messages,
			&plugins.Message{
				Level: plugins.Message_INFO,
				Code:  "PATH",
				Text:  pair.Name,
				Keys:  []string{"paths", pair.Name}})
	}
	return messages
}

func main() {
	env, err := plugins.NewEnvironment()
	env.RespondAndExitIfError(err)

	messages := make([]*plugins.Message, 0, 0)

	for _, model := range env.Request.Models {
		switch model.TypeUrl {
		case "openapi.v2.Document":
			documentv2 := &openapiv2.Document{}
			err = proto.Unmarshal(model.Value, documentv2)
			if err == nil {
				messages = checkPathsV2(documentv2, messages)
			}
		case "openapi.v3.Document":
			documentv3 := &openapiv3.Document{}
			err = proto.Unmarshal(model.Value, documentv3)
			if err == nil {
				messages = checkPathsV3(documentv3, messages)
			}
		}
	}

	env.RespondAndExitIfError(err)
	env.Response.Messages = messages
	env.RespondAndExit()
}
