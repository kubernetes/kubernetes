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

// gnostic-plugin-request is a development tool that captures and optionally
// displays the contents of the gnostic plugin interface.
package main

import (
	"log"

	"github.com/golang/protobuf/jsonpb"
	"github.com/golang/protobuf/proto"
	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
	plugins "github.com/googleapis/gnostic/plugins"
	surface "github.com/googleapis/gnostic/surface"
)

func main() {
	env, err := plugins.NewEnvironment()
	env.RespondAndExitIfError(err)

	if env.Verbose {
		for _, model := range env.Request.Models {
			log.Printf("model %s", model.TypeUrl)
			switch model.TypeUrl {
			case "openapi.v2.Document":
				document := &openapiv2.Document{}
				err = proto.Unmarshal(model.Value, document)
				if err == nil {
					log.Printf("%+v", document)
				}
			case "openapi.v3.Document":
				document := &openapiv3.Document{}
				err = proto.Unmarshal(model.Value, document)
				if err == nil {
					log.Printf("%+v", document)
				}
			case "surface.v1.Model":
				document := &surface.Model{}
				err = proto.Unmarshal(model.Value, document)
				if err == nil {
					log.Printf("%+v", document)
				}
			}
		}
	}

	// export the plugin request as JSON
	{
		file := &plugins.File{}
		file.Name = "plugin-request.json"
		m := jsonpb.Marshaler{Indent: " "}
		s, err := m.MarshalToString(env.Request)
		file.Data = []byte(s)
		env.RespondAndExitIfError(err)
		env.Response.Files = append(env.Response.Files, file)
	}
	// export the plugin request as binary protobuf
	{
		file := &plugins.File{}
		file.Name = "plugin-request.pb"
		file.Data, err = proto.Marshal(env.Request)
		env.RespondAndExitIfError(err)
		env.Response.Files = append(env.Response.Files, file)
	}
	env.RespondAndExit()
}
