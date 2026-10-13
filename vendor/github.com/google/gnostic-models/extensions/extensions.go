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

package gnostic_extension_v1

import (
	"io/ioutil"
	"log"
	"os"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

type extensionHandler func(name string, yamlInput string) (bool, proto.Message, error)

// Main implements the main program of an extension handler.
func Main(handler extensionHandler) {
	// unpack the request
	data, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		log.Println("File error:", err.Error())
		os.Exit(1)
	}
	if len(data) == 0 {
		log.Println("No input data.")
		os.Exit(1)
	}
	request := &ExtensionHandlerRequest{}
	err = proto.Unmarshal(data, request)
	if err != nil {
		log.Println("Input error:", err.Error())
		os.Exit(1)
	}
	// call the handler
	handled, output, err := handler(request.Wrapper.ExtensionName, request.Wrapper.Yaml)
	// respond with the output of the handler
	response := &ExtensionHandlerResponse{
		Handled: false, // default assumption
		Errors:  make([]string, 0),
	}
	if err != nil {
		response.Errors = append(response.Errors, err.Error())
	} else if handled {
		response.Handled = true
		response.Value, err = anypb.New(output)
		if err != nil {
			response.Errors = append(response.Errors, err.Error())
		}
	}
	responseBytes, _ := proto.Marshal(response)
	os.Stdout.Write(responseBytes)
}
