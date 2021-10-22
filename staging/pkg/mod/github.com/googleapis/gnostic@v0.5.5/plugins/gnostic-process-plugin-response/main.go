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

// gnostic-process-plugin-response is a development tool that processes
// the output of a gnostic plugin in the same way that it would be handled
// by gnostic itself.
package main

import (
	"flag"
	"io/ioutil"
	"log"
	"os"

	"github.com/golang/protobuf/proto"
	plugins "github.com/googleapis/gnostic/plugins"
)

func exitIfError(err error) {
	if err != nil {
		log.Printf("%+v", err)
		os.Exit(-1)
	}
}

func main() {
	output := flag.String("output", "-", "Output file or directory")
	flag.Parse()

	// Read the plugin response data from stdin.
	pluginData, err := ioutil.ReadAll(os.Stdin)
	exitIfError(err)
	response := &plugins.Response{}
	err = proto.Unmarshal(pluginData, response)
	exitIfError(err)

	// Handle the response in the standard (gnostic) way.
	err = plugins.HandleResponse(response, *output)
	exitIfError(err)
}
