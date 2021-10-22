/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"log"
	"os"

	"github.com/googleapis/gnostic/compiler"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/kube-openapi/pkg/schemaconv"
	"k8s.io/kube-openapi/pkg/util/proto"
)

func main() {
	if len(os.Args) != 1 {
		log.Fatal("this program takes input on stdin and writes output to stdout.")
	}

	var info yaml.MapSlice
	if err := yaml.NewDecoder(os.Stdin).Decode(&info); err != nil {
		log.Fatalf("error decoding stdin: %v", err)
	}

	document, err := openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
	if err != nil {
		log.Fatalf("error interpreting stdin: %v", err)
	}

	models, err := proto.NewOpenAPIData(document)
	if err != nil {
		log.Fatalf("error interpreting models: %v", err)
	}

	newSchema, err := schemaconv.ToSchema(models)
	if err != nil {
		log.Fatalf("error converting schema format: %v", err)
	}

	if err := yaml.NewEncoder(os.Stdout).Encode(newSchema); err != nil {
		log.Fatalf("error writing new schema: %v", err)
	}

}
