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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	builderv2 "k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/util"
	"k8s.io/kube-openapi/test/integration/pkg/generated"
)

// TODO: Change this to output the generated swagger to stdout.
const defaultSwaggerFile = "generated.json"

func main() {
	// Get the name of the generated swagger file from the args
	// if it exists; otherwise use the default file name.
	swaggerFilename := defaultSwaggerFile
	if len(os.Args) > 1 {
		swaggerFilename = os.Args[1]
	}

	// Generate the definition names from the map keys returned
	// from GetOpenAPIDefinitions. Anonymous function returning empty
	// Ref is not used.
	var defNames []string
	for name, _ := range generated.GetOpenAPIDefinitions(func(name string) spec.Ref {
		return spec.Ref{}
	}) {
		defNames = append(defNames, name)
	}

	// Create a minimal builder config, then call the builder with the definition names.
	config := createOpenAPIBuilderConfig()
	config.GetDefinitions = generated.GetOpenAPIDefinitions
	// Build the Paths using a simple WebService for the final spec
	swagger, serr := builderv2.BuildOpenAPISpec(createWebServices(), config)
	if serr != nil {
		log.Fatalf("ERROR: %s", serr.Error())
	}

	// Marshal the swagger spec into JSON, then write it out.
	specBytes, err := json.MarshalIndent(swagger, " ", " ")
	if err != nil {
		log.Fatalf("json marshal error: %s", err.Error())
	}
	err = ioutil.WriteFile(swaggerFilename, specBytes, 0644)
	if err != nil {
		log.Fatalf("stdout write error: %s", err.Error())
	}
}

// CreateOpenAPIBuilderConfig hard-codes some values in the API builder
// config for testing.
func createOpenAPIBuilderConfig() *common.Config {
	return &common.Config{
		ProtocolList:   []string{"https"},
		IgnorePrefixes: []string{"/swaggerapi"},
		Info: &spec.Info{
			InfoProps: spec.InfoProps{
				Title:   "Integration Test",
				Version: "1.0",
			},
		},
		ResponseDefinitions: map[string]spec.Response{
			"NotFound": spec.Response{
				ResponseProps: spec.ResponseProps{
					Description: "Entity not found.",
				},
			},
		},
		CommonResponses: map[int]spec.Response{
			404: *spec.ResponseRef("#/responses/NotFound"),
		},
	}
}

// createWebServices hard-codes a simple WebService which only defines a GET path
// for testing.
func createWebServices() []*restful.WebService {
	w := new(restful.WebService)
	w.Route(buildRouteForType(w, "dummytype", "Foo"))
	w.Route(buildRouteForType(w, "dummytype", "Bar"))
	w.Route(buildRouteForType(w, "dummytype", "Baz"))
	w.Route(buildRouteForType(w, "dummytype", "Waldo"))
	w.Route(buildRouteForType(w, "listtype", "AtomicList"))
	w.Route(buildRouteForType(w, "listtype", "MapList"))
	w.Route(buildRouteForType(w, "listtype", "SetList"))
	w.Route(buildRouteForType(w, "uniontype", "TopLevelUnion"))
	w.Route(buildRouteForType(w, "uniontype", "InlinedUnion"))
	w.Route(buildRouteForType(w, "custom", "Bal"))
	w.Route(buildRouteForType(w, "custom", "Bak"))
	w.Route(buildRouteForType(w, "custom", "Bac"))
	w.Route(buildRouteForType(w, "custom", "Bah"))
	w.Route(buildRouteForType(w, "maptype", "GranularMap"))
	w.Route(buildRouteForType(w, "maptype", "AtomicMap"))
	w.Route(buildRouteForType(w, "structtype", "GranularStruct"))
	w.Route(buildRouteForType(w, "structtype", "AtomicStruct"))
	w.Route(buildRouteForType(w, "structtype", "DeclaredAtomicStruct"))
	w.Route(buildRouteForType(w, "defaults", "Defaulted"))
	return []*restful.WebService{w}
}

// Implements OpenAPICanonicalTypeNamer
var _ = util.OpenAPICanonicalTypeNamer(&typeNamer{})

type typeNamer struct {
	pkg  string
	name string
}

func (t *typeNamer) OpenAPICanonicalTypeName() string {
	return fmt.Sprintf("k8s.io/kube-openapi/test/integration/testdata/%s.%s", t.pkg, t.name)
}

func buildRouteForType(ws *restful.WebService, pkg, name string) *restful.RouteBuilder {
	namer := typeNamer{
		pkg:  pkg,
		name: name,
	}
	return ws.GET(fmt.Sprintf("test/%s/%s", pkg, strings.ToLower(name))).
		To(func(*restful.Request, *restful.Response) {}).
		Writes(&namer)
}
