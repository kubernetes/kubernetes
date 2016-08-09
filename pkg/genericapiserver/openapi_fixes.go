/*
Copyright 2016 The Kubernetes Authors.

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

package genericapiserver

import (
	"bytes"
	"github.com/go-openapi/spec"
	"strings"
	"unicode"
)

// Functions in this file customize generated openAPI spec to make it a valid spec.
// In an ideal world, these function should not exists, however current implementation of routes
// (manly in api_installer.go) make it hard to fix all of these in a timely manner.

type pathOperationID struct {
	path string
	id   string
}

var opFix = map[pathOperationID]string{
	pathOperationID{"/api/v1/configmaps", "listNamespacedConfigMap"}:   "listAllNamespacesConfigMap",
	pathOperationID{"/api/v1/endpoints", "listNamespacedEndpoints"}:    "listAllNamespacesEndpoints",
	pathOperationID{"/api/v1/events", "listNamespacedEvent"}:           "listAllNamespacesEvent",
	pathOperationID{"/api/v1/limitranges", "listNamespacedLimitRange"}: "listAllNamespacesLimitRange",
}

// OpenAPIFixOperation tries to rename operation IDs to make OpenAPI spec valid. The best solution to repeated
// ops is to fix it in the source. If that is not possible, the opFix map would let customize the renaming. If the ID
// is not in the opFix map, the name will be concatenated with a string computed from the path. see inferNamePartFromPath
// for more information.
func OpenAPIFixOperation(path string, op *spec.Operation, repeated bool) {
	if strings.HasSuffix(path, "{path}") {
		op.ID = op.ID + "WithPath"
		return
	}
	if repeated {
		op.ID = op.ID + "For" + inferNamePartFromPath(path)
		return
	}
	fix, exists := opFix[pathOperationID{path: path, id: op.ID}]
	if !exists {
		return
	}
	op.ID = fix
}

// inferNamePartFromPath creates an string by parsing path, removing separators and convert the path to CamelCase.
func inferNamePartFromPath(path string) string {
	var out bytes.Buffer
	separators := map[rune]bool{
		'/': true,
		'.': true,
	}
	toUpper := true
	insideBrackets := false
	for _, chr := range path {
		switch {
		case insideBrackets:
			if chr == '}' {
				insideBrackets = false
			}
		case chr == '{':
			insideBrackets = true
		case separators[chr]:
			toUpper = true
		case toUpper:
			toUpper = false
			out.WriteRune(unicode.ToUpper(chr))
		default:
			out.WriteRune(chr)
		}
	}
	return out.String()
}

// OpenAPIFixPath rewrites path if necessary to make it compatible with open-api spec.
func OpenAPIFixPath(path string) string {
	if strings.HasSuffix(path, ":*}") {
		return path[:len(path)-3] + "}"
	}
	return path
}

// OpenAPIFixResponses makes sure we would have a status code response for each responses. If there is none, a default OK
// response will be added.
func OpenAPIFixResponses(responses *spec.Responses) {
	if len(responses.StatusCodeResponses) == 0 {
		responses.StatusCodeResponses[200] = spec.Response{
			ResponseProps: spec.ResponseProps{
				Description: "OK",
			},
		}
	}
}
