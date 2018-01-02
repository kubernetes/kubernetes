// Copyright 2015 go-swagger maintainers
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

package fmts

import "github.com/go-openapi/swag"

var (
	// YAMLMatcher matches yaml
	YAMLMatcher = swag.YAMLMatcher
	// YAMLToJSON converts YAML unmarshaled data into json compatible data
	YAMLToJSON = swag.YAMLToJSON
	// BytesToYAMLDoc converts raw bytes to a map[string]interface{}
	BytesToYAMLDoc = swag.BytesToYAMLDoc
	// YAMLDoc loads a yaml document from either http or a file and converts it to json
	YAMLDoc = swag.YAMLDoc
	// YAMLData loads a yaml document from either http or a file
	YAMLData = swag.YAMLData
)
