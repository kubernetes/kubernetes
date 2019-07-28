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

// Package defaultconfig provides the default
// transformer configurations
package defaultconfig

import (
	"bytes"
)

// GetDefaultFieldSpecs returns default fieldSpecs.
func GetDefaultFieldSpecs() []byte {
	configData := [][]byte{
		[]byte(namePrefixFieldSpecs),
		[]byte(commonLabelFieldSpecs),
		[]byte(commonAnnotationFieldSpecs),
		[]byte(namespaceFieldSpecs),
		[]byte(varReferenceFieldSpecs),
		[]byte(nameReferenceFieldSpecs),
	}
	return bytes.Join(configData, []byte("\n"))
}

// GetDefaultFieldSpecsAsMap returns default fieldSpecs
// as a string->string map.
func GetDefaultFieldSpecsAsMap() map[string]string {
	result := make(map[string]string)
	result["nameprefix"] = namePrefixFieldSpecs
	result["commonlabels"] = commonLabelFieldSpecs
	result["commonannotations"] = commonAnnotationFieldSpecs
	result["namespace"] = namespaceFieldSpecs
	result["varreference"] = varReferenceFieldSpecs
	result["namereference"] = nameReferenceFieldSpecs
	return result
}
