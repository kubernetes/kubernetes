// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

import (
	"bytes"
)

// GetDefaultFieldSpecs returns default fieldSpecs.
func GetDefaultFieldSpecs() []byte {
	configData := [][]byte{
		[]byte(namePrefixFieldSpecs),
		[]byte(nameSuffixFieldSpecs),
		[]byte(commonLabelFieldSpecs),
		[]byte(commonAnnotationFieldSpecs),
		[]byte(namespaceFieldSpecs),
		[]byte(varReferenceFieldSpecs),
		[]byte(nameReferenceFieldSpecs),
		[]byte(imagesFieldSpecs),
		[]byte(replicasFieldSpecs),
	}
	return bytes.Join(configData, []byte("\n"))
}

// GetDefaultFieldSpecsAsMap returns default fieldSpecs
// as a string->string map.
func GetDefaultFieldSpecsAsMap() map[string]string {
	result := make(map[string]string)
	result["nameprefix"] = namePrefixFieldSpecs
	result["namesuffix"] = nameSuffixFieldSpecs
	result["commonlabels"] = commonLabelFieldSpecs
	result["commonannotations"] = commonAnnotationFieldSpecs
	result["namespace"] = namespaceFieldSpecs
	result["varreference"] = varReferenceFieldSpecs
	result["namereference"] = nameReferenceFieldSpecs
	result["images"] = imagesFieldSpecs
	result["replicas"] = replicasFieldSpecs
	return result
}
