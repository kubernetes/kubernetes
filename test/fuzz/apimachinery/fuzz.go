/*
Copyright 2022 The Kubernetes Authors.

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

package apimachinery

import (
	fuzz "github.com/AdaLogics/go-fuzz-headers"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
)

// FuzzParseQuantity implements a fuzzer
// that targets resource.ParseQuantity
func FuzzParseQuantity(data []byte) int {
	_, _ = resource.ParseQuantity(string(data))
	return 1
}

// FuzzMeta1ParseToLabelSelector implements a fuzzer
// that targets metav1.ParseToLabelSelector
func FuzzMeta1ParseToLabelSelector(data []byte) int {
	_, _ = metav1.ParseToLabelSelector(string(data))
	return 1
}

// FuzzParseSelector implements a fuzzer
// that targets fields.ParseSelector
func FuzzParseSelector(data []byte) int {
	_, _ = fields.ParseSelector(string(data))
	return 1
}

// FuzzLabelsParse implements a fuzzer
// that targets labels.Parse
func FuzzLabelsParse(data []byte) int {
	_, _ = labels.Parse(string(data))
	return 1
}

// FuzzParseGroupVersion implements a fuzzer
// that targets schema.ParseGroupVersion
func FuzzParseGroupVersion(data []byte) int {
	_, _ = schema.ParseGroupVersion(string(data))
	return 1
}

// FuzzParseResourceArg implements a fuzzer
// that targets schema.ParseResourceArg
func FuzzParseResourceArg(data []byte) int {
	_, _ = schema.ParseResourceArg(string(data))
	return 1
}

// FuzzParseVersion implements a fuzzer
// that targets:
// - version.ParseSemantic,
// - version/(*Version).String()
// - version.ParseGeneric
// - version/(*Version).AtLeast(*Version)
func FuzzParseVersion(data []byte) int {
	f := fuzz.NewConsumer(data)
	vString1, err := f.GetString()
	if err != nil {
		return 0
	}
	v1, err := version.ParseSemantic(vString1)
	if err != nil {
		return 0
	}

	// Test if the Version will crash (*Version).String()
	_ = v1.String()

	vString2, err := f.GetString()
	if err != nil {
		return 0
	}
	v2, err := version.ParseGeneric(vString2)
	if err != nil {
		return 0
	}
	_ = v1.AtLeast(v2)
	return 1
}
