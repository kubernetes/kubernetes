/*
Copyright 2019 The Kubernetes Authors.

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

// Package yaml implements fuzzers for yaml deserialization routines in
// Kubernetes. These targets are compatible with the github.com/dvyukov/go-fuzz
// fuzzing framework.
package yaml

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	sigyaml "sigs.k8s.io/yaml"
)

// FuzzDurationStrict is a fuzz target for strict-unmarshaling Duration defined
// in "k8s.io/apimachinery/pkg/apis/meta/v1". This target also checks that the
// unmarshaled result can be marshaled back to the input.
func FuzzDurationStrict(b []byte) int {
	var durationHolder struct {
		D metav1.Duration `json:"d"`
	}
	if err := sigyaml.UnmarshalStrict(b, &durationHolder); err != nil {
		return 0
	}
	result, err := sigyaml.Marshal(&durationHolder)
	if err != nil {
		panic(err)
	}
	// Result is in the format "d: <duration>\n", so strip off the trailing
	// newline and convert durationHolder.D to the expected format.
	resultStr := strings.TrimSpace(string(result[:]))
	inputStr := fmt.Sprintf("d: %s", durationHolder.D.Duration)
	if resultStr != inputStr {
		panic(fmt.Sprintf("result(%v) != input(%v)", resultStr, inputStr))
	}
	return 1
}

// FuzzMicroTimeStrict is a fuzz target for strict-unmarshaling MicroTime
// defined in "k8s.io/apimachinery/pkg/apis/meta/v1". This target also checks
// that the unmarshaled result can be marshaled back to the input.
func FuzzMicroTimeStrict(b []byte) int {
	var microTimeHolder struct {
		T metav1.MicroTime `json:"t"`
	}
	if err := sigyaml.UnmarshalStrict(b, &microTimeHolder); err != nil {
		return 0
	}
	result, err := sigyaml.Marshal(&microTimeHolder)
	if err != nil {
		panic(err)
	}
	// Result is in the format "t: <time>\n", so strip off the trailing
	// newline and convert microTimeHolder.T to the expected format. If
	// time is zero, the value is marshaled to "null".
	resultStr := strings.TrimSpace(string(result[:]))
	var inputStr string
	if microTimeHolder.T.Time.IsZero() {
		inputStr = "t: null"
	} else {
		inputStr = fmt.Sprintf("t: %s", microTimeHolder.T.Time)
	}
	if resultStr != inputStr {
		panic(fmt.Sprintf("result(%v) != input(%v)", resultStr, inputStr))
	}
	return 1
}

// FuzzSigYaml is a fuzz target for "sigs.k8s.io/yaml" unmarshaling.
func FuzzSigYaml(b []byte) int {
	t := struct{}{}
	m := map[string]interface{}{}
	var out int
	if err := sigyaml.Unmarshal(b, &m); err == nil {
		out = 1
	}
	if err := sigyaml.Unmarshal(b, &t); err == nil {
		out = 1
	}
	return out
}

// FuzzTimeStrict is a fuzz target for strict-unmarshaling Time defined in
// "k8s.io/apimachinery/pkg/apis/meta/v1". This target also checks that the
// unmarshaled result can be marshaled back to the input.
func FuzzTimeStrict(b []byte) int {
	var timeHolder struct {
		T metav1.Time `json:"t"`
	}
	if err := sigyaml.UnmarshalStrict(b, &timeHolder); err != nil {
		return 0
	}
	result, err := sigyaml.Marshal(&timeHolder)
	if err != nil {
		panic(err)
	}
	// Result is in the format "t: <time>\n", so strip off the trailing
	// newline and convert timeHolder.T to the expected format. If time is
	// zero, the value is marshaled to "null".
	resultStr := strings.TrimSpace(string(result[:]))
	var inputStr string
	if timeHolder.T.Time.IsZero() {
		inputStr = "t: null"
	} else {
		inputStr = fmt.Sprintf("t: %s", timeHolder.T.Time)
	}
	if resultStr != inputStr {
		panic(fmt.Sprintf("result(%v) != input(%v)", resultStr, inputStr))
	}
	return 1
}

// FuzzYamlV2 is a fuzz target for "gopkg.in/yaml.v2" unmarshaling.
func FuzzYamlV2(b []byte) int {
	t := struct{}{}
	m := map[string]interface{}{}
	var out int
	if err := yaml.Unmarshal(b, &m); err == nil {
		out = 1
	}
	if err := yaml.Unmarshal(b, &t); err == nil {
		out = 1
	}
	return out
}
