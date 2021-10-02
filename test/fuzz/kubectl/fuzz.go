//go:build gofuzz
// +build gofuzz

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

package kubectl

import (
	fuzz "github.com/AdaLogics/go-fuzz-headers"
	"io"
	envutil "k8s.io/kubectl/pkg/cmd/set/env"
	"k8s.io/kubectl/pkg/util/certificate"
)

// FuzzParseCSR implements a fuzzer
// that targets certificate.ParseCSR
func FuzzParseCSR(data []byte) int {
	_, _ = certificate.ParseCSR(data)
	return 1
}

// FuzzParseEnv implements a fuzzer
// that targets envutil.ParseEnv
func FuzzParseEnv(data []byte) int {
	f := fuzz.NewConsumer(data)

	// Create a pseudo-random spec.
	// Will be used as argument to the fuzz target

	// length of slice:
	qty, err := f.GetInt()
	if err != nil {
		return 0
	}
	spec := make([]string, qty, qty)

	// fill slice with values
	for i := 0; i < qty; i++ {
		s, err := f.GetString()
		if err != nil {
			return 0
		}
		spec = append(spec, s)
	}
	var r io.Reader
	_, _, _, _ = envutil.ParseEnv(spec, r)
	return 1
}
