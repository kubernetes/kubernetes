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

package kubelet

import (
	fuzz "github.com/AdaLogics/go-fuzz-headers"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

// FuzzParseQOSReserve implements a fuzzer
// that targets cm.ParseQOSReserved
func FuzzParseQOSReserve(data []byte) int {
	f := fuzz.NewConsumer(data)

	// Create a pseudo-random map.
	// Will be used as argument to the fuzz target
	m := make(map[string]string)
	err := f.FuzzMap(&m)
	if err != nil {
		return 0
	}
	_, _ = cm.ParseQOSReserved(m)
	return 1
}

// FuzzParseCPUSet implements a fuzzer
// that targets:
// - cpuset.Parse
// - cpuset/(CPUSet).String
func FuzzParseCPUSet(data []byte) int {
	cs, err := cpuset.Parse(string(data))
	if err != nil {
		return 0
	}
	_ = cs.String()
	return 1
}
