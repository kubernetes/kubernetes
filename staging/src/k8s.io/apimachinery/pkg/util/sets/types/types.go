/*
Copyright 2015 The Kubernetes Authors.

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

// Package types just provides input types to the set generator. It also
// contains a "go generate" block.
// (You must first `go install k8s.io/kube-gen/cmd/set-gen`)
package types

//go:generate set-gen -i k8s.io/kubernetes/pkg/util/sets/types

type ReferenceSetTypes struct {
	// These types all cause files to be generated.
	// These types should be reflected in the ouput of
	// the "//pkg/util/sets:set-gen" genrule.
	a int64
	b int
	c byte
	d string
}
