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

package framework

import (
	"github.com/onsi/ginkgo"
)

type spec struct {
	text string
	body func()
}

var specs []spec

// DescribeWithTestContext is like ginkgo.Describe, but in contrast to
// ginkgo, the framework implementation will invoke the spec body at a
// time when TestContext has been fully initialized. A callback
// invoked via framework.Describe therefore can add or parameterize
// specs depending on the test configuration.
func DescribeWithTestContext(text string, body func()) bool {
	specs = append(specs, spec{text, body})

	// Always returns a value for use in
	// var _ = framework.DescribeWithContext(...).
	return true
}

func describeSpecs() {
	for _, spec := range specs {
		ginkgo.Describe(spec.text, spec.body)
	}
}
