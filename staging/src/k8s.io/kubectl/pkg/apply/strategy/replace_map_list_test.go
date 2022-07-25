/*
Copyright 2017 The Kubernetes Authors.

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

package strategy_test

import (
	. "github.com/onsi/ginkgo/v2"

	"k8s.io/kubectl/pkg/apply/strategy"
)

var _ = Describe("Replacing fields of type list without openapi", func() {
	Context("where a field is has been updated", func() {
		It("should replace the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  bar:
  - name: bar1
    value: 1
  - name: bar2
    value: 2
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  bar:
  - name: bar1
    value: 1
  - name: bar2
    value: 2
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
  bar:
  - name: bar1
    value: 1
  - name: bar3
    value: 3
  - name: bar4
    value: 4
`)
			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  bar:
  - name: bar1
    value: 1
  - name: bar2
    value: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})
