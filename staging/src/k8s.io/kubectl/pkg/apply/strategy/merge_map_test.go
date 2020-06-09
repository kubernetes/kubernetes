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
	. "github.com/onsi/ginkgo"

	"k8s.io/kubectl/pkg/apply/strategy"
)

var _ = Describe("Merging fields of type map with openapi for some fields", func() {
	Context("where a field has been deleted", func() {
		It("should delete the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo1:
    bar: "baz1"
    image: "1"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo2: null
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo1:
    bar: "baz1"
    image: "1"
  foo2:
    bar: "baz2"
    image: "2"
  foo3:
    bar: "baz3"
    image: "3"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo3:
    bar: "baz3"
    image: "3"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been added", func() {
		It("should add the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1:
    bar: "baz1"
    image: "1"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo1:
    bar: "baz1"
    image: "1"
  foo2:
    bar: "baz2"
    image: "2"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo1:
    bar: "baz1"
    image: "1"
  foo2:
    bar: "baz2"
    image: "2"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been updated", func() {
		It("should update the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1:
    bar: "baz1=1"
    image: "1-1"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo1:
    bar: "baz1-1"
    image: "1-1"
  foo2:
    bar: "baz2-1"
    image: "2-1"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 2
  foo1:
    bar: "baz1-0"
    image: "1-0"
  foo2:
    bar: "baz2-0"
    image: "2-0"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  foo1:
    bar: "baz1-1"
    image: "1-1"
  foo2:
    bar: "baz2-1"
    image: "2-1"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})
