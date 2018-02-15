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

package strategy_test

import (
	. "github.com/onsi/ginkgo"

	"k8s.io/kubernetes/pkg/kubectl/apply/strategy"
)

var _ = Describe("Comparing fields of remote and recorded ", func() {
	Context("Test conflict in map fields of remote and recorded", func() {
		It("If conflicts found, expected return false", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo1: "key1"
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo2: "baz2-1"
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo1: "baz1-0"
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo2: "baz2-1"
`)
			res := false
			runTest(strategy.CreateCompareStrategy(), recorded, local, remote, expected, res)
		})
	})

	Context("Test conflict in list fields of remote and recorded ", func() {
		It("If conflicts found, expected return false", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "c"
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "d"
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "c"
`)
			res := false
			runTest(strategy.CreateCompareStrategy(), recorded, local, remote, expected, res)
		})
	})

	Context("Test conflict in Map-List-Map nested fields of remote and recorded ", func() {
		It("If conflicts found, expected return false", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  containers:
  - finalizers: "bar1"
  - docker: "tst"
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  containers:
  - finalizers: "bar2"
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  containers:
  - finalizers: "bar1"
  - docker: "test"
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
metadata:
  - containers: "docker"
`)
			res := false
			runTest(strategy.CreateCompareStrategy(), recorded, local, remote, expected, res)
		})
	})

	Context("Test conflicts in nested map field", func() {
		It("If conflicts found, expected return false", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo1:
    name: "key1"
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo2:
    name: "baz2-1"
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo1:
    name: "baz1-0"
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  foo2: "baz2-1"
`)
			res := false
			runTest(strategy.CreateCompareStrategy(), recorded, local, remote, expected, res)
		})
	})

})
