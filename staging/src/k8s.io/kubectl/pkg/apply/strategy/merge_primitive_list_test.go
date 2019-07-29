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

var _ = Describe("Merging fields of type list-of-primitive with openapi", func() {
	Context("where one of the items has been deleted", func() {
		It("should delete the deleted item", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "c"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "c"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "c"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "c"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items is only on the remote", func() {
		It("should move the remote-only item to the end but keep it", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "c"
  - "b"
  - "a"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "c"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items is repeated", func() {
		It("should de-duplicate the repeated items", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "a"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where some items are deleted and others are on remote only", func() {
		It("should retain the correct items in the correct order", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "c"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "c"
  - "a"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "d"
  - "b"
  - "c"
  - "a"
  - "e"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "c"
  - "d"
  - "e"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})
