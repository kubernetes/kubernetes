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

var _ = Describe("Merging fields of type map with openapi", func() {
	Context("where a field has been deleted", func() {
		It("should delete the field when it is the only field in the map", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - recorded/remote match
  paused: true
  # delete - recorded/remote differ
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - not present in recorded
  replicas: null
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
`)

			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should delete the field when there are other fields in the map", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - recorded/remote match
  paused: true
  # delete - recorded/remote differ
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - not present in recorded
  replicas: null
  # keep
  revisionHistoryLimit: 1
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
  revisionHistoryLimit: 1
`)

			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  revisionHistoryLimit: 1
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
  paused: true
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # Add this - it is missing from recorded and remote
  replicas: 3
  # Add this - it is missing from remote but matches recorded
  paused: true
  # Add this - it is missing from remote and differs from recorded
  progressDeadlineSeconds: 2
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
  paused: true
  progressDeadlineSeconds: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been updated", func() {
		It("should add the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  paused: true
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # Missing from recorded
  replicas: 3
  # Matches the recorded
  paused: true
  # Differs from recorded
  progressDeadlineSeconds: 2
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 2
  paused: false
  progressDeadlineSeconds: 3
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should update the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 2
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 2
`)

			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})

var _ = Describe("Merging fields of type map without openapi", func() {
	Context("where a field has been deleted", func() {
		It("should delete the field when it is the only field in the map", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  # delete - recorded/remote match
  paused: true
  # delete - recorded/remote differ
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  # delete - not present in recorded
  replicas: null
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
`)

			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should delete the field when there are other fields in the map", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  # delete - recorded/remote match
  paused: true
  # delete - recorded/remote differ
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  # delete - not present in recorded
  replicas: null
  # keep
  revisionHistoryLimit: 1
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
  revisionHistoryLimit: 1
`)

			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  revisionHistoryLimit: 1
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been added", func() {
		It("should add the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  paused: true
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  # Add this - it is missing from recorded and remote
  replicas: 3
  # Add this - it is missing from remote but matches recorded
  paused: true
  # Add this - it is missing from remote and differs from recorded
  progressDeadlineSeconds: 2
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
`)
			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been updated", func() {
		It("should add the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  paused: true
  progressDeadlineSeconds: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  # Matches recorded
  replicas: 3
  # Matches the recorded
  paused: true
  # Differs from recorded
  progressDeadlineSeconds: 2
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 2
  paused: false
  progressDeadlineSeconds: 3
`)
			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 3
  paused: true
  progressDeadlineSeconds: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should update the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 2
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 3
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 2
`)

			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  replicas: 3
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})

var _ = Describe("Merging fields of type map with openapi", func() {
	Context("where a field has been deleted", func() {
		It("should delete the field when it is the only field in the map", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - recorded/remote match
  foo: true
  # delete - recorded/remote differ
  bar: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - not present in recorded
  baz: null
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  baz: 3
  foo: true
  bar: 2
`)

			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should delete the field when there are other fields in the map", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - recorded/remote match
  foo: true
  # delete - recorded/remote differ
  bar: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # delete - not present in recorded
  baz: null
  # keep
  biz: 1
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  bar: 3
  foo: true
  baz: 2
  biz: 1
`)

			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  biz: 1
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
  foo: true
  biz: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # Add this - it is missing from recorded and remote
  baz: 3
  # Add this - it is missing from remote but matches recorded
  foo: true
  # Add this - it is missing from remote and differs from recorded
  biz: 2
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
  baz: 3
  foo: true
  biz: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been updated", func() {
		It("should add the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo: true
  baz: 1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  # Missing from recorded
  bar: 3
  # Matches the recorded
  foo: true
  # Differs from recorded
  baz: 2
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  bar: 2
  foo: false
  baz: 3
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  bar: 3
  foo: true
  baz: 2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})
