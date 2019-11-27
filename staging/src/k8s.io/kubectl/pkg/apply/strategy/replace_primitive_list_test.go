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

var _ = Describe("Replacing fields of type list with openapi", func() {
	Context("where the field has been deleted", func() {
		It("should delete the field if present in recorded and missing from local.", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - z
        - "y"
`)

			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should delete the field if missing in recorded and set to null in local.", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
`)

			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - z
        - "y"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where the field is has been added", func() {
		It("should add the field when missing from recorded", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should add the field when even when present in recorded", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should add the field when the parent field is missing as well", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
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
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been updated", func() {
		It("should replace the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
        - z
        - "y"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should replace the field even if recorded matches", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
        - z
        - "y"
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})

		It("should replace the field even if the only change is ordering", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - e
        - c
        - f
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - f
        - e
        - c
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})

var _ = Describe("Replacing fields of type list with openapi for the type, but not the field", func() {
	Context("where a field is has been updated", func() {
		It("should replace the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
      otherstuff:
      - name: container1
        command:
        - e
        - f
        - g
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - a
        - b
        - c
        - z
        - "y"
      otherstuff:
      - name: container1
        command:
        - s
        - d
        - f
      - name: container2
        command:
        - h
        - i
        - j
      - name: container3
        command:
        - k
        - l
        - m
`)
			expected := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        command:
        - c
        - e
        - f
      otherstuff:
      - name: container1
        command:
        - e
        - f
        - g
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})

var _ = Describe("Replacing fields of type list without openapi", func() {
	Context("where the field has been deleted", func() {
		It("should delete the field.", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    command:
    - a
    - b
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    arguments:
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    command:
    - a
    - b
    - z
    - "y"
    # explicitly delete this
    arguments:
    - a
    - b
    - z
    - "y"
    # keep this
    env:
    - a
    - b
    - z
    - "y"
`)

			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    env:
    - a
    - b
    - z
    - "y"
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where the field is has been added", func() {
		It("should add the field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    command:
    - a
    - b
    - z
    - "y"
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    # missing from recorded - add
    command:
    - a
    - b
    - z
    - "y"
    # missing from recorded - add
    arguments:
    - c
    - d
    - q
    - w
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
  template:
    command:
    - a
    - b
    - z
    - "y"
    arguments:
    - c
    - d
    - q
    - w
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where a field is has been updated", func() {
		It("should replace field", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    command:
    - a
    - b
    - c
    env:
    - s
    - "t"
    - u
`)
			local := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    command:
    - a
    - b
    - z
    - "y"
    arguments:
    - c
    - d
    - q
    - w
    env:
    - s
    - "t"
    - u
`)
			remote := create(`
apiVersion: apps/v1
kind: Foo
spec:
spec:
  template:
    command:
    - a
    - b
    - c
    - z
    - "y"
    arguments:
    - c
    - d
    - i
    env:
    - u
    - s
    - "t"
`)
			expected := create(`
apiVersion: apps/v1
kind: Foo
spec:
  template:
    command:
    - a
    - b
    - z
    - "y"
    arguments:
    - c
    - d
    - q
    - w
    env:
    - s
    - "t"
    - u
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})
