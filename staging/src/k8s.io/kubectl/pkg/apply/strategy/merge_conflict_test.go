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

	"k8s.io/kubectl/pkg/apply/strategy"
)

var _ = Describe("Comparing fields of remote and recorded ", func() {
	Context("Test conflict in map fields of remote and recorded", func() {
		It("If conflicts found, expected return error", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1: "key1"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo2: "baz2-1"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1: "baz1-0"
`)

			expect := hasConflict
			// map fields have conflict : recorded {foo1 : "key1"}, remote {foo1 : "baz1-0"}
			runConflictTest(strategy.Create(strategy.Options{FailOnConflict: true}), recorded, local, remote, expect)
		})
	})

	Context("Test conflict in list fields of remote and recorded ", func() {
		It("If conflicts found, expected return false", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
metadata:
  finalizers:
  - "a"
  - "b"
  - "d"
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
  - "a"
  - "b"
  - "c"
`)
			expect := hasConflict
			// primatie lists have conflicts: recorded [a, b, d], remote [a, b, c]
			runConflictTest(strategy.Create(strategy.Options{FailOnConflict: true}), recorded, local, remote, expect)
		})
	})

	Context("Test conflict in Map-List fields of remote and recorded ", func() {
		It("should leave the item", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item1
        image: image1
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item2
        image: image2
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item1
          image: image3
`)
			expect := hasConflict
			// map list has conflict :  recorded {containers: [ {name: item1, image: image1} ]} , remote {containers: [ {name: item1, image: image3} ]}
			runConflictTest(strategy.Create(strategy.Options{FailOnConflict: true}), recorded, local, remote, expect)
		})
	})

	Context("Test conflicts in nested map field", func() {
		It("If conflicts found, expected return error", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1:
    name: "key1"
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1:
    name: "baz1-0"
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  foo1:
    name: "baz1-1"
`)
			expect := hasConflict
			// nested map has conflict : recorded {foo1: {name: "key1"}}, remote {foo1: {name : "baz1-1"}}
			runConflictTest(strategy.Create(strategy.Options{FailOnConflict: true}), recorded, local, remote, expect)
		})
	})

	Context("Test conflicts in complicated map, list", func() {
		It("Should catch conflict in key-value in map element", func() {
			recorded := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        ports:
        - containerPort: 8080
          protocol: TCP
          hostPort: 2020
        - containerPort: 8080
          protocol: UDP
          hostPort: 2022
`)
			local := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        ports:
        - containerPort: 8080
          protocol: TCP
          hostPort: 2020
`)
			remote := create(`
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        ports:
        - containerPort: 8080
          protocol: TCP
          hostPort: 2020
        - containerPort: 8080
          protocol: UDP
          hostPort: 2022
          hostIP: "127.0.0.1"
`)
			expect := noConflict
			runConflictTest(strategy.Create(strategy.Options{FailOnConflict: true}), recorded, local, remote, expect)
		})
	})
})
