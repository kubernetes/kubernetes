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

	"k8s.io/kubernetes/pkg/kubectl/apply/strategy"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
	tst "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

var _ = Describe("Replacing fields of type map with openapi for some fields", func() {
	var resources openapi.Resources
	BeforeEach(func() {
		resources = tst.NewFakeResources("test_swagger.json")
	})

	Context("where a field is has been updated", func() {
		It("should update the field", func() {
			recorded := create(`
apiVersion: extensions/v1beta1
kind: ReplicaSet
spec:
  template:
    containers:
    - name: container1
      image: image1
`)
			local := create(`
apiVersion: extensions/v1beta1
kind: ReplicaSet
spec:
  template:
    containers:
    - name: container1
      image: image1
`)
			remote := create(`
apiVersion: extensions/v1beta1
kind: ReplicaSet
spec:
  template:
    containers:
    - name: container1
      image: image1
    - name: container2
      image: image2
    - name: container3
      image: image3
`)
			expected := create(`
apiVersion: extensions/v1beta1
kind: ReplicaSet
spec:
  template:
    containers:
    - name: container1
      image: image1
`)

			// Use modified swagger for ReplicaSet spec
			runWith(strategy.Create(strategy.Options{}), recorded, local, remote, expected, resources)
		})
	})
})
