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
)

var _ = Describe("Merging fields with the retainkeys strategy", func() {
	Context("where some fields are only defined remotely", func() {
		It("should drop those fields ", func() {
			recorded := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
`)
			local := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
    type: Recreate
`)
			remote := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
`)
			expected := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
    type: Recreate
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where some fields are defined both locally and remotely", func() {
		It("should merge those fields", func() {
			recorded := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
`)
			local := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 2
`)
			remote := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
`)
			expected := create(`
apiVersion: extensions/v1beta1
kind: Deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 2
      maxSurge: 1
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where the elements are in a list and some fields are only defined remotely", func() {
		It("should drop those fields ", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      volumes:
      - name: cache-volume
        emptyDir:
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      volumes:
      - name: cache-volume
        hostPath:
          path: /tmp/cache-volume
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      volumes:
      - name: cache-volume
        emptyDir:
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where the elements are in a list", func() {
		It("the fields defined both locally and remotely should be merged", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      volumes:
      - name: cache-volume
        hostPath:
          path: /tmp/cache-volume
        emptyDir:
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      volumes:
      - name: cache-volume
        hostPath:
          path: /tmp/cache-volume
          type: Directory
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      volumes:
      - name: cache-volume
        hostPath:
          path: /tmp/cache-volume
          type: Directory
        emptyDir:
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})
