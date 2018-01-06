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

var _ = Describe("Merging fields of type list-of-map with openapi", func() {
	Context("where one of the items has been deleted resulting in the containers being empty", func() {
		It("should set the containers field to null", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item
        image: image
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image
        - name: item2
          image: image2
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items has been deleted", func() {
		It("should be deleted from the result", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item-keep
        image: image-keep
      - name: item-delete
        image: image-delete
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item-keep
        image: image-keep
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item-keep
          image: image-keep
        - name: item-delete
          image: image-delete
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item-keep
        image: image-keep
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items is only in the remote", func() {
		It("should leave the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item2
        image: image2
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: item2
        image: image2
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item2
          image: image2
        - name: item
          image: image
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items differs from the remote value and is missing from the recorded", func() {
		It("should update the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:1
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items differs from the remote value but matches the recorded", func() {
		It("should update the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:1
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items is missing from the remote but matches the recorded", func() {
		It("should add the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where one of the items is missing from the remote and missing from the recorded ", func() {
		It("should add the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:

`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: item
          image: image:2
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})

	Context("where the order of the resolved, local and remote lists differs", func() {
		It("should keep the order specified in local and append items appears only in remote", func() {
			recorded := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: recorded-local
        image: recorded:b
        timeoutSeconds: 2
      - name: recorded-remote
        image: recorded:c
        timeoutSeconds: 3
      - name: recorded-local-remote
        image: recorded:d
        timeoutSeconds: 4
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: local
          image: local:a
          initialDelaySeconds: 15
        - name: recorded-local-remote
          image: local:b
          initialDelaySeconds: 16
        - name: local-remote
          image: local:c
          initialDelaySeconds: 17
        - name: recorded-local
          image: local:d
          initialDelaySeconds: 18
`)
			remote := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: remote
          image: remote:a
          imagePullPolicy: Always
        - name: recorded-remote
          image: remote:b
          imagePullPolicy: Always
        - name: local-remote
          image: remote:c
          imagePullPolicy: Always
        - name: recorded-local-remote
          image: remote:d
          imagePullPolicy: Always
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: local
          image: local:a
          initialDelaySeconds: 15
        - name: recorded-local-remote
          image: local:b
          imagePullPolicy: Always
          initialDelaySeconds: 16
        - name: local-remote
          image: local:c
          imagePullPolicy: Always
          initialDelaySeconds: 17
        - name: recorded-local
          image: local:d
          initialDelaySeconds: 18
        - name: remote
          image: remote:a
          imagePullPolicy: Always
`)
			run(strategy.Create(strategy.Options{}), recorded, local, remote, expected)
		})
	})
})

var _ = Describe("Merging fields of type list-of-map with openapi containing a multi-field mergekey", func() {
	var resources openapi.Resources
	BeforeEach(func() {
		resources = tst.NewFakeResources("test_swagger.json")
	})

	Context("where one of the items has been deleted", func() {
		It("should delete the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
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
apiVersion: apps/v1beta1
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
apiVersion: apps/v1beta1
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
			expected := create(`
apiVersion: apps/v1beta1
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
			runWith(strategy.Create(strategy.Options{}), recorded, local, remote, expected, resources)
		})
	})

	Context("where one of the items has been updated", func() {
		It("should merge updates to the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
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
          hostPort: 2021
`)
			local := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        ports:
        - containerPort: 8080
          protocol: TCP
          hostPort: 2023
        - containerPort: 8080
          protocol: UDP
          hostPort: 2022
`)
			remote := create(`
apiVersion: apps/v1beta1
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
          hostPort: 2021
          hostIP: "127.0.0.1"
`)
			expected := create(`
apiVersion: apps/v1beta1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: container
        ports:
        - containerPort: 8080
          protocol: TCP
          hostPort: 2023
        - containerPort: 8080
          protocol: UDP
          hostPort: 2022
          hostIP: "127.0.0.1"
`)
			runWith(strategy.Create(strategy.Options{}), recorded, local, remote, expected, resources)
		})
	})

	Context("where one of the items has been added", func() {
		It("should add the item", func() {
			recorded := create(`
apiVersion: apps/v1beta1
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
			local := create(`
apiVersion: apps/v1beta1
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
			remote := create(`
apiVersion: apps/v1beta1
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
          hostIP: "127.0.0.1"
`)
			expected := create(`
apiVersion: apps/v1beta1
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
          hostIP: "127.0.0.1"
        - containerPort: 8080
          protocol: UDP
          hostPort: 2022
`)
			runWith(strategy.Create(strategy.Options{}), recorded, local, remote, expected, resources)
		})
	})
})

var _ = Describe("Merging fields of type list-of-map with openapi", func() {
	Context("containing a replace-keys sub strategy", func() {
		It("should apply the replace-key strategy when merging the item", func() {
		})
	})
})
