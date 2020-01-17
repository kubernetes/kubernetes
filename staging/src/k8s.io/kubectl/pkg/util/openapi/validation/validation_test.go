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

package validation

import (
	"path/filepath"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/kube-openapi/pkg/util/proto/validation"

	// This dependency is needed to register API types.
	"k8s.io/kube-openapi/pkg/util/proto/testing"
	"k8s.io/kubectl/pkg/util/openapi"
)

var fakeSchema = testing.Fake{Path: filepath.Join("..", "..", "..", "..", "testdata", "openapi", "swagger.json")}

var _ = Describe("resource validation using OpenAPI Schema", func() {
	var validator *SchemaValidation
	BeforeEach(func() {
		s, err := fakeSchema.OpenAPISchema()
		Expect(err).To(BeNil())
		resources, err := openapi.NewOpenAPIData(s)
		Expect(err).To(BeNil())
		validator = NewSchemaValidation(resources)
		Expect(validator).ToNot(BeNil())
	})

	It("finds Deployment in Schema and validates it", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    name: redis-master
  name: name
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - image: redis
        name: redis
`))
		Expect(err).To(BeNil())
	})

	It("validates a valid pod", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: redis-master
  name: name
spec:
  containers:
  - args:
    - this
    - is
    - an
    - ok
    - command
    image: gcr.io/fake_project/fake_image:fake_tag
    name: master
`))
		Expect(err).To(BeNil())
	})

	It("finds invalid command (string instead of []string) in Json Pod", func() {
		err := validator.ValidateBytes([]byte(`
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "name",
    "labels": {
      "name": "redis-master"
    }
  },
  "spec": {
    "containers": [
      {
        "name": "master",
	"image": "gcr.io/fake_project/fake_image:fake_tag",
        "args": "this is a bad command"
      }
    ]
  }
}
`))
		Expect(err).To(Equal(utilerrors.NewAggregate([]error{
			validation.ValidationError{
				Path: "Pod.spec.containers[0].args",
				Err: validation.InvalidTypeError{
					Path:     "io.k8s.api.core.v1.Container.args",
					Expected: "array",
					Actual:   "string",
				},
			},
		})))
	})

	It("fails because hostPort is string instead of int", func() {
		err := validator.ValidateBytes([]byte(`
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "apache-php",
    "labels": {
      "name": "apache-php"
    }
  },
  "spec": {
    "volumes": [{
        "name": "shared-disk"
    }],
    "containers": [
      {
        "name": "apache-php",
        "image": "gcr.io/fake_project/fake_image:fake_tag",
        "ports": [
          {
            "name": "apache",
            "hostPort": "13380",
            "containerPort": 80,
            "protocol": "TCP"
          }
        ],
        "volumeMounts": [
          {
            "name": "shared-disk",
            "mountPath": "/var/www/html"
          }
        ]
      }
    ]
  }
}
`))

		Expect(err).To(Equal(utilerrors.NewAggregate([]error{
			validation.ValidationError{
				Path: "Pod.spec.containers[0].ports[0].hostPort",
				Err: validation.InvalidTypeError{
					Path:     "io.k8s.api.core.v1.ContainerPort.hostPort",
					Expected: "integer",
					Actual:   "string",
				},
			},
		})))

	})

	It("fails because volume is not an array of object", func() {
		err := validator.ValidateBytes([]byte(`
{
  "kind": "Pod",
  "apiVersion": "v1",
  "metadata": {
    "name": "apache-php",
    "labels": {
      "name": "apache-php"
    }
  },
  "spec": {
    "volumes": [
        "name": "shared-disk"
    ],
    "containers": [
      {
        "name": "apache-php",
	"image": "gcr.io/fake_project/fake_image:fake_tag",
        "ports": [
          {
            "name": "apache",
            "hostPort": 13380,
            "containerPort": 80,
            "protocol": "TCP"
          }
        ],
        "volumeMounts": [
          {
            "name": "shared-disk",
            "mountPath": "/var/www/html"
          }
        ]
      }
    ]
  }
}
`))
		Expect(err.Error()).To(Equal("invalid character ':' after array element"))
	})

	It("fails because some string lists have empty strings", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: redis-master
  name: name
spec:
  containers:
  - image: gcr.io/fake_project/fake_image:fake_tag
    name: master
    args:
    -
    command:
    -
`))

		Expect(err).To(Equal(utilerrors.NewAggregate([]error{
			validation.ValidationError{
				Path: "Pod.spec.containers[0].args",
				Err: validation.InvalidObjectTypeError{
					Path: "Pod.spec.containers[0].args[0]",
					Type: "nil",
				},
			},
			validation.ValidationError{
				Path: "Pod.spec.containers[0].command",
				Err: validation.InvalidObjectTypeError{
					Path: "Pod.spec.containers[0].command[0]",
					Type: "nil",
				},
			},
		})))
	})

	It("fails if required fields are missing", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: redis-master
  name: name
spec:
  containers:
  - command: ["my", "command"]
`))

		Expect(err).To(Equal(utilerrors.NewAggregate([]error{
			validation.ValidationError{
				Path: "Pod.spec.containers[0]",
				Err: validation.MissingRequiredFieldError{
					Path:  "io.k8s.api.core.v1.Container",
					Field: "name",
				},
			},
		})))
	})

	It("fails if required fields are empty", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: redis-master
  name: name
spec:
  containers:
  - image:
    name:
`))

		Expect(err).To(Equal(utilerrors.NewAggregate([]error{
			validation.ValidationError{
				Path: "Pod.spec.containers[0]",
				Err: validation.MissingRequiredFieldError{
					Path:  "io.k8s.api.core.v1.Container",
					Field: "name",
				},
			},
		})))
	})

	It("is fine with empty non-mandatory fields", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: redis-master
  name: name
spec:
  containers:
  - image: image
    name: name
    command:
`))

		Expect(err).To(BeNil())
	})

	It("can validate lists", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: List
items:
  - apiVersion: v1
    kind: Pod
    metadata:
      labels:
        name: redis-master
      name: name
    spec:
      containers:
      - name: name
`))

		Expect(err).To(BeNil())
	})

	It("fails because apiVersion is not provided", func() {
		err := validator.ValidateBytes([]byte(`
kind: Pod
metadata:
  name: name
spec:
  containers:
  - name: name
    image: image
`))
		Expect(err.Error()).To(Equal("apiVersion not set"))
	})

	It("fails because apiVersion type is not string and kind is not provided", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: 1
metadata:
  name: name
spec:
  containers:
  - name: name
    image: image
`))
		Expect(err.Error()).To(Equal("[apiVersion isn't string type, kind not set]"))
	})

	It("fails because List first item is missing kind and second item is missing apiVersion", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: v1
kind: List
items:
- apiVersion: v1
  metadata:
    name: name
  spec:
    replicas: 1
    template:
      metadata:
        labels:
          name: name
      spec:
        containers:
        - name: name
          image: image
- kind: Service
  metadata:
    name: name
  spec:
    type: NodePort
    ports:
    - port: 123
      targetPort: 1234
      name: name
    selector:
      name: name
`))
		Expect(err.Error()).To(Equal("[kind not set, apiVersion not set]"))
	})

	It("is fine with crd resource with List as a suffix kind name, which may not be a list of resources", func() {
		err := validator.ValidateBytes([]byte(`
apiVersion: fake.com/v1
kind: FakeList
metadata:
  name: fake
spec:
  foo: bar
`))
		Expect(err).To(BeNil())
	})
})
