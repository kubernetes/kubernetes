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

package validation_test

import (
	"fmt"
	"path/filepath"

	"github.com/ghodss/yaml"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/util/proto/testing"
	"k8s.io/kube-openapi/pkg/util/proto/validation"
)

var fakeSchema = testing.Fake{Path: filepath.Join("..", "testdata", "swagger.json")}

func Validate(models proto.Models, model string, data string) []error {
	var obj interface{}
	if err := yaml.Unmarshal([]byte(data), &obj); err != nil {
		return []error{fmt.Errorf("pre-validation: failed to parse yaml: %v", err)}
	}
	return ValidateObj(models, model, obj)
}

// ValidateObj validates an object produced by decoding json or yaml.
// Numbers may be int64 or float64.
func ValidateObj(models proto.Models, model string, obj interface{}) []error {
	schema := models.LookupModel(model)
	if schema == nil {
		return []error{fmt.Errorf("pre-validation: couldn't find model %s", model)}
	}

	return validation.ValidateModel(obj, schema, model)
}

var _ = Describe("resource validation using OpenAPI Schema", func() {
	var models proto.Models
	BeforeEach(func() {
		s, err := fakeSchema.OpenAPISchema()
		Expect(err).To(BeNil())
		models, err = proto.NewOpenAPIData(s)
		Expect(err).To(BeNil())
	})

	It("finds Deployment in Schema and validates it", func() {
		err := Validate(models, "io.k8s.api.apps.v1beta1.Deployment", `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    name: redis-master
  name: name
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - image: redis
        name: redis
`)
		Expect(err).To(BeNil())
	})

	It("validates a valid pod", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)
		Expect(err).To(BeNil())
	})

	It("finds invalid command (string instead of []string) in Json Pod", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)
		Expect(err).To(Equal([]error{
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0].args",
				Err: validation.InvalidTypeError{
					Path:     "io.k8s.api.core.v1.Container.args",
					Expected: "array",
					Actual:   "string",
				},
			},
		}))
	})

	It("fails because hostPort is string instead of int", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)

		Expect(err).To(Equal([]error{
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0].ports[0].hostPort",
				Err: validation.InvalidTypeError{
					Path:     "io.k8s.api.core.v1.ContainerPort.hostPort",
					Expected: "integer",
					Actual:   "string",
				},
			},
		}))

	})

	It("fails because volume is not an array of object", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)
		Expect(err).To(BeNil())
	})

	It("fails because some string lists have empty strings", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)

		Expect(err).To(Equal([]error{
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0].args",
				Err: validation.InvalidObjectTypeError{
					Path: "io.k8s.api.core.v1.Pod.spec.containers[0].args[0]",
					Type: "nil",
				},
			},
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0].command",
				Err: validation.InvalidObjectTypeError{
					Path: "io.k8s.api.core.v1.Pod.spec.containers[0].command[0]",
					Type: "nil",
				},
			},
		}))
	})

	It("fails if required fields are missing", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
apiVersion: v1
kind: Pod
metadata:
  labels:
    name: redis-master
  name: name
spec:
  containers:
  - command: ["my", "command"]
`)

		Expect(err).To(Equal([]error{
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0]",
				Err: validation.MissingRequiredFieldError{
					Path:  "io.k8s.api.core.v1.Container",
					Field: "name",
				},
			},
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0]",
				Err: validation.MissingRequiredFieldError{
					Path:  "io.k8s.api.core.v1.Container",
					Field: "image",
				},
			},
		}))
	})

	It("fails if required fields are empty", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)

		Expect(err).To(Equal([]error{
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0]",
				Err: validation.MissingRequiredFieldError{
					Path:  "io.k8s.api.core.v1.Container",
					Field: "name",
				},
			},
			validation.ValidationError{
				Path: "io.k8s.api.core.v1.Pod.spec.containers[0]",
				Err: validation.MissingRequiredFieldError{
					Path:  "io.k8s.api.core.v1.Container",
					Field: "image",
				},
			},
		}))
	})

	It("is fine with empty non-mandatory fields", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
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
`)

		Expect(err).To(BeNil())
	})

	It("fails because apiVersion is not provided", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
kind: Pod
metadata:
  name: name
spec:
  containers:
  - name: name
    image: image
`)
		Expect(err).To(BeNil())
	})

	It("fails because apiVersion type is not string and kind is not provided", func() {
		err := Validate(models, "io.k8s.api.core.v1.Pod", `
apiVersion: 1
metadata:
  name: name
spec:
  containers:
  - name: name
    image: image
`)
		Expect(err).To(BeNil())
	})

	// verify integer literals are considered to be compatible with float schema fields
	It("validates integer values for float fields", func() {
		err := ValidateObj(models, "io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1.CustomResourceDefinition", map[string]interface{}{
			"apiVersion": "apiextensions.k8s.io/v1",
			"kind":       "CustomResourceDefinition",
			"metadata":   map[string]interface{}{"name": "foo"},
			"spec": map[string]interface{}{
				"scope": "Namespaced",
				"group": "example.com",
				"names": map[string]interface{}{
					"plural": "numbers",
					"kind":   "Number",
				},
				"versions": []interface{}{
					map[string]interface{}{
						"name":    "v1",
						"served":  true,
						"storage": true,
						"schema": map[string]interface{}{
							"openAPIV3Schema": map[string]interface{}{
								"properties": map[string]interface{}{
									"replicas": map[string]interface{}{
										"default": int64(1),
										"minimum": int64(0),
										"type":    "integer",
									},
									"resources": map[string]interface{}{
										"default": float64(1.1),
										"minimum": float64(0.1),
										"type":    "number",
									},
								},
							},
						},
					},
				},
			},
		})
		Expect(err).To(BeNil())
	})
})
