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

package openapi_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

var _ = Describe("Reading apps/v1beta1/Deployment from openAPIData", func() {
	var resources openapi.Resources
	BeforeEach(func() {
		s, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		resources, err = openapi.NewOpenAPIData(s)
		Expect(err).To(BeNil())
	})

	gvk := schema.GroupVersionKind{
		Kind:    "Deployment",
		Version: "v1beta1",
		Group:   "apps",
	}

	var schema openapi.Schema
	It("should lookup the Schema by its GroupVersionKind", func() {
		schema = resources.LookupResource(gvk)
		Expect(schema).ToNot(BeNil())
	})

	var deployment *openapi.PropertiesMap
	It("should be a PropertiesMap", func() {
		deployment = schema.(*openapi.PropertiesMap)
		Expect(deployment).ToNot(BeNil())
	})

	It("should have a kind key of type string", func() {
		Expect(deployment.Fields).To(HaveKey("kind"))
		key := deployment.Fields["kind"].(*openapi.Primitive)
		Expect(key).ToNot(BeNil())
		Expect(key.Type).To(Equal("string"))
	})

	It("should have a apiVersion key of type string", func() {
		Expect(deployment.Fields).To(HaveKey("apiVersion"))
		key := deployment.Fields["apiVersion"].(*openapi.Primitive)
		Expect(key).ToNot(BeNil())
		Expect(key.Type).To(Equal("string"))
	})

	It("should have a metadata key of type Reference", func() {
		Expect(deployment.Fields).To(HaveKey("metadata"))
		key := deployment.Fields["metadata"].(*openapi.Reference)
		Expect(key).ToNot(BeNil())
		Expect(key.Reference).To(Equal("io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"))
		subSchema := key.GetSubSchema().(*openapi.PropertiesMap)
		Expect(subSchema).ToNot(BeNil())
	})

	It("should have a status key of type Reference", func() {
		Expect(deployment.Fields).To(HaveKey("status"))
		key := deployment.Fields["status"].(*openapi.Reference)
		Expect(key).ToNot(BeNil())
		Expect(key.Reference).To(Equal("io.k8s.api.apps.v1beta1.DeploymentStatus"))
		subSchema := key.GetSubSchema().(*openapi.PropertiesMap)
		Expect(subSchema).ToNot(BeNil())
	})

	var spec *openapi.PropertiesMap
	It("should have a spec key of type Reference", func() {
		Expect(deployment.Fields).To(HaveKey("spec"))
		key := deployment.Fields["spec"].(*openapi.Reference)
		Expect(key).ToNot(BeNil())
		Expect(key.Reference).To(Equal("io.k8s.api.apps.v1beta1.DeploymentSpec"))
		spec = key.GetSubSchema().(*openapi.PropertiesMap)
		Expect(spec).ToNot(BeNil())
	})

	It("should have a spec with no gvk", func() {
		_, found := spec.GetExtensions()["x-kubernetes-group-version-kind"]
		Expect(found).To(BeFalse())
	})

	It("should have a spec with a PodTemplateSpec sub-field", func() {
		Expect(spec.Fields).To(HaveKey("template"))
		key := spec.Fields["template"].(*openapi.Reference)
		Expect(key).ToNot(BeNil())
		Expect(key.Reference).To(Equal("io.k8s.api.core.v1.PodTemplateSpec"))
	})
})
