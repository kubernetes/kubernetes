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
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kube-openapi/pkg/util/proto/testing"
	"k8s.io/kubectl/pkg/util/openapi"
)

var fakeSchema = testing.Fake{Path: filepath.Join("..", "..", "..", "testdata", "openapi", "swagger.json")}

var _ = Describe("Reading apps/v1/Deployment from openAPIData", func() {
	var resources openapi.Resources
	BeforeEach(func() {
		s, err := fakeSchema.OpenAPISchema()
		Expect(err).ToNot(HaveOccurred())
		resources, err = openapi.NewOpenAPIData(s)
		Expect(err).ToNot(HaveOccurred())
	})

	gvk := schema.GroupVersionKind{
		Kind:    "Deployment",
		Version: "v1",
		Group:   "apps",
	}

	var schema proto.Schema
	It("should lookup the Schema by its GroupVersionKind", func() {
		schema = resources.LookupResource(gvk)
		Expect(schema).ToNot(BeNil())
		Expect(schema.(*proto.Kind)).ToNot(BeNil())
		consumes := resources.GetConsumes(gvk, "PATCH")
		Expect(consumes).ToNot(BeNil())
		Expect(consumes).To(HaveLen(4))
	})
})

var _ = Describe("Reading authorization.k8s.io/v1/SubjectAccessReview from openAPIData", func() {
	var resources openapi.Resources
	BeforeEach(func() {
		s, err := fakeSchema.OpenAPISchema()
		Expect(err).ToNot(HaveOccurred())
		resources, err = openapi.NewOpenAPIData(s)
		Expect(err).ToNot(HaveOccurred())
	})

	gvk := schema.GroupVersionKind{
		Kind:    "SubjectAccessReview",
		Version: "v1",
		Group:   "authorization.k8s.io",
	}

	var schema proto.Schema
	It("should lookup the Schema by its GroupVersionKind", func() {
		schema = resources.LookupResource(gvk)
		Expect(schema).ToNot(BeNil())
		sar := schema.(*proto.Kind)
		Expect(sar).ToNot(BeNil())
		Expect(sar.Fields).To(HaveKey("spec"))
		specRef := sar.Fields["spec"].(proto.Reference)
		Expect(specRef).ToNot(BeNil())
		Expect(specRef.Reference()).To(Equal("io.k8s.api.authorization.v1.SubjectAccessReviewSpec"))
		Expect(specRef.SubSchema().(*proto.Kind)).ToNot(BeNil())
	})
})
