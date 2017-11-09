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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/runtime/schema"
	tst "k8s.io/apimachinery/pkg/util/openapi/testing"
	"k8s.io/kube-openapi/pkg/util/proto"
)

var fake = tst.FakeResources{Getter: &tst.Fake{
	Path: filepath.Join("..", "..", "..", "..", "..", "..", "..", "api", "openapi-spec", "swagger.json"),
}}

var _ = Describe("Reading apps/v1beta1/Deployment from openAPIData", func() {
	gvk := schema.GroupVersionKind{
		Kind:    "Deployment",
		Version: "v1beta1",
		Group:   "apps",
	}

	var schema proto.Schema
	It("should lookup the Schema by its GroupVersionKind", func() {
		schema = fake.LookupResource(gvk)
		Expect(schema).ToNot(BeNil())
	})

	var deployment *proto.Kind
	It("should be a Kind", func() {
		deployment = schema.(*proto.Kind)
		Expect(deployment).ToNot(BeNil())
	})
})

var _ = Describe("Reading authorization.k8s.io/v1/SubjectAccessReview from openAPIData", func() {
	gvk := schema.GroupVersionKind{
		Kind:    "SubjectAccessReview",
		Version: "v1",
		Group:   "authorization.k8s.io",
	}

	var schema proto.Schema
	It("should lookup the Schema by its GroupVersionKind", func() {
		schema = fake.LookupResource(gvk)
		Expect(schema).ToNot(BeNil())
	})

	var sarspec *proto.Kind
	It("should be a Kind and have a spec", func() {
		sar := schema.(*proto.Kind)
		Expect(sar).ToNot(BeNil())
		Expect(sar.Fields).To(HaveKey("spec"))
		specRef := sar.Fields["spec"].(proto.Reference)
		Expect(specRef).ToNot(BeNil())
		Expect(specRef.Reference()).To(Equal("io.k8s.api.authorization.v1.SubjectAccessReviewSpec"))
		sarspec = specRef.SubSchema().(*proto.Kind)
		Expect(sarspec).ToNot(BeNil())
	})
})
