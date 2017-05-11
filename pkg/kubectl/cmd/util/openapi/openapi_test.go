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
	"fmt"

	"github.com/go-openapi/spec"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi"
)

var _ = Describe("Reading apps/v1beta1/Deployment from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		s, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(s)
		Expect(err).To(BeNil())
	})

	deploymentName := "io.k8s.kubernetes.pkg.apis.apps.v1beta1.Deployment"
	gvk := schema.GroupVersionKind{
		Kind:    "Deployment",
		Version: "v1beta1",
		Group:   "apps",
	}

	It("should find the name by its GroupVersionKind", func() {
		name, found := instance.GroupVersionKindToName[gvk]
		Expect(found).To(BeTrue())
		Expect(name).To(Equal(deploymentName))
	})

	var definition openapi.Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[deploymentName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(deploymentName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should lookup the Kind by its GroupVersionKind", func() {
		d, found := instance.LookupResource(gvk)
		Expect(found).To(BeTrue())
		Expect(d).To(Equal(definition))
	})

	It("should find the definition GroupVersionKind", func() {
		Expect(definition.GroupVersionKind).To(Equal(gvk))
	})

	It("should find the definition GroupVersionKind extensions", func() {
		Expect(definition.Extensions).To(HaveKey("x-kubernetes-group-version-kind"))
	})

	It("should find the definition fields", func() {
		By("for 'kind'")
		Expect(definition.Fields).To(HaveKeyWithValue("kind", openapi.Type{
			TypeName:    "string",
			IsPrimitive: true,
		}))

		By("for 'apiVersion'")
		Expect(definition.Fields).To(HaveKeyWithValue("apiVersion", openapi.Type{
			TypeName:    "string",
			IsPrimitive: true,
		}))

		By("for 'metadata'")
		Expect(definition.Fields).To(HaveKeyWithValue("metadata", openapi.Type{
			TypeName: "io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta",
			IsKind:   true,
		}))

		By("for 'spec'")
		Expect(definition.Fields).To(HaveKeyWithValue("spec", openapi.Type{
			TypeName: "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentSpec",
			IsKind:   true,
		}))

		By("for 'status'")
		Expect(definition.Fields).To(HaveKeyWithValue("status", openapi.Type{
			TypeName: "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentStatus",
			IsKind:   true,
		}))
	})
})

var _ = Describe("Reading apps/v1beta1/DeploymentStatus from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	deploymentStatusName := "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentStatus"

	var definition openapi.Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[deploymentStatusName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(deploymentStatusName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should not find the definition GroupVersionKind", func() {
		Expect(definition.GroupVersionKind).To(Equal(schema.GroupVersionKind{}))
	})

	It("should not find the definition GroupVersionKind extensions", func() {
		_, found := definition.Extensions["x-kubernetes-group-version-kind"]
		Expect(found).To(BeFalse())
	})

	It("should find the definition fields", func() {
		By("for 'availableReplicas'")
		Expect(definition.Fields).To(HaveKeyWithValue("availableReplicas", openapi.Type{
			TypeName:    "integer",
			IsPrimitive: true,
		}))

		By("for 'conditions'")
		Expect(definition.Fields).To(HaveKeyWithValue("conditions", openapi.Type{
			TypeName: "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentCondition array",
			IsArray:  true,
			ElementType: &openapi.Type{
				TypeName: "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentCondition",
				IsKind:   true,
			},
			Extensions: spec.Extensions{
				"x-kubernetes-patch-merge-key": "type",
				"x-kubernetes-patch-strategy":  "merge",
			},
		}))
	})
})

var _ = Describe("Reading apps/v1beta1/DeploymentSpec from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	deploymentSpecName := "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentSpec"

	var definition openapi.Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[deploymentSpecName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(deploymentSpecName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should not find the definition GroupVersionKind", func() {
		Expect(definition.GroupVersionKind).To(Equal(schema.GroupVersionKind{}))
	})

	It("should not find the definition GroupVersionKind extensions", func() {
		_, found := definition.Extensions["x-kubernetes-group-version-kind"]
		Expect(found).To(BeFalse())
	})

	It("should find the definition fields", func() {
		By("for 'template'")
		Expect(definition.Fields).To(HaveKeyWithValue("template", openapi.Type{
			TypeName: "io.k8s.kubernetes.pkg.api.v1.PodTemplateSpec",
			IsKind:   true,
		}))
	})
})

var _ = Describe("Reading v1/ObjectMeta from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	objectMetaName := "io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"

	var definition openapi.Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[objectMetaName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(objectMetaName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should not find the definition GroupVersionKind", func() {
		Expect(definition.GroupVersionKind).To(Equal(schema.GroupVersionKind{}))
	})

	It("should not find the definition GroupVersionKind extensions", func() {
		_, found := definition.Extensions["x-kubernetes-group-version-kind"]
		Expect(found).To(BeFalse())
	})

	It("should find the definition fields", func() {
		By("for 'finalizers'")
		Expect(definition.Fields).To(HaveKeyWithValue("finalizers", openapi.Type{
			TypeName: "string array",
			IsArray:  true,
			ElementType: &openapi.Type{
				TypeName:    "string",
				IsPrimitive: true,
			},
			Extensions: spec.Extensions{
				"x-kubernetes-patch-strategy": "merge",
			},
		}))

		By("for 'ownerReferences'")
		Expect(definition.Fields).To(HaveKeyWithValue("ownerReferences", openapi.Type{
			TypeName: "io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference array",
			IsArray:  true,
			ElementType: &openapi.Type{
				TypeName: "io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference",
				IsKind:   true,
			},
			Extensions: spec.Extensions{
				"x-kubernetes-patch-merge-key": "uid",
				"x-kubernetes-patch-strategy":  "merge",
			},
		}))

		By("for 'labels'")
		Expect(definition.Fields).To(HaveKeyWithValue("labels", openapi.Type{
			TypeName: "string map",
			IsMap:    true,
			ElementType: &openapi.Type{
				TypeName:    "string",
				IsPrimitive: true,
			},
		}))
	})
})

var _ = Describe("Reading v1/NodeStatus from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	nodeStatusName := "io.k8s.kubernetes.pkg.api.v1.NodeStatus"

	var definition openapi.Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[nodeStatusName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(nodeStatusName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should not find the definition GroupVersionKind", func() {
		Expect(definition.GroupVersionKind).To(Equal(schema.GroupVersionKind{}))
	})

	It("should not find the definition GroupVersionKind extensions", func() {
		_, found := definition.Extensions["x-kubernetes-group-version-kind"]
		Expect(found).To(BeFalse())
	})

	It("should find the definition fields", func() {
		By("for 'allocatable'")
		Expect(definition.Fields).To(HaveKeyWithValue("allocatable", openapi.Type{
			TypeName: "io.k8s.apimachinery.pkg.api.resource.Quantity map",
			IsMap:    true,
			ElementType: &openapi.Type{
				TypeName: "io.k8s.apimachinery.pkg.api.resource.Quantity",
				IsKind:   true,
			},
		}))
	})
})

var _ = Describe("Reading Utility Definitions from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	Context("for util.intstr.IntOrString", func() {
		var definition openapi.Kind
		It("should find the definition by name", func() {
			intOrStringName := "io.k8s.apimachinery.pkg.util.intstr.IntOrString"
			var found bool
			definition, found = instance.NameToDefinition[intOrStringName]
			Expect(found).To(BeTrue())
			Expect(definition.Name).To(Equal(intOrStringName))
			Expect(definition.PrimitiveType).To(Equal("string"))
		})
	})

	Context("for apis.meta.v1.Time", func() {
		var definition openapi.Kind
		It("should find the definition by name", func() {
			intOrStringName := "io.k8s.apimachinery.pkg.apis.meta.v1.Time"
			var found bool
			definition, found = instance.NameToDefinition[intOrStringName]
			Expect(found).To(BeTrue())
			Expect(definition.Name).To(Equal(intOrStringName))
			Expect(definition.PrimitiveType).To(Equal("string"))
		})
	})
})

var _ = Describe("When parsing the openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	It("should result in each definition and field having a single type", func() {
		for _, d := range instance.NameToDefinition {
			Expect(d.Name).ToNot(BeEmpty())
			for n, f := range d.Fields {
				Expect(f.TypeName).ToNot(BeEmpty(),
					fmt.Sprintf("TypeName for %v.%v is empty %+v", d.Name, n, f))
				Expect(oneOf(f.IsArray, f.IsMap, f.IsPrimitive, f.IsKind)).To(BeTrue(),
					fmt.Sprintf("%+v has multiple types", f))
			}
		}
	})

	It("should find every GroupVersionKind by name", func() {
		for _, name := range instance.GroupVersionKindToName {
			_, found := instance.NameToDefinition[name]
			Expect(found).To(BeTrue())
		}
	})
})

var _ = Describe("Reading authorization/v1/SubjectAccessReviewSpec from openAPIData", func() {
	var instance *openapi.Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = openapi.NewOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	subjectAccessReviewSpecName := "io.k8s.kubernetes.pkg.apis.authorization.v1.SubjectAccessReviewSpec"

	var definition openapi.Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[subjectAccessReviewSpecName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(subjectAccessReviewSpecName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should find the definition fields", func() {
		By("for 'allocatable'")
		Expect(definition.Fields).To(HaveKeyWithValue("extra", openapi.Type{
			TypeName: "string array map",
			IsMap:    true,
			ElementType: &openapi.Type{
				TypeName: "string array",
				IsArray:  true,
				ElementType: &openapi.Type{
					TypeName:    "string",
					IsPrimitive: true,
				},
			},
		}))
	})
})

func oneOf(values ...bool) bool {
	found := false
	for _, v := range values {
		if v && found {
			return false
		}
		if v {
			found = true
		}
	}
	return found
}
