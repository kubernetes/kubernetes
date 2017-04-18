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

package openapi

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"fmt"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var _ = Describe("Reading apps/v1beta1/Deployment from openAPIData", func() {
	var instance *Resources
	BeforeEach(func() {
		s, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(s)
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

	var definition Kind
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
		_, found := definition.Extensions["x-kubernetes-group-version-kind"]
		Expect(found).To(BeTrue())
	})

	It("should find the definition fields", func() {
		By("for 'kind'")
		kindField, found := definition.Fields["kind"]
		Expect(found).To(BeTrue())

		Expect(kindField.TypeName).To(Equal("string"))
		Expect(kindField.IsKind).To(BeFalse())
		Expect(kindField.IsPrimitive).To(BeTrue())
		Expect(kindField.IsArray).To(BeFalse())
		Expect(kindField.ElementType).To(BeNil())

		By("for 'apiVersion'")
		versionField, found := definition.Fields["apiVersion"]
		Expect(found).To(BeTrue())
		Expect(versionField.TypeName).To(Equal("string"))
		Expect(versionField.IsKind).To(BeFalse())
		Expect(versionField.IsPrimitive).To(BeTrue())
		Expect(versionField.IsArray).To(BeFalse())
		Expect(versionField.ElementType).To(BeNil())

		By("for 'metadata'")
		metadataField, found := definition.Fields["metadata"]
		Expect(found).To(BeTrue())
		Expect(metadataField.TypeName).To(Equal("io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"))
		Expect(metadataField.IsKind).To(BeTrue())
		Expect(metadataField.IsPrimitive).To(BeFalse())
		Expect(metadataField.IsArray).To(BeFalse())
		Expect(metadataField.ElementType).To(BeNil())

		By("for 'spec'")
		specField, found := definition.Fields["spec"]
		Expect(found).To(BeTrue())
		Expect(specField.TypeName).To(Equal("io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentSpec"))
		Expect(specField.IsKind).To(BeTrue())
		Expect(specField.IsPrimitive).To(BeFalse())
		Expect(specField.IsArray).To(BeFalse())
		Expect(specField.ElementType).To(BeNil())

		By("for 'status'")
		statusField, found := definition.Fields["status"]
		Expect(found).To(BeTrue())
		Expect(statusField.TypeName).To(Equal("io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentStatus"))
		Expect(statusField.IsKind).To(BeTrue())
		Expect(statusField.IsPrimitive).To(BeFalse())
		Expect(statusField.IsArray).To(BeFalse())
		Expect(statusField.ElementType).To(BeNil())
	})
})

var _ = Describe("Reading apps/v1beta1/DeploymentStatus from openAPIData", func() {
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	deploymentStatusName := "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentStatus"

	var definition Kind
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
		availableReplicas, found := definition.Fields["availableReplicas"]
		Expect(found).To(BeTrue())
		Expect(availableReplicas.TypeName).To(Equal("integer"))
		Expect(availableReplicas.IsKind).To(BeFalse())
		Expect(availableReplicas.IsPrimitive).To(BeTrue())
		Expect(availableReplicas.IsArray).To(BeFalse())
		Expect(availableReplicas.ElementType).To(BeNil())

		By("for 'conditions'")
		conditionsField, found := definition.Fields["conditions"]
		Expect(found).To(BeTrue())
		Expect(conditionsField.TypeName).To(Equal("io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentCondition array"))
		Expect(conditionsField.IsKind).To(BeFalse())
		Expect(conditionsField.IsPrimitive).To(BeFalse())
		Expect(conditionsField.IsArray).To(BeTrue())
		Expect(conditionsField.ElementType).ToNot(BeNil())
		Expect(conditionsField.ElementType.TypeName).
			To(Equal("io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentCondition"))
		Expect(conditionsField.ElementType.IsKind).To(BeTrue())
		Expect(conditionsField.ElementType.IsPrimitive).To(BeFalse())
		Expect(conditionsField.ElementType.IsMap).To(BeFalse())
		Expect(conditionsField.ElementType.IsArray).To(BeFalse())

		patchMergeKey, found := conditionsField.Extensions.GetString("x-kubernetes-patch-merge-key")
		Expect(found).To(BeTrue())
		Expect(patchMergeKey).To(Equal("type"))

		patchStrategy, found := conditionsField.Extensions.GetString("x-kubernetes-patch-strategy")
		Expect(found).To(BeTrue())
		Expect(patchStrategy).To(Equal("merge"))
	})
})

var _ = Describe("Reading apps/v1beta1/DeploymentSpec from openAPIData", func() {
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	deploymentSpecName := "io.k8s.kubernetes.pkg.apis.apps.v1beta1.DeploymentSpec"

	var definition Kind
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
		templateField, found := definition.Fields["template"]
		Expect(found).To(BeTrue())
		Expect(templateField.TypeName).To(Equal("io.k8s.kubernetes.pkg.api.v1.PodTemplateSpec"))
		Expect(templateField.IsKind).To(BeTrue())
		Expect(templateField.IsPrimitive).To(BeFalse())
		Expect(templateField.IsArray).To(BeFalse())
		Expect(templateField.IsMap).To(BeFalse())
	})
})

var _ = Describe("Reading v1/ObjectMeta from openAPIData", func() {
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	objectMetaName := "io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta"

	var definition Kind
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
		finalizersField, found := definition.Fields["finalizers"]
		Expect(found).To(BeTrue())
		Expect(finalizersField.TypeName).To(Equal("string array"))
		Expect(finalizersField.IsKind).To(BeFalse())
		Expect(finalizersField.IsPrimitive).To(BeFalse())
		Expect(finalizersField.IsArray).To(BeTrue())

		Expect(finalizersField.ElementType).ToNot(BeNil())
		Expect(finalizersField.ElementType.TypeName).To(Equal("string"))
		Expect(finalizersField.ElementType.IsKind).To(BeFalse())
		Expect(finalizersField.ElementType.IsPrimitive).To(BeTrue())
		Expect(finalizersField.IsMap).To(BeFalse())

		By("for 'ownerReferences'")
		ownerReferencesField, found := definition.Fields["ownerReferences"]
		Expect(found).To(BeTrue())
		Expect(ownerReferencesField.TypeName).
			To(Equal("io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference array"))
		Expect(ownerReferencesField.IsKind).To(BeFalse())
		Expect(ownerReferencesField.IsPrimitive).To(BeFalse())
		Expect(ownerReferencesField.IsArray).To(BeTrue())
		Expect(ownerReferencesField.IsMap).To(BeFalse())
		Expect(ownerReferencesField.ElementType).ToNot(BeNil())
		Expect(ownerReferencesField.ElementType.TypeName).
			To(Equal("io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference"))
		Expect(ownerReferencesField.ElementType.ElementType).To(BeNil())
		Expect(ownerReferencesField.ElementType.IsPrimitive).To(BeFalse())
		Expect(ownerReferencesField.ElementType.IsArray).To(BeFalse())
		Expect(ownerReferencesField.ElementType.IsMap).To(BeFalse())
		Expect(ownerReferencesField.ElementType.IsKind).To(BeTrue())

		By("for 'labels'")
		labelsField, found := definition.Fields["labels"]
		Expect(found).To(BeTrue())
		Expect(labelsField.TypeName).To(Equal("string map"))
		Expect(labelsField.IsKind).To(BeFalse())
		Expect(labelsField.IsPrimitive).To(BeFalse())
		Expect(labelsField.IsArray).To(BeFalse())
		Expect(labelsField.IsMap).To(BeTrue())
		Expect(labelsField.ElementType).ToNot(BeNil())
		Expect(labelsField.ElementType.TypeName).To(Equal("string"))
		Expect(labelsField.ElementType.IsKind).To(BeFalse())
		Expect(labelsField.ElementType.IsPrimitive).To(BeTrue())
	})
})

var _ = Describe("Reading v1/NodeStatus from openAPIData", func() {
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	nodeStatusName := "io.k8s.kubernetes.pkg.api.v1.NodeStatus"

	var definition Kind
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
		allocatableField, found := definition.Fields["allocatable"]
		Expect(found).To(BeTrue())
		Expect(allocatableField.TypeName).
			To(Equal("io.k8s.apimachinery.pkg.api.resource.Quantity map"))
		Expect(allocatableField.IsKind).To(BeFalse())
		Expect(allocatableField.IsPrimitive).To(BeFalse())
		Expect(allocatableField.IsArray).To(BeFalse())
		Expect(allocatableField.IsMap).To(BeTrue())
		Expect(allocatableField.ElementType).ToNot(BeNil())
		Expect(allocatableField.ElementType.TypeName).
			To(Equal("io.k8s.apimachinery.pkg.api.resource.Quantity"))
		Expect(allocatableField.ElementType.IsKind).To(BeTrue())
		Expect(allocatableField.ElementType.IsPrimitive).To(BeFalse())
	})
})

var _ = Describe("Reading Utility Definitions from openAPIData", func() {
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	Context("for util.intstr.IntOrString", func() {
		var definition Kind
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
		var definition Kind
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
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
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
	var instance *Resources
	BeforeEach(func() {
		d, err := data.OpenAPISchema()
		Expect(err).To(BeNil())
		instance, err = newOpenAPIData(d)
		Expect(err).To(BeNil())
	})

	subjectAccessReviewSpecName := "io.k8s.kubernetes.pkg.apis.authorization.v1.SubjectAccessReviewSpec"

	var definition Kind
	It("should find the definition by name", func() {
		var found bool
		definition, found = instance.NameToDefinition[subjectAccessReviewSpecName]
		Expect(found).To(BeTrue())
		Expect(definition.Name).To(Equal(subjectAccessReviewSpecName))
		Expect(definition.PrimitiveType).To(BeEmpty())
	})

	It("should find the definition fields", func() {
		By("for 'allocatable'")
		exraField, found := definition.Fields["extra"]
		Expect(found).To(BeTrue())
		Expect(exraField.TypeName).
			To(Equal("string array map"))
		Expect(exraField.IsKind).To(BeFalse())
		Expect(exraField.IsPrimitive).To(BeFalse())
		Expect(exraField.IsArray).To(BeFalse())
		Expect(exraField.IsMap).To(BeTrue())
		Expect(exraField.ElementType).NotTo(BeNil())

		Expect(exraField.ElementType.TypeName).
			To(Equal("string array"))
		Expect(exraField.ElementType.IsKind).To(BeFalse())
		Expect(exraField.ElementType.IsPrimitive).To(BeFalse())
		Expect(exraField.ElementType.IsArray).To(BeTrue())
		Expect(exraField.ElementType.IsMap).To(BeFalse())
		Expect(exraField.ElementType.ElementType).NotTo(BeNil())

		Expect(exraField.ElementType.ElementType.TypeName).
			To(Equal("string"))
		Expect(exraField.ElementType.ElementType.IsKind).To(BeFalse())
		Expect(exraField.ElementType.ElementType.IsPrimitive).To(BeTrue())
		Expect(exraField.ElementType.ElementType.IsArray).To(BeFalse())
		Expect(exraField.ElementType.ElementType.IsMap).To(BeFalse())
		Expect(exraField.ElementType.ElementType.ElementType).To(BeNil())
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
