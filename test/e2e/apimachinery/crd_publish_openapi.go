/*
Copyright 2019 The Kubernetes Authors.

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

package apimachinery

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"sigs.k8s.io/yaml"

	"k8s.io/apiserver/pkg/cel/environment"
	openapiutil "k8s.io/kube-openapi/pkg/util"
	"k8s.io/utils/pointer"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	k8sclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	"k8s.io/kubernetes/test/utils/crd"
	admissionapi "k8s.io/pod-security-admission/api"
)

var (
	metaPattern = `"kind":"%s","apiVersion":"%s/%s","metadata":{"name":"%s"}`
)

var _ = SIGDescribe("CustomResourcePublishOpenAPI [Privileged:ClusterAdmin]", func() {
	f := framework.NewDefaultFramework("crd-publish-openapi")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, with validation schema
		Description: Register a custom resource definition with a validating schema consisting of objects, arrays and
		primitives. Attempt to create and apply a change a custom resource using valid properties, via kubectl;
		kubectl validation MUST pass. Attempt both operations with unknown properties and without required
		properties; kubectl validation MUST reject the operations. Attempt kubectl explain; the output MUST
		explain the custom resource properties. Attempt kubectl explain on custom resource properties; the output MUST
		explain the nested custom resource properties.
		All validation should be the same.
	*/
	framework.ConformanceIt("works for CRD with validation schema", func(ctx context.Context) {
		crd, err := setupCRD(f, schemaFoo, "foo", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}

		meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-foo")
		ns := fmt.Sprintf("--namespace=%v", f.Namespace.Name)

		ginkgo.By("kubectl validation (kubectl create and apply) allows request with known and required properties")
		validCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}]}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, validCR, ns, "create", "-f", "-"); err != nil {
			framework.Failf("failed to create valid CR %s: %v", validCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-foo"); err != nil {
			framework.Failf("failed to delete valid CR: %v", err)
		}
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, validCR, ns, "apply", "-f", "-"); err != nil {
			framework.Failf("failed to apply valid CR %s: %v", validCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-foo"); err != nil {
			framework.Failf("failed to delete valid CR: %v", err)
		}

		ginkgo.By("kubectl validation (kubectl create and apply) rejects request with value outside defined enum values")
		badEnumValueCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar", "feeling":"NonExistentValue"}]}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, badEnumValueCR, ns, "create", "-f", "-"); err == nil || !strings.Contains(err.Error(), `Unsupported value: "NonExistentValue"`) {
			framework.Failf("unexpected no error when creating CR with unknown enum value: %v", err)
		}

		// TODO: server-side validation and client-side validation produce slightly different error messages.
		// Because server-side is default in beta but not GA yet, we will produce different behaviors in the default vs GA only conformance tests. We have made the error generic enough to pass both, but should go back and make the error more specific once server-side validation goes GA.
		ginkgo.By("kubectl validation (kubectl create and apply) rejects request with unknown properties when disallowed by the schema")
		unknownCR := fmt.Sprintf(`{%s,"spec":{"foo":true}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, unknownCR, ns, "create", "-f", "-"); err == nil || (!strings.Contains(err.Error(), `unknown field "foo"`) && !strings.Contains(err.Error(), `unknown field "spec.foo"`)) {
			framework.Failf("unexpected no error when creating CR with unknown field: %v", err)
		}
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, unknownCR, ns, "apply", "-f", "-"); err == nil || (!strings.Contains(err.Error(), `unknown field "foo"`) && !strings.Contains(err.Error(), `unknown field "spec.foo"`)) {
			framework.Failf("unexpected no error when applying CR with unknown field: %v", err)
		}

		// TODO: see above note, we should check the value of the error once server-side validation is GA.
		ginkgo.By("kubectl validation (kubectl create and apply) rejects request without required properties")
		noRequireCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"age":"10"}]}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, noRequireCR, ns, "create", "-f", "-"); err == nil || (!strings.Contains(err.Error(), `missing required field "name"`) && !strings.Contains(err.Error(), `spec.bars[0].name: Required value`)) {
			framework.Failf("unexpected no error when creating CR without required field: %v", err)
		}
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, noRequireCR, ns, "apply", "-f", "-"); err == nil || (!strings.Contains(err.Error(), `missing required field "name"`) && !strings.Contains(err.Error(), `spec.bars[0].name: Required value`)) {
			framework.Failf("unexpected no error when applying CR without required field: %v", err)
		}

		ginkgo.By("kubectl explain works to explain CR properties")
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural, `(?s)DESCRIPTION:.*Foo CRD for Testing.*FIELDS:.*apiVersion.*<string>.*APIVersion defines.*spec.*<Object>.*Specification of Foo`); err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("kubectl explain works to explain CR properties recursively")
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural+".metadata", `(?s)DESCRIPTION:.*Standard object's metadata.*FIELDS:.*creationTimestamp.*<string>.*CreationTimestamp is a timestamp`); err != nil {
			framework.Failf("%v", err)
		}
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural+".spec", `(?s)DESCRIPTION:.*Specification of Foo.*FIELDS:.*bars.*<\[\]Object>.*List of Bars and their specs`); err != nil {
			framework.Failf("%v", err)
		}
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural+".spec.bars", `(?s)(FIELD|RESOURCE):.*bars.*<\[\]Object>.*DESCRIPTION:.*List of Bars and their specs.*FIELDS:.*bazs.*<\[\]string>.*List of Bazs.*name.*<string>.*Name of Bar`); err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("kubectl explain works to return error when explain is called on property that doesn't exist")
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, "explain", crd.Crd.Spec.Names.Plural+".spec.bars2"); err == nil || !strings.Contains(err.Error(), `field "bars2" does not exist`) {
			framework.Failf("unexpected no error when explaining property that doesn't exist: %v", err)
		}

		if err := cleanupCRD(ctx, f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, with x-kubernetes-preserve-unknown-fields in object
		Description: Register a custom resource definition with x-kubernetes-preserve-unknown-fields in the top level object.
		Attempt to create and apply a change a custom resource, via kubectl; kubectl validation MUST accept unknown
		properties. Attempt kubectl explain; the output MUST contain a valid DESCRIPTION stanza.
	*/
	framework.ConformanceIt("works for CRD without validation schema", func(ctx context.Context) {
		crd, err := setupCRD(f, nil, "empty", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}

		meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
		ns := fmt.Sprintf("--namespace=%v", f.Namespace.Name)

		ginkgo.By("kubectl validation (kubectl create and apply) allows request with any unknown properties")
		randomCR := fmt.Sprintf(`{%s,"a":{"b":[{"c":"d"}]}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, randomCR, ns, "create", "-f", "-"); err != nil {
			framework.Failf("failed to create random CR %s for CRD without schema: %v", randomCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, randomCR, ns, "apply", "-f", "-"); err != nil {
			framework.Failf("failed to apply random CR %s for CRD without schema: %v", randomCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}

		ginkgo.By("kubectl explain works to explain CR without validation schema")
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural, `(?s)DESCRIPTION:.*<empty>`); err != nil {
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(ctx, f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, with x-kubernetes-preserve-unknown-fields at root
		Description: Register a custom resource definition with x-kubernetes-preserve-unknown-fields in the schema root.
		Attempt to create and apply a change a custom resource, via kubectl; kubectl validation MUST accept unknown
		properties. Attempt kubectl explain; the output MUST show the custom resource KIND.
	*/
	framework.ConformanceIt("works for CRD preserving unknown fields at the schema root", func(ctx context.Context) {
		crd, err := setupCRDAndVerifySchema(f, schemaPreserveRoot, nil, "unknown-at-root", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}

		meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
		ns := fmt.Sprintf("--namespace=%v", f.Namespace.Name)

		ginkgo.By("kubectl validation (kubectl create and apply) allows request with any unknown properties")
		randomCR := fmt.Sprintf(`{%s,"a":{"b":[{"c":"d"}]}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, randomCR, ns, "create", "-f", "-"); err != nil {
			framework.Failf("failed to create random CR %s for CRD that allows unknown properties at the root: %v", randomCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, randomCR, ns, "apply", "-f", "-"); err != nil {
			framework.Failf("failed to apply random CR %s for CRD without schema: %v", randomCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}

		ginkgo.By("kubectl explain works to explain CR")
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural, fmt.Sprintf(`(?s)KIND:.*%s`, crd.Crd.Spec.Names.Kind)); err != nil {
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(ctx, f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, with x-kubernetes-preserve-unknown-fields in embedded object
		Description: Register a custom resource definition with x-kubernetes-preserve-unknown-fields in an embedded object.
		Attempt to create and apply a change a custom resource, via kubectl; kubectl validation MUST accept unknown
		properties. Attempt kubectl explain; the output MUST show that x-preserve-unknown-properties is used on the
		nested field.
	*/
	framework.ConformanceIt("works for CRD preserving unknown fields in an embedded object", func(ctx context.Context) {
		crd, err := setupCRDAndVerifySchema(f, schemaPreserveNested, nil, "unknown-in-nested", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}

		meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
		ns := fmt.Sprintf("--namespace=%v", f.Namespace.Name)

		ginkgo.By("kubectl validation (kubectl create and apply) allows request with any unknown properties")
		randomCR := fmt.Sprintf(`{%s,"spec":{"a":null,"b":[{"c":"d"}]}}`, meta)
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, randomCR, ns, "create", "-f", "-"); err != nil {
			framework.Failf("failed to create random CR %s for CRD that allows unknown properties in a nested object: %v", randomCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}
		if _, err := e2ekubectl.RunKubectlInput(f.Namespace.Name, randomCR, ns, "apply", "-f", "-"); err != nil {
			framework.Failf("failed to apply random CR %s for CRD without schema: %v", randomCR, err)
		}
		if _, err := e2ekubectl.RunKubectl(f.Namespace.Name, ns, "delete", crd.Crd.Spec.Names.Plural, "test-cr"); err != nil {
			framework.Failf("failed to delete random CR: %v", err)
		}

		ginkgo.By("kubectl explain works to explain CR")
		if err := verifyKubectlExplain(f.Namespace.Name, crd.Crd.Spec.Names.Plural, `(?s)DESCRIPTION:.*preserve-unknown-properties in nested field for Testing`); err != nil {
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(ctx, f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, varying groups
		Description: Register multiple custom resource definitions spanning different groups and versions;
		OpenAPI definitions MUST be published for custom resource definitions.
	*/
	framework.ConformanceIt("works for multiple CRDs of different groups", func(ctx context.Context) {
		ginkgo.By("CRs in different groups (two CRDs) show up in OpenAPI documentation")
		crdFoo, err := setupCRD(f, schemaFoo, "foo", "v1")
		if err != nil {
			framework.Failf("%v", err)
		}
		crdWaldo, err := setupCRD(f, schemaWaldo, "waldo", "v1beta1")
		if err != nil {
			framework.Failf("%v", err)
		}
		if crdFoo.Crd.Spec.Group == crdWaldo.Crd.Spec.Group {
			framework.Failf("unexpected: CRDs should be of different group %v, %v", crdFoo.Crd.Spec.Group, crdWaldo.Crd.Spec.Group)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdWaldo, "v1beta1"), schemaWaldo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdFoo, "v1"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdWaldo); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, varying versions
		Description: Register a custom resource definition with multiple versions; OpenAPI definitions MUST be published
		for custom resource definitions.
	*/
	framework.ConformanceIt("works for multiple CRDs of same group but different versions", func(ctx context.Context) {
		ginkgo.By("CRs in the same group but different versions (one multiversion CRD) show up in OpenAPI documentation")
		crdMultiVer, err := setupCRD(f, schemaFoo, "multi-ver", "v2", "v3")
		if err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v3"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v2"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdMultiVer); err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("CRs in the same group but different versions (two CRDs) show up in OpenAPI documentation")
		crdFoo, err := setupCRD(f, schemaFoo, "common-group", "v4")
		if err != nil {
			framework.Failf("%v", err)
		}
		crdWaldo, err := setupCRD(f, schemaWaldo, "common-group", "v5")
		if err != nil {
			framework.Failf("%v", err)
		}
		if crdFoo.Crd.Spec.Group != crdWaldo.Crd.Spec.Group {
			framework.Failf("unexpected: CRDs should be of the same group %v, %v", crdFoo.Crd.Spec.Group, crdWaldo.Crd.Spec.Group)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdWaldo, "v5"), schemaWaldo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdFoo, "v4"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdWaldo); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, varying kinds
		Description: Register multiple custom resource definitions in the same group and version but spanning different kinds;
		OpenAPI definitions MUST be published for custom resource definitions.
	*/
	framework.ConformanceIt("works for multiple CRDs of same group and version but different kinds", func(ctx context.Context) {
		ginkgo.By("CRs in the same group and version but different kinds (two CRDs) show up in OpenAPI documentation")
		crdFoo, err := setupCRD(f, schemaFoo, "common-group", "v6")
		if err != nil {
			framework.Failf("%v", err)
		}
		crdWaldo, err := setupCRD(f, schemaWaldo, "common-group", "v6")
		if err != nil {
			framework.Failf("%v", err)
		}
		if crdFoo.Crd.Spec.Group != crdWaldo.Crd.Spec.Group {
			framework.Failf("unexpected: CRDs should be of the same group %v, %v", crdFoo.Crd.Spec.Group, crdWaldo.Crd.Spec.Group)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdWaldo, "v6"), schemaWaldo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdFoo, "v6"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := cleanupCRD(ctx, f, crdWaldo); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, version rename
		Description: Register a custom resource definition with multiple versions; OpenAPI definitions MUST be published
		for custom resource definitions. Rename one of the versions of the custom resource definition via a patch;
		OpenAPI definitions MUST update to reflect the rename.
	*/
	framework.ConformanceIt("updates the published spec when one version gets renamed", func(ctx context.Context) {
		ginkgo.By("set up a multi version CRD")
		crdMultiVer, err := setupCRD(f, schemaFoo, "multi-ver", "v2", "v3")
		if err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v3"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v2"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("rename a version")
		patch := []byte(`[
			{"op":"test","path":"/spec/versions/1/name","value":"v3"},
			{"op": "replace", "path": "/spec/versions/1/name", "value": "v4"}
		]`)
		crdMultiVer.Crd, err = crdMultiVer.APIExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Patch(ctx, crdMultiVer.Crd.Name, types.JSONPatchType, patch, metav1.PatchOptions{})
		if err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("check the new version name is served")
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v4"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		ginkgo.By("check the old version name is removed")
		if err := waitForDefinitionCleanup(f.ClientSet, definitionName(crdMultiVer, "v3")); err != nil {
			framework.Failf("%v", err)
		}
		ginkgo.By("check the other version is not changed")
		if err := waitForDefinition(f.ClientSet, definitionName(crdMultiVer, "v2"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		// TestCrd.Versions is different from TestCrd.Crd.Versions, we have to manually
		// update the name there. Used by cleanupCRD
		crdMultiVer.Crd.Spec.Versions[1].Name = "v4"
		if err := cleanupCRD(ctx, f, crdMultiVer); err != nil {
			framework.Failf("%v", err)
		}
	})

	/*
		Release: v1.16
		Testname: Custom Resource OpenAPI Publish, stop serving version
		Description: Register a custom resource definition with multiple versions. OpenAPI definitions MUST be published
		for custom resource definitions. Update the custom resource definition to not serve one of the versions. OpenAPI
		definitions MUST be updated to not contain the version that is no longer served.
	*/
	framework.ConformanceIt("removes definition from spec when one version gets changed to not be served", func(ctx context.Context) {
		ginkgo.By("set up a multi version CRD")
		crd, err := setupCRD(f, schemaFoo, "multi-to-single-ver", "v5", "v6alpha1")
		if err != nil {
			framework.Failf("%v", err)
		}
		// just double check. setupCRD() checked this for us already
		if err := waitForDefinition(f.ClientSet, definitionName(crd, "v6alpha1"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}
		if err := waitForDefinition(f.ClientSet, definitionName(crd, "v5"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("mark a version not serverd")
		crd.Crd, err = crd.APIExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(ctx, crd.Crd.Name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("%v", err)
		}
		crd.Crd.Spec.Versions[1].Served = false
		crd.Crd, err = crd.APIExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Update(ctx, crd.Crd, metav1.UpdateOptions{})
		if err != nil {
			framework.Failf("%v", err)
		}

		ginkgo.By("check the unserved version gets removed")
		if err := waitForDefinitionCleanup(f.ClientSet, definitionName(crd, "v6alpha1")); err != nil {
			framework.Failf("%v", err)
		}
		ginkgo.By("check the other version is not changed")
		if err := waitForDefinition(f.ClientSet, definitionName(crd, "v5"), schemaFoo); err != nil {
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(ctx, f, crd); err != nil {
			framework.Failf("%v", err)
		}
	})

	// Marked as flaky until https://github.com/kubernetes/kubernetes/issues/65517 is solved.
	f.It(f.WithFlaky(), "kubectl explain works for CR with the same resource name as built-in object.", func(ctx context.Context) {
		customServiceShortName := fmt.Sprintf("ksvc-%d", time.Now().Unix()) // make short name unique
		opt := func(crd *apiextensionsv1.CustomResourceDefinition) {
			crd.ObjectMeta = metav1.ObjectMeta{Name: "services." + crd.Spec.Group}
			crd.Spec.Names = apiextensionsv1.CustomResourceDefinitionNames{
				Plural:     "services",
				Singular:   "service",
				ListKind:   "ServiceList",
				Kind:       "Service",
				ShortNames: []string{customServiceShortName},
			}
		}
		crdSvc, err := setupCRDAndVerifySchemaWithOptions(f, schemaCustomService, schemaCustomService, "service", []string{"v1"}, opt)
		if err != nil {
			framework.Failf("%v", err)
		}

		if err := verifyKubectlExplain(f.Namespace.Name, customServiceShortName+".spec", `(?s)DESCRIPTION:.*Specification of CustomService.*FIELDS:.*dummy.*<string>.*Dummy property`); err != nil {
			_ = cleanupCRD(ctx, f, crdSvc) // need to remove the crd since its name is unchanged
			framework.Failf("%v", err)
		}

		if err := cleanupCRD(ctx, f, crdSvc); err != nil {
			framework.Failf("%v", err)
		}
	})
})

func setupCRD(f *framework.Framework, schema []byte, groupSuffix string, versions ...string) (*crd.TestCrd, error) {
	expect := schema
	if schema == nil {
		// to be backwards compatible, we expect CRD controller to treat
		// CRD with nil schema specially and publish an empty schema
		expect = []byte(`type: object`)
	}
	return setupCRDAndVerifySchema(f, schema, expect, groupSuffix, versions...)
}

func setupCRDAndVerifySchema(f *framework.Framework, schema, expect []byte, groupSuffix string, versions ...string) (*crd.TestCrd, error) {
	return setupCRDAndVerifySchemaWithOptions(f, schema, expect, groupSuffix, versions)
}

func setupCRDAndVerifySchemaWithOptions(f *framework.Framework, schema, expect []byte, groupSuffix string, versions []string, options ...crd.Option) (*crd.TestCrd, error) {
	group := fmt.Sprintf("%s-test-%s.example.com", f.BaseName, groupSuffix)
	if len(versions) == 0 {
		return nil, fmt.Errorf("require at least one version for CRD")
	}

	props := &apiextensionsv1.JSONSchemaProps{}
	if schema != nil {
		if err := yaml.Unmarshal(schema, props); err != nil {
			return nil, err
		}
	}

	options = append(options, func(crd *apiextensionsv1.CustomResourceDefinition) {
		var apiVersions []apiextensionsv1.CustomResourceDefinitionVersion
		for i, version := range versions {
			version := apiextensionsv1.CustomResourceDefinitionVersion{
				Name:    version,
				Served:  true,
				Storage: i == 0,
			}
			// set up validation when input schema isn't nil
			if schema != nil {
				version.Schema = &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: props,
				}
			} else {
				version.Schema = &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						XPreserveUnknownFields: pointer.BoolPtr(true),
						Type:                   "object",
					},
				}
			}
			apiVersions = append(apiVersions, version)
		}
		crd.Spec.Versions = apiVersions
	})
	crd, err := crd.CreateMultiVersionTestCRD(f, group, options...)
	if err != nil {
		return nil, fmt.Errorf("failed to create CRD: %w", err)
	}

	for _, v := range crd.Crd.Spec.Versions {
		if err := waitForDefinition(f.ClientSet, definitionName(crd, v.Name), expect); err != nil {
			return nil, fmt.Errorf("%v", err)
		}
	}
	return crd, nil
}

func cleanupCRD(ctx context.Context, f *framework.Framework, crd *crd.TestCrd) error {
	_ = crd.CleanUp(ctx)
	for _, v := range crd.Crd.Spec.Versions {
		name := definitionName(crd, v.Name)
		if err := waitForDefinitionCleanup(f.ClientSet, name); err != nil {
			return fmt.Errorf("%v", err)
		}
	}
	return nil
}

const waitSuccessThreshold = 10

// mustSucceedMultipleTimes calls f multiple times on success and only returns true if all calls are successful.
// This is necessary to avoid flaking tests where one call might hit a good apiserver while in HA other apiservers
// might be lagging behind. Calling f multiple times reduces the chance exponentially.
func mustSucceedMultipleTimes(n int, f func() (bool, error)) func() (bool, error) {
	return func() (bool, error) {
		for i := 0; i < n; i++ {
			ok, err := f()
			if err != nil || !ok {
				return ok, err
			}
		}
		return true, nil
	}
}

// waitForDefinition waits for given definition showing up in swagger with given schema.
// If schema is nil, only the existence of the given name is checked.
func waitForDefinition(c k8sclientset.Interface, name string, schema []byte) error {
	expect := spec.Schema{}
	if err := convertJSONSchemaProps(schema, &expect); err != nil {
		return err
	}

	err := waitForOpenAPISchema(c, func(spec *spec.Swagger) (bool, string) {
		d, ok := spec.SwaggerProps.Definitions[name]
		if !ok {
			return false, fmt.Sprintf("spec.SwaggerProps.Definitions[\"%s\"] not found", name)
		}
		if schema != nil {
			// drop properties and extension that we added
			dropDefaults(&d)
			if !apiequality.Semantic.DeepEqual(expect, d) {
				return false, fmt.Sprintf("spec.SwaggerProps.Definitions[\"%s\"] not match; expect: %v, actual: %v", name, expect, d)
			}
		}
		return true, ""
	})
	if err != nil {
		return fmt.Errorf("failed to wait for definition %q to be served with the right OpenAPI schema: %w", name, err)
	}
	return nil
}

// waitForDefinitionCleanup waits for given definition to be removed from swagger
func waitForDefinitionCleanup(c k8sclientset.Interface, name string) error {
	err := waitForOpenAPISchema(c, func(spec *spec.Swagger) (bool, string) {
		if _, ok := spec.SwaggerProps.Definitions[name]; ok {
			return false, fmt.Sprintf("spec.SwaggerProps.Definitions[\"%s\"] still exists", name)
		}
		return true, ""
	})
	if err != nil {
		return fmt.Errorf("failed to wait for definition %q not to be served anymore: %w", name, err)
	}
	return nil
}

func waitForOpenAPISchema(c k8sclientset.Interface, pred func(*spec.Swagger) (bool, string)) error {
	client := c.Discovery().RESTClient().(*rest.RESTClient).Client
	url := c.Discovery().RESTClient().Get().AbsPath("openapi", "v2").URL()
	lastMsg := ""
	etag := ""
	var etagSpec *spec.Swagger
	if err := wait.Poll(500*time.Millisecond, 60*time.Second, mustSucceedMultipleTimes(waitSuccessThreshold, func() (bool, error) {
		// download spec with etag support
		spec := &spec.Swagger{}
		req, err := http.NewRequest("GET", url.String(), nil)
		if err != nil {
			return false, err
		}
		req.Close = true // enforce a new connection to hit different HA API servers
		if len(etag) > 0 {
			req.Header.Set("If-None-Match", fmt.Sprintf(`"%s"`, etag))
		}
		resp, err := client.Do(req)
		if err != nil {
			return false, err
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusNotModified {
			spec = etagSpec
		} else if resp.StatusCode != http.StatusOK {
			return false, fmt.Errorf("unexpected response: %d", resp.StatusCode)
		} else if bs, err := io.ReadAll(resp.Body); err != nil {
			return false, err
		} else if err := json.Unmarshal(bs, spec); err != nil {
			return false, err
		} else {
			etag = strings.Trim(resp.Header.Get("ETag"), `"`)
			etagSpec = spec
		}

		var ok bool
		ok, lastMsg = pred(spec)
		return ok, nil
	})); err != nil {
		return fmt.Errorf("failed to wait for OpenAPI spec validating condition: %v; lastMsg: %s", err, lastMsg)
	}
	return nil
}

// convertJSONSchemaProps converts JSONSchemaProps in YAML to spec.Schema
func convertJSONSchemaProps(in []byte, out *spec.Schema) error {
	external := apiextensionsv1.JSONSchemaProps{}
	if err := yaml.UnmarshalStrict(in, &external); err != nil {
		return err
	}
	internal := apiextensions.JSONSchemaProps{}
	if err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(&external, &internal, nil); err != nil {
		return err
	}
	kubeOut := spec.Schema{}
	formatPostProcessor := validation.StripUnsupportedFormatsPostProcessorForVersion(environment.DefaultCompatibilityVersion())
	if err := validation.ConvertJSONSchemaPropsWithPostProcess(&internal, &kubeOut, formatPostProcessor); err != nil {
		return err
	}
	bs, err := json.Marshal(kubeOut)
	if err != nil {
		return err
	}
	return json.Unmarshal(bs, out)
}

// dropDefaults drops properties and extension that we added to a schema
func dropDefaults(s *spec.Schema) {
	delete(s.Properties, "metadata")
	delete(s.Properties, "apiVersion")
	delete(s.Properties, "kind")
	delete(s.Extensions, "x-kubernetes-group-version-kind")
	delete(s.Extensions, "x-kubernetes-selectable-fields")
}

func verifyKubectlExplain(ns, name, pattern string) error {
	result, err := e2ekubectl.RunKubectl(ns, "explain", name)
	if err != nil {
		return fmt.Errorf("failed to explain %s: %w", name, err)
	}
	r := regexp.MustCompile(pattern)
	if !r.Match([]byte(result)) {
		return fmt.Errorf("kubectl explain %s result {%s} doesn't match pattern {%s}", name, result, pattern)
	}
	return nil
}

// definitionName returns the openapi definition name for given CRD in given version
func definitionName(crd *crd.TestCrd, version string) string {
	return openapiutil.ToRESTFriendlyName(fmt.Sprintf("%s/%s/%s", crd.Crd.Spec.Group, version, crd.Crd.Spec.Names.Kind))
}

var schemaFoo = []byte(`description: Foo CRD for Testing
type: object
properties:
  spec:
    type: object
    description: Specification of Foo
    properties:
      bars:
        description: List of Bars and their specs.
        type: array
        items:
          type: object
          required:
          - name
          properties:
            name:
              description: Name of Bar.
              type: string
            age:
              description: Age of Bar.
              type: string
            feeling:
              description: Whether Bar is feeling great.
              type: string
              enum:
              - Great
              - Down
            bazs:
              description: List of Bazs.
              items:
                type: string
              type: array
  status:
    description: Status of Foo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object
          properties:
            name:
              description: Name of Bar.
              type: string
            available:
              description: Whether the Bar is installed.
              type: boolean
            quxType:
              description: Indicates to external qux type.
              pattern: in-tree|out-of-tree
              type: string`)

var schemaCustomService = []byte(`description: CustomService CRD for Testing
type: object
properties:
  spec:
    description: Specification of CustomService
    type: object
    properties:
      dummy:
        description: Dummy property.
        type: string
`)

var schemaWaldo = []byte(`description: Waldo CRD for Testing
type: object
properties:
  spec:
    description: Specification of Waldo
    type: object
    properties:
      dummy:
        description: Dummy property.
        type: object
  status:
    description: Status of Waldo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object`)

var schemaPreserveRoot = []byte(`description: preserve-unknown-properties at root for Testing
x-kubernetes-preserve-unknown-fields: true
type: object
properties:
  spec:
    description: Specification of Waldo
    type: object
    properties:
      dummy:
        description: Dummy property.
        type: object
  status:
    description: Status of Waldo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object`)

var schemaPreserveNested = []byte(`description: preserve-unknown-properties in nested field for Testing
type: object
properties:
  spec:
    description: Specification of Waldo
    type: object
    x-kubernetes-preserve-unknown-fields: true
    properties:
      dummy:
        description: Dummy property.
        type: object
  status:
    description: Status of Waldo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object`)
