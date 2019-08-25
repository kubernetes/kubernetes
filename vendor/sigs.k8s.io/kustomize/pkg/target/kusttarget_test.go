// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target_test

import (
	"encoding/base64"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	. "sigs.k8s.io/kustomize/pkg/target"
	"sigs.k8s.io/kustomize/pkg/types"
)

const (
	kustomizationContent = `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: foo-
nameSuffix: -bar
namespace: ns1
commonLabels:
  app: nginx
commonAnnotations:
  note: This is a test annotation
resources:
  - deployment.yaml
  - namespace.yaml
generatorOptions:
  disableNameSuffixHash: false
configMapGenerator:
- name: literalConfigMap
  literals:
  - DB_USERNAME=admin
  - DB_PASSWORD=somepw
secretGenerator:
- name: secret
  literals:
    - DB_USERNAME=admin
    - DB_PASSWORD=somepw
  type: Opaque
patchesJson6902:
- target:
    group: apps
    version: v1
    kind: Deployment
    name: dply1
  path: jsonpatch.json
`
	deploymentContent = `
apiVersion: apps/v1
metadata:
  name: dply1
kind: Deployment
`
	namespaceContent = `
apiVersion: v1
kind: Namespace
metadata:
  name: ns1
`
	jsonpatchContent = `[
    {"op": "add", "path": "/spec/replica", "value": "3"}
]`
)

func TestResources(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/whatever")
	th.WriteK("/whatever/", kustomizationContent)
	th.WriteF("/whatever/deployment.yaml", deploymentContent)
	th.WriteF("/whatever/namespace.yaml", namespaceContent)
	th.WriteF("/whatever/jsonpatch.json", jsonpatchContent)

	resources := []*resource.Resource{
		th.RF().FromMapWithName("dply1", map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      "foo-dply1-bar",
				"namespace": "ns1",
				"labels": map[string]interface{}{
					"app": "nginx",
				},
				"annotations": map[string]interface{}{
					"note": "This is a test annotation",
				},
			},
			"spec": map[string]interface{}{
				"replica": "3",
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": "nginx",
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"annotations": map[string]interface{}{
							"note": "This is a test annotation",
						},
						"labels": map[string]interface{}{
							"app": "nginx",
						},
					},
				},
			},
		}),
		th.RF().FromMapWithName("ns1", map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Namespace",
			"metadata": map[string]interface{}{
				"name": "foo-ns1-bar",
				"labels": map[string]interface{}{
					"app": "nginx",
				},
				"annotations": map[string]interface{}{
					"note": "This is a test annotation",
				},
			},
		}),
		th.RF().FromMapWithName("literalConfigMap",
			map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "ConfigMap",
				"metadata": map[string]interface{}{
					"name":      "foo-literalConfigMap-bar-8d2dkb8k24",
					"namespace": "ns1",
					"labels": map[string]interface{}{
						"app": "nginx",
					},
					"annotations": map[string]interface{}{
						"note": "This is a test annotation",
					},
				},
				"data": map[string]interface{}{
					"DB_USERNAME": "admin",
					"DB_PASSWORD": "somepw",
				},
			}),
		th.RF().FromMapWithName("secret",
			map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Secret",
				"metadata": map[string]interface{}{
					"name":      "foo-secret-bar-9btc7bt4kb",
					"namespace": "ns1",
					"labels": map[string]interface{}{
						"app": "nginx",
					},
					"annotations": map[string]interface{}{
						"note": "This is a test annotation",
					},
				},
				"type": ifc.SecretTypeOpaque,
				"data": map[string]interface{}{
					"DB_USERNAME": base64.StdEncoding.EncodeToString([]byte("admin")),
					"DB_PASSWORD": base64.StdEncoding.EncodeToString([]byte("somepw")),
				},
			}),
	}

	expected := resmap.New()
	for _, r := range resources {
		if err := expected.Append(r); err != nil {
			t.Fatalf("unexpected error %v", err)
		}
	}

	actual, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("unexpected Resources error %v", err)
	}

	if err = expected.ErrorIfNotEqualLists(actual); err != nil {
		t.Fatalf("unexpected inequality: %v", err)
	}
}

func TestKustomizationNotFound(t *testing.T) {
	_, err := NewKustTarget(
		loadertest.NewFakeLoader("/foo"), nil, nil, nil)
	if err == nil {
		t.Fatalf("expected an error")
	}
	if err.Error() !=
		`unable to find one of 'kustomization.yaml', 'kustomization.yml' or 'Kustomization' in directory '/foo'` {
		t.Fatalf("unexpected error: %q", err)
	}
}

func TestResourceNotFound(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/whatever")
	th.WriteK("/whatever", kustomizationContent)
	_, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err == nil {
		t.Fatalf("Didn't get the expected error for an unknown resource")
	}
	if !strings.Contains(err.Error(), `cannot read file`) {
		t.Fatalf("unexpected error: %q", err)
	}
}

func findSecret(m resmap.ResMap) *resource.Resource {
	for _, r := range m.Resources() {
		if r.OrgId().Kind == "Secret" {
			return r
		}
	}
	return nil
}

func TestDisableNameSuffixHash(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/whatever")
	th.WriteK("/whatever/", kustomizationContent)
	th.WriteF("/whatever/deployment.yaml", deploymentContent)
	th.WriteF("/whatever/namespace.yaml", namespaceContent)
	th.WriteF("/whatever/jsonpatch.json", jsonpatchContent)

	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("unexpected Resources error %v", err)
	}
	secret := findSecret(m)
	if secret == nil {
		t.Errorf("Expected to find a Secret")
	}
	if secret.GetName() != "foo-secret-bar-9btc7bt4kb" {
		t.Errorf("unexpected secret resource name: %s", secret.GetName())
	}

	th.WriteK("/whatever/",
		strings.Replace(kustomizationContent,
			"disableNameSuffixHash: false",
			"disableNameSuffixHash: true", -1))
	m, err = th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("unexpected Resources error %v", err)
	}
	secret = findSecret(m)
	if secret == nil {
		t.Errorf("Expected to find a Secret")
	}
	if secret.GetName() != "foo-secret-bar" { // No hash at end.
		t.Errorf("unexpected secret resource name: %s", secret.GetName())
	}
}

func TestIssue596AllowDirectoriesThatAreSubstringsOfEachOther(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlays/aws-sandbox2.us-east-1")
	th.WriteK("/app/base", "")
	th.WriteK("/app/overlays/aws", `
resources:
- ../../base
`)
	th.WriteK("/app/overlays/aws-nonprod", `
resources:
- ../aws
`)
	th.WriteK("/app/overlays/aws-sandbox2.us-east-1", `
resources:
- ../aws-nonprod
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, "")
}

// To simplify tests, these vars specified in alphabetical order.
var someVars = []types.Var{
	{
		Name: "AWARD",
		ObjRef: types.Target{
			APIVersion: "v7",
			Gvk:        gvk.Gvk{Kind: "Service"},
			Name:       "nobelPrize"},
		FieldRef: types.FieldSelector{FieldPath: "some.arbitrary.path"},
	},
	{
		Name: "BIRD",
		ObjRef: types.Target{
			APIVersion: "v300",
			Gvk:        gvk.Gvk{Kind: "Service"},
			Name:       "heron"},
		FieldRef: types.FieldSelector{FieldPath: "metadata.name"},
	},
	{
		Name: "FRUIT",
		ObjRef: types.Target{
			Gvk:  gvk.Gvk{Kind: "Service"},
			Name: "apple"},
		FieldRef: types.FieldSelector{FieldPath: "metadata.name"},
	},
	{
		Name: "VEGETABLE",
		ObjRef: types.Target{
			Gvk:  gvk.Gvk{Kind: "Leafy"},
			Name: "kale"},
		FieldRef: types.FieldSelector{FieldPath: "metadata.name"},
	},
}

func TestGetAllVarsSimple(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	th.WriteK("/app", `
vars:
  - name: AWARD
    objref:
      kind: Service
      name: nobelPrize
      apiVersion: v7
    fieldref:
      fieldpath: some.arbitrary.path
  - name: BIRD
    objref:
      kind: Service
      name: heron
      apiVersion: v300
`)
	ra, err := th.MakeKustTarget().AccumulateTarget()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	vars := ra.Vars()
	if len(vars) != 2 {
		t.Fatalf("unexpected size %d", len(vars))
	}
	for i := range vars[:2] {
		if !reflect.DeepEqual(vars[i], someVars[i]) {
			t.Fatalf("unexpected var[%d]:\n  %v\n  %v", i, vars[i], someVars[i])
		}
	}
}

func TestGetAllVarsNested(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlays/o2")
	th.WriteK("/app/base", `
vars:
  - name: AWARD
    objref:
      kind: Service
      name: nobelPrize
      apiVersion: v7
    fieldref:
      fieldpath: some.arbitrary.path
  - name: BIRD
    objref:
      kind: Service
      name: heron
      apiVersion: v300
`)
	th.WriteK("/app/overlays/o1", `
vars:
  - name: FRUIT
    objref:
      kind: Service
      name: apple
resources:
- ../../base
`)
	th.WriteK("/app/overlays/o2", `
vars:
  - name: VEGETABLE
    objref:
      kind: Leafy
      name: kale
resources:
- ../o1
`)
	ra, err := th.MakeKustTarget().AccumulateTarget()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	vars := ra.Vars()
	if len(vars) != 4 {
		for i, v := range vars {
			fmt.Printf("%v: %v\n", i, v)
		}
		t.Fatalf("expected 4 vars, got %d", len(vars))
	}
	for i := range vars {
		if !reflect.DeepEqual(vars[i], someVars[i]) {
			t.Fatalf("unexpected var[%d]:\n  %v\n  %v", i, vars[i], someVars[i])
		}
	}
}

func TestVarCollisionsForbidden(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlays/o2")
	th.WriteK("/app/base", `
vars:
  - name: AWARD
    objref:
      kind: Service
      name: nobelPrize
      apiVersion: v7
    fieldref:
      fieldpath: some.arbitrary.path
  - name: BIRD
    objref:
      kind: Service
      name: heron
      apiVersion: v300
`)
	th.WriteK("/app/overlays/o1", `
vars:
  - name: AWARD
    objref:
      kind: Service
      name: academy
resources:
- ../../base
`)
	th.WriteK("/app/overlays/o2", `
vars:
  - name: VEGETABLE
    objref:
      kind: Leafy
      name: kale
resources:
- ../o1
`)
	_, err := th.MakeKustTarget().AccumulateTarget()
	if err == nil {
		t.Fatalf("expected var collision")
	}
	if !strings.Contains(err.Error(),
		"var 'AWARD' already encountered") {
		t.Fatalf("unexpected error: %v", err)
	}
}
