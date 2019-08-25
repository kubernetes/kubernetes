// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func TestNamespaceTransformer1(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "NamespaceTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: NamespaceTransformer
metadata:
  name: notImportantHere
namespace: test
fieldSpecs:
- path: metadata/namespace
  create: true
`, `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm2
  namespace: foo
---
apiVersion: v1
kind: Namespace
metadata:
  name: ns1
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: default
  namespace: system
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: service-account
  namespace: system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: manager-rolebinding
subjects:
- kind: ServiceAccount
  name: default
  namespace: system
- kind: ServiceAccount
  name: service-account
  namespace: system
- kind: ServiceAccount
  name: another
  namespace: random
---
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: crd
`)

	th.AssertActualEqualsExpected(rm, `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
  namespace: test
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm2
  namespace: test
---
apiVersion: v1
kind: Namespace
metadata:
  name: ns1
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: default
  namespace: test
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: service-account
  namespace: test
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: manager-rolebinding
subjects:
- kind: ServiceAccount
  name: default
  namespace: test
- kind: ServiceAccount
  name: service-account
  namespace: test
- kind: ServiceAccount
  name: another
  namespace: random
---
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: crd
`)
}

func TestNamespaceTransformerClusterLevelKinds(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "NamespaceTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	const noChangeExpected = `
apiVersion: v1
kind: Namespace
metadata:
  name: ns1
---
kind: CustomResourceDefinition
metadata:
  name: crd1
---
kind: ClusterRole
metadata:
  name: cr1
---
kind: ClusterRoleBinding
metadata:
  name: crb1
---
kind: PersistentVolume
metadata:
  name: pv1
`
	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: NamespaceTransformer
metadata:
  name: notImportantHere
namespace: test
fieldSpecs:
- path: metadata/namespace
  create: true
`, noChangeExpected)

	th.AssertActualEqualsExpected(rm, noChangeExpected)
}
