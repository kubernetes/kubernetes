// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

const (
	content = `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
---
apiVersion: v1
kind: Secret
metadata:
  name: secret1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deploy1
spec:
  template:
    spec:
      containers:
      - env:
          name: CM_FOO
          valueFrom:
            configMapKeyRef:
              key: someKey
              name: cm1
        envFrom:
          configMapRef:
            key: someKey
            name: cm1
          secretRef:
            key: someKey
            name: secret1
        image: nginx:1.7.9
        name: nginx
`
	inv = `
apiVersion: v1
kind: ConfigMap
metadata:
  annotations:
    kustomize.config.k8s.io/Inventory: '{"current":{"apps_v1_Deployment|~X|deploy1":null,"~G_v1_ConfigMap|~X|cm1":null,"~G_v1_Secret|~X|secret1":null}}'
    kustomize.config.k8s.io/InventoryHash: h44788gt7g
  name: pruneCM
  namespace: default
`
)

func TestInventoryTransformerCollect(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "InventoryTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: InventoryTransformer
metadata:
  name: notImportantHere
policy: GarbageCollect
name: pruneCM
namespace: default
`, content)

	th.AssertActualEqualsExpected(rm, inv)
}

func TestInventoryTransformerIgnore(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "InventoryTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: InventoryTransformer
metadata:
  name: notImportantHere
policy: GarbageIgnore
name: pruneCM
namespace: default
`, content)

	th.AssertActualEqualsExpected(rm, content+"---"+inv)
}

func TestInventoryTransformerDefaultPolicy(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "InventoryTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: InventoryTransformer
metadata:
  name: notImportantHere
name: pruneCM
namespace: default
`, content)

	th.AssertActualEqualsExpected(rm, content+"---"+inv)
}
