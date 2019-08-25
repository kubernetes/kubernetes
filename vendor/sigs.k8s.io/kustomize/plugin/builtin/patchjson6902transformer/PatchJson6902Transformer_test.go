// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func TestPatchJson6902Transformer(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "PatchJson6902Transformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	th.WriteF("/app/jsonpatch.json", `[
    {"op": "add", "path": "/spec/replica", "value": "3"}
]`)

	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: PatchJson6902Transformer
metadata:
  name: notImportantHere
patches:
- target:
    group: apps
    version: v1
    kind: Deployment
    name: myDeploy
  path: jsonpatch.json
`, `
apiVersion: apps/v1
metadata:
  name: myDeploy
kind: Deployment
spec:
  replica: 1
`)

	th.AssertActualEqualsExpected(rm, `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myDeploy
spec:
  replica: "3"
`)
}
