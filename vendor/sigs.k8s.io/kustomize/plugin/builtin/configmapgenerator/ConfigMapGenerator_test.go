// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func TestConfigMapGenerator(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "ConfigMapGenerator")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	th.WriteF("/app/devops.env", `
SERVICE_PORT=32
`)
	th.WriteF("/app/uxteam.env", `
COLOR=red
`)

	rm := th.LoadAndRunGenerator(`
apiVersion: builtin
kind: ConfigMapGenerator
metadata:
  name: myMapGen
name: myMap
envs:
- devops.env
- uxteam.env
literals:
- FRUIT=apple
- VEGETABLE=carrot
`)

	th.AssertActualEqualsExpected(rm, `
apiVersion: v1
data:
  COLOR: red
  FRUIT: apple
  SERVICE_PORT: "32"
  VEGETABLE: carrot
kind: ConfigMap
metadata:
  name: myMap
`)
}
