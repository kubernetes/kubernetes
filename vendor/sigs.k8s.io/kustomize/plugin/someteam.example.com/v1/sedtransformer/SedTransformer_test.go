// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func TestSedTransformer(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildExecPlugin("someteam.example.com", "v1", "SedTransformer")
	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	th.WriteF("/app/sed-input.txt", `
s/$FRUIT/orange/g
s/$VEGGIE/tomato/g
`)

	rm := th.LoadAndRunTransformer(`
apiVersion: someteam.example.com/v1
kind: SedTransformer
metadata:
  name: notImportantHere
argsOneLiner: s/one/two/g
argsFromFile: sed-input.txt
`,
		`apiVersion: apps/v1
kind: MeatBall
metadata:
  name: notImportantHere
beans: one one one one
fruit: $FRUIT
vegetable: $VEGGIE
`)

	th.AssertActualEqualsExpected(rm, `
apiVersion: apps/v1
beans: two two two two
fruit: orange
kind: MeatBall
metadata:
  annotations: {}
  name: notImportantHere
vegetable: tomato
`)
}
