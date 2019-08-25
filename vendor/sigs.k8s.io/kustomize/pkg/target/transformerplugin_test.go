// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target_test

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func writeDeployment(th *kusttest_test.KustTestHarness, path string) {
	th.WriteF(path, `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myDeployment
spec:
  template:
    metadata:
      labels:
        backend: awesome
    spec:
      containers:
      - name: whatever
        image: whatever
`)
}

func writeStringPrefixer(
	th *kusttest_test.KustTestHarness, path, name string) {
	th.WriteF(path, `
apiVersion: someteam.example.com/v1
kind: StringPrefixer
metadata:
  name: `+name+`
`)
}

func writeDatePrefixer(
	th *kusttest_test.KustTestHarness, path, name string) {
	th.WriteF(path, `
apiVersion: someteam.example.com/v1
kind: DatePrefixer
metadata:
  name: `+name+`
`)
}

func TestOrderedTransformers(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "StringPrefixer")

	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "DatePrefixer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")
	th.WriteK("/app", `
resources:
- deployment.yaml
transformers:
- peachPrefixer.yaml
- date1Prefixer.yaml
- applePrefixer.yaml
- date2Prefixer.yaml
`)
	writeDeployment(th, "/app/deployment.yaml")
	writeStringPrefixer(th, "/app/applePrefixer.yaml", "apple")
	writeStringPrefixer(th, "/app/peachPrefixer.yaml", "peach")
	writeDatePrefixer(th, "/app/date1Prefixer.yaml", "date1")
	writeDatePrefixer(th, "/app/date2Prefixer.yaml", "date2")
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	// TODO: Fix #1164; the value of the name: field below
	// should be: 2018-05-11-peach-2018-05-11-apple-myDeployment
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: 2018-05-11-apple-2018-05-11-apple-myDeployment
spec:
  template:
    metadata:
      labels:
        backend: awesome
    spec:
      containers:
      - image: whatever
        name: whatever
`)
}

func TestPluginsNotEnabled(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "StringPrefixer")

	th := kusttest_test.NewKustTestHarness(t, "/app")
	th.WriteK("/app", `
transformers:
- stringPrefixer.yaml
`)
	writeStringPrefixer(th, "/app/stringPrefixer.yaml", "apple")

	_, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "unable to load plugin StringPrefixer") {
		t.Fatalf("unexpected err: %v", err)
	}
}

func TestSedTransformer(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildExecPlugin(
		"someteam.example.com", "v1", "SedTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")
	th.WriteK("/app", `
resources:
- configmap.yaml

transformers:
- sed-transformer.yaml

configMapGenerator:
- name: test
  literals:
  - FOO=$FOO
  - BAR=$BAR
`)
	th.WriteF("/app/sed-transformer.yaml", `
apiVersion: someteam.example.com/v1
kind: SedTransformer
metadata:
  name: some-random-name
argsFromFile: sed-input.txt
`)
	th.WriteF("/app/sed-input.txt", `
s/$FOO/foo/g
s/$BAR/bar/g
`)

	th.WriteF("/app/configmap.yaml", `
apiVersion: v1
kind: ConfigMap
metadata:
  name: configmap-a
  annotations:
    kustomize.k8s.io/Generated: "false"
data:
  foo: $FOO
`)

	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  foo: foo
kind: ConfigMap
metadata:
  annotations:
    kustomize.k8s.io/Generated: "false"
  name: configmap-a
---
apiVersion: v1
data:
  BAR: bar
  FOO: foo
kind: ConfigMap
metadata:
  annotations: {}
  name: test-k4bkhftttd
`)
}

func TestTransformedTransformers(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "StringPrefixer")

	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "DatePrefixer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app/overlay")

	th.WriteK("/app/base", `
resources:
- stringPrefixer.yaml
transformers:
- datePrefixer.yaml
`)
	writeStringPrefixer(th, "/app/base/stringPrefixer.yaml", "apple")
	writeDatePrefixer(th, "/app/base/datePrefixer.yaml", "date")

	th.WriteK("/app/overlay", `
resources:
- deployment.yaml
transformers:
- ../base
`)
	writeDeployment(th, "/app/overlay/deployment.yaml")

	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: 2018-05-11-apple-myDeployment
spec:
  template:
    metadata:
      labels:
        backend: awesome
    spec:
      containers:
      - image: whatever
        name: whatever
`)
}
