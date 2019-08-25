// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
	"sigs.k8s.io/kustomize/plugin"
)

func TestImageTagTransformer(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "ImageTagTransformer")

	th := kusttest_test.NewKustTestPluginHarness(t, "/app")

	rm := th.LoadAndRunTransformer(`
apiVersion: builtin
kind: ImageTagTransformer
metadata:
  name: notImportantHere
imageTag:
  name: nginx
  newTag: v2
`, `
group: apps
apiVersion: v1
kind: Deployment
metadata:
  name: deploy1
spec:
  template:
    spec:
      initContainers:
      - name: nginx2
        image: my-nginx:1.8.0
      - name: init-alpine
        image: alpine:1.8.0
      containers:
      - name: ngnix
        image: nginx:1.7.9
      - name: repliaced-with-digest
        image: foobar:1
      - name: postgresdb
        image: postgres:1.8.0
`)

	th.AssertActualEqualsExpected(rm, `
apiVersion: v1
group: apps
kind: Deployment
metadata:
  name: deploy1
spec:
  template:
    spec:
      containers:
      - image: nginx:v2
        name: ngnix
      - image: foobar:1
        name: repliaced-with-digest
      - image: postgres:1.8.0
        name: postgresdb
      initContainers:
      - image: my-nginx:1.8.0
        name: nginx2
      - image: alpine:1.8.0
        name: init-alpine
`)
}
