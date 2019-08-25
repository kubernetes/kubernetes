// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package plugins_test

import (
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	. "sigs.k8s.io/kustomize/pkg/plugins"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/plugin"
)

const (
	secretGenerator = `
apiVersion: builtin
kind: SecretGenerator
metadata:
  name: secretGenerator
name: mySecret
behavior: merge
envFiles:
- a.env
- b.env
valueFiles:
- longsecret.txt
literals:
- FRUIT=apple
- VEGETABLE=carrot
`
	someServiceGenerator = `
apiVersion: someteam.example.com/v1
kind: SomeServiceGenerator
metadata:
  name: myServiceGenerator
service: my-service
port: "12345"
`
)

func TestLoader(t *testing.T) {
	tc := plugin.NewEnvForTest(t).Set()
	defer tc.Reset()

	tc.BuildGoPlugin(
		"builtin", "", "SecretGenerator")
	tc.BuildGoPlugin(
		"someteam.example.com", "v1", "SomeServiceGenerator")

	rmF := resmap.NewFactory(resource.NewFactory(
		kunstruct.NewKunstructuredFactoryImpl()))

	l := NewLoader(ActivePluginConfig(), rmF)
	if l == nil {
		t.Fatal("expect non-nil loader")
	}

	ldr := loadertest.NewFakeLoader("/foo")

	m, err := rmF.NewResMapFromBytes([]byte(
		someServiceGenerator + "---\n" + secretGenerator))
	if err != nil {
		t.Fatal(err)
	}

	_, err = l.LoadGenerators(ldr, m)
	if err != nil {
		t.Fatal(err)
	}
}
