// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/pkg/gvk"
)

func TestMakeDefaultConfig(t *testing.T) {
	// Confirm default can be made without fatal error inside call.
	_ = MakeDefaultConfig()
}

func TestFromFiles(t *testing.T) {

	ldr := loadertest.NewFakeLoader("/app")
	ldr.AddFile("/app/config.yaml", []byte(`
namePrefix:
- path: nameprefix/path
  kind: SomeKind
`))
	tcfg, err := NewFactory(ldr).FromFiles([]string{"/app/config.yaml"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := &TransformerConfig{
		NamePrefix: []FieldSpec{
			{
				Gvk:  gvk.Gvk{Kind: "SomeKind"},
				Path: "nameprefix/path",
			},
		},
	}
	if !reflect.DeepEqual(tcfg, expected) {
		t.Fatalf("expected %v\n but go6t %v\n", expected, tcfg)
	}
}
