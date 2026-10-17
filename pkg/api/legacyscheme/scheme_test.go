/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package legacyscheme

import (
	"io"
	"testing"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2"
)

// initTestObject is a minimal runtime.Object usable with AddKnownTypeWithName.
type initTestObject struct {
	runtime.TypeMeta
}

func (o *initTestObject) DeepCopyObject() runtime.Object {
	out := *o
	return &out
}

// initTestSource/Dest let the test register a conversion func that
// isn't one of NewScheme's built-in default conversions.
type initTestSource int
type initTestDest int

// TestInit covers the behavior that makes Init safe to call more
// than once: Scheme keeps its identity (so anything that captured a pointer
// to it earlier observes the result), while its content reflects only the
// current init func state, not an accumulation of past calls' states. The
// same checks run both before and after flipping enabled, so each one is
// exercised in both states.
func TestInit(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "legacyscheme.test", Version: "v1", Kind: "InitTestObject"}
	var enabled bool
	Scheme.AddInitFunc(func(logger klog.Logger, scheme *runtime.Scheme) error {
		if enabled {
			scheme.AddKnownTypeWithName(gvk, &initTestObject{})
		}
		return nil
	})

	// Registered once, before the first Init call; every Init call
	// must carry it over (Converter's own Clone correctness is covered in
	// detail by the conversion package's own tests).
	if err := Scheme.Converter().RegisterUntypedConversionFunc(
		(*initTestSource)(nil), (*initTestDest)(nil),
		func(a, b interface{}, s conversion.Scope) error {
			*b.(*initTestDest) = initTestDest(*a.(*initTestSource))
			return nil
		},
	); err != nil {
		t.Fatal(err)
	}

	check := func(label string, wantRegistered bool) {
		t.Helper()
		if got := Scheme.Recognizes(gvk); got != wantRegistered {
			t.Errorf("%s: Scheme.Recognizes = %v, want %v", label, got, wantRegistered)
		}
		err := Codecs.LegacyCodec(gvk.GroupVersion()).Encode(&initTestObject{}, io.Discard)
		if wantRegistered && err != nil {
			t.Errorf("%s: Codecs must reflect the current Scheme: %v", label, err)
		}
		if !wantRegistered && err == nil {
			t.Errorf("%s: expected an error encoding a type that is not registered", label)
		}
		if ParameterCodec == nil {
			t.Errorf("%s: ParameterCodec must be set", label)
		}
		var src initTestSource = 42
		var dst initTestDest
		if err := Scheme.Convert(&src, &dst, nil); err != nil {
			t.Errorf("%s: Scheme must keep the conversion func registered before the first Init: %v", label, err)
		} else if dst != 42 {
			t.Errorf("%s: conversion func did not run, got %v", label, dst)
		}
	}

	if err := Scheme.Init(klog.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	check("disabled", false)

	prevScheme := Scheme
	enabled = true
	if err := Scheme.Init(klog.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if Scheme != prevScheme {
		t.Error("Init must mutate Scheme in place, not replace it with a new instance")
	}
	check("enabled", true)
	if !prevScheme.Recognizes(gvk) {
		t.Error("a pointer captured before Init ran must observe the new state, since Scheme's identity never changes")
	}
}
