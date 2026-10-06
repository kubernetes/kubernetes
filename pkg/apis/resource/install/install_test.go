/*
Copyright 2022 The Kubernetes Authors.

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

package install

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	internal "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/v1beta1"
	"k8s.io/kubernetes/pkg/features"
)

func TestResourceVersioner(t *testing.T) {
	claim := internal.ResourceClaim{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "10"}}
	version, err := meta.NewAccessor().ResourceVersion(&claim)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	claimList := internal.ResourceClaimList{ListMeta: metav1.ListMeta{ResourceVersion: "10"}}
	version, err = meta.NewAccessor().ResourceVersion(&claimList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	claim := internal.ResourceClaim{}
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(schema.GroupVersion{Group: "resource.k8s.io", Version: "v1beta2"}), &claim)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := internal.ResourceClaim{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != "resource.k8s.io/v1beta2" || other.Kind != "ResourceClaim" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestUnversioned(t *testing.T) {
	for _, obj := range []runtime.Object{
		&metav1.Status{},
	} {
		if unversioned, ok := legacyscheme.Scheme.IsUnversioned(obj); !unversioned || !ok {
			t.Errorf("%v is expected to be unversioned", reflect.TypeOf(obj))
		}
	}
}

func TestV1beta1FeatureGate(t *testing.T) {
	gvk := v1beta1.SchemeGroupVersion.WithKind("ResourceClaim")

	for name, tc := range map[string]struct {
		enabled       bool
		wantInstalled bool
	}{
		"disabled-by-default": {enabled: false, wantInstalled: false},
		"enabled":             {enabled: true, wantInstalled: true},
	} {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceV1beta1API, tc.enabled)

			scheme := runtime.NewScheme()
			Install(scheme)
			if err := scheme.Init(klog.Background()); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if got := scheme.Recognizes(gvk); got != tc.wantInstalled {
				t.Errorf("Scheme.Recognizes(%v) = %v, want %v", gvk, got, tc.wantInstalled)
			}
		})
	}
}
