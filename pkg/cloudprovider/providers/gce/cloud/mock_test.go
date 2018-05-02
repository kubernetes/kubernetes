/*
Copyright 2017 The Kubernetes Authors.

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

package cloud

import (
	"context"
	"reflect"
	"testing"

	alpha "google.golang.org/api/compute/v0.alpha"
	beta "google.golang.org/api/compute/v0.beta"
	ga "google.golang.org/api/compute/v1"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/filter"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

func TestMocks(t *testing.T) {
	t.Parallel()

	// This test uses Addresses, but the logic that is generated is the same for
	// other basic objects.
	const region = "us-central1"

	ctx := context.Background()
	pr := &SingleProjectRouter{"mock-project"}
	mock := NewMockGCE(pr)

	keyAlpha := meta.RegionalKey("key-alpha", region)
	keyBeta := meta.RegionalKey("key-beta", region)
	keyGA := meta.RegionalKey("key-ga", region)
	key := keyAlpha

	// Get not found.
	if _, err := mock.AlphaAddresses().Get(ctx, key); err == nil {
		t.Errorf("AlphaAddresses().Get(%v, %v) = _, nil; want error", ctx, key)
	}
	if _, err := mock.BetaAddresses().Get(ctx, key); err == nil {
		t.Errorf("BetaAddresses().Get(%v, %v) = _, nil; want error", ctx, key)
	}
	if _, err := mock.Addresses().Get(ctx, key); err == nil {
		t.Errorf("Addresses().Get(%v, %v) = _, nil; want error", ctx, key)
	}
	// Insert.
	{
		obj := &alpha.Address{}
		if err := mock.AlphaAddresses().Insert(ctx, keyAlpha, obj); err != nil {
			t.Errorf("AlphaAddresses().Insert(%v, %v, %v) = %v; want nil", ctx, key, obj, err)
		}
	}
	{
		obj := &beta.Address{}
		if err := mock.BetaAddresses().Insert(ctx, keyBeta, obj); err != nil {
			t.Errorf("BetaAddresses().Insert(%v, %v, %v) = %v; want nil", ctx, key, obj, err)
		}
	}
	{
		obj := &ga.Address{}
		if err := mock.Addresses().Insert(ctx, keyGA, &ga.Address{Name: "ga"}); err != nil {
			t.Errorf("Addresses().Insert(%v, %v, %v) = %v; want nil", ctx, key, obj, err)
		}
	}
	// Get across versions.
	if obj, err := mock.AlphaAddresses().Get(ctx, key); err != nil {
		t.Errorf("AlphaAddresses().Get(%v, %v) = %v, %v; want nil", ctx, key, obj, err)
	}
	if obj, err := mock.BetaAddresses().Get(ctx, key); err != nil {
		t.Errorf("BetaAddresses().Get(%v, %v) = %v, %v; want nil", ctx, key, obj, err)
	}
	if obj, err := mock.Addresses().Get(ctx, key); err != nil {
		t.Errorf("Addresses().Get(%v, %v) = %v, %v; want nil", ctx, key, obj, err)
	}
	// List across versions.
	want := map[string]bool{"key-alpha": true, "key-beta": true, "key-ga": true}
	{
		objs, err := mock.AlphaAddresses().List(ctx, region, filter.None)
		if err != nil {
			t.Errorf("AlphaAddresses().List(%v, %v, %v) = %v, %v; want _, nil", ctx, region, filter.None, objs, err)
		} else {
			got := map[string]bool{}
			for _, obj := range objs {
				got[obj.Name] = true
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("AlphaAddresses().List(); got %+v, want %+v", got, want)
			}
		}
	}
	{
		objs, err := mock.BetaAddresses().List(ctx, region, filter.None)
		if err != nil {
			t.Errorf("BetaAddresses().List(%v, %v, %v) = %v, %v; want _, nil", ctx, region, filter.None, objs, err)
		} else {
			got := map[string]bool{}
			for _, obj := range objs {
				got[obj.Name] = true
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("AlphaAddresses().List(); got %+v, want %+v", got, want)
			}
		}
	}
	{
		objs, err := mock.Addresses().List(ctx, region, filter.None)
		if err != nil {
			t.Errorf("Addresses().List(%v, %v, %v) = %v, %v; want _, nil", ctx, region, filter.None, objs, err)
		} else {
			got := map[string]bool{}
			for _, obj := range objs {
				got[obj.Name] = true
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("AlphaAddresses().List(); got %+v, want %+v", got, want)
			}
		}
	}
	// Delete across versions.
	if err := mock.AlphaAddresses().Delete(ctx, keyAlpha); err != nil {
		t.Errorf("AlphaAddresses().Delete(%v, %v) = %v; want nil", ctx, key, err)
	}
	if err := mock.BetaAddresses().Delete(ctx, keyBeta); err != nil {
		t.Errorf("BetaAddresses().Delete(%v, %v) = %v; want nil", ctx, key, err)
	}
	if err := mock.Addresses().Delete(ctx, keyGA); err != nil {
		t.Errorf("Addresses().Delete(%v, %v) = %v; want nil", ctx, key, err)
	}
	// Delete not found.
	if err := mock.AlphaAddresses().Delete(ctx, keyAlpha); err == nil {
		t.Errorf("AlphaAddresses().Delete(%v, %v) = nil; want error", ctx, key)
	}
	if err := mock.BetaAddresses().Delete(ctx, keyBeta); err == nil {
		t.Errorf("BetaAddresses().Delete(%v, %v) = nil; want error", ctx, key)
	}
	if err := mock.Addresses().Delete(ctx, keyGA); err == nil {
		t.Errorf("Addresses().Delete(%v, %v) = nil; want error", ctx, key)
	}
}
