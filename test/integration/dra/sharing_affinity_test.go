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

package dra

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestSharingAffinity_GateEnabledPersists(t *testing.T) {
	server := startSharingAffinityServer(t, true, framework.SharedEtcd())
	forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
		slice := sharingAffinitySlice()
		created := createSharingAffinitySlice(t, ctx, client, sharingAffinityObject(t, gv, slice))
		assertSharingAffinity(t, created, slice.Spec.SharingAffinity)
		stored := getSharingAffinitySlice(t, ctx, client, created.GetName())
		assertSharingAffinity(t, stored, slice.Spec.SharingAffinity)
	})
}

func TestSharingAffinity_ApplyAtomicList(t *testing.T) {
	server := startSharingAffinityServer(t, true, framework.SharedEtcd())
	forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
		slice := sharingAffinitySlice()
		slice.GenerateName = ""
		slice.Name = "sharing-affinity-apply-" + gv.Version
		created, err := client.Apply(ctx, slice.Name, sharingAffinityObject(t, gv, slice), metav1.ApplyOptions{FieldManager: "first-publisher"})
		if err != nil {
			t.Fatalf("apply initial sharing affinity: %v", err)
		}
		assertSharingAffinity(t, created, slice.Spec.SharingAffinity)

		replacement := slice.DeepCopy()
		replacement.Spec.SharingAffinity[0].Name = "tenant"
		obj := sharingAffinityObject(t, gv, replacement)
		if _, err := client.Apply(ctx, slice.Name, obj, metav1.ApplyOptions{FieldManager: "second-publisher"}); !apierrors.IsConflict(err) {
			t.Fatalf("applying a different key must conflict with the atomic list owner, got: %v", err)
		}
		stored := getSharingAffinitySlice(t, ctx, client, slice.Name)
		assertSharingAffinity(t, stored, slice.Spec.SharingAffinity)
		if stored.GetResourceVersion() != created.GetResourceVersion() {
			t.Fatal("conflicting apply changed the stored resourceVersion")
		}

		updated, err := client.Apply(ctx, slice.Name, obj, metav1.ApplyOptions{FieldManager: "second-publisher", Force: true})
		if err != nil {
			t.Fatalf("force apply replacement sharing affinity: %v", err)
		}
		assertSharingAffinity(t, updated, replacement.Spec.SharingAffinity)
		stored = getSharingAffinitySlice(t, ctx, client, slice.Name)
		assertSharingAffinity(t, stored, replacement.Spec.SharingAffinity)
	})
}

func TestSharingAffinity_GateDisabledCreateDrops(t *testing.T) {
	server := startSharingAffinityServer(t, false, framework.SharedEtcd())
	forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
		created := createSharingAffinitySlice(t, ctx, client, sharingAffinityObject(t, gv, sharingAffinitySlice()))
		assertSharingAffinity(t, created, nil)
		stored := getSharingAffinitySlice(t, ctx, client, created.GetName())
		assertSharingAffinity(t, stored, nil)
	})
}

func TestSharingAffinity_GateDisabledUpdateDropsNew(t *testing.T) {
	server := startSharingAffinityServer(t, false, framework.SharedEtcd())
	forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
		slice := sharingAffinitySlice()
		slice.Spec.SharingAffinity = nil
		created := createSharingAffinitySlice(t, ctx, client, sharingAffinityObject(t, gv, slice))
		stored := getSharingAffinitySlice(t, ctx, client, created.GetName())
		assertSharingAffinity(t, stored, nil)

		slice.Spec.SharingAffinity = sharingAffinitySlice().Spec.SharingAffinity
		update := sharingAffinityObject(t, gv, slice)
		update.SetName(stored.GetName())
		update.SetResourceVersion(stored.GetResourceVersion())
		update.SetLabels(map[string]string{"unrelated-update": "accepted"})
		updated, err := client.Update(ctx, update, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("introduce sharing affinity with gate disabled: %v", err)
		}
		assertSharingAffinity(t, updated, nil)
		stored = getSharingAffinitySlice(t, ctx, client, created.GetName())
		assertSharingAffinity(t, stored, nil)
		if stored.GetLabels()["unrelated-update"] != "accepted" {
			t.Fatal("unrelated label update was not persisted")
		}
	})
}

func TestSharingAffinity_GateDisabledUpdatePreservesExisting(t *testing.T) {
	// SharedEtcd allocates a new prefix on each call. Both servers must use
	// this exact config, not merely the same etcd endpoint.
	etcdOptions := framework.SharedEtcd()
	existing := map[schema.GroupVersion]*unstructured.Unstructured{}
	if !t.Run("enabled", func(t *testing.T) {
		server := startSharingAffinityServer(t, true, etcdOptions)
		forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
			slice := sharingAffinitySlice()
			created := createSharingAffinitySlice(t, ctx, client, sharingAffinityObject(t, gv, slice))
			assertSharingAffinity(t, created, slice.Spec.SharingAffinity)
			stored := getSharingAffinitySlice(t, ctx, client, created.GetName())
			assertSharingAffinity(t, stored, slice.Spec.SharingAffinity)
			existing[gv] = stored
		})
	}) {
		return
	}

	// The enabled subtest has torn down its server before gates change.
	t.Run("disabled", func(t *testing.T) {
		server := startSharingAffinityServer(t, false, etcdOptions)
		forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
			before, ok := existing[gv]
			if !ok {
				t.Fatalf("%s was not exercised before restart", gv)
			}
			stored := getSharingAffinitySlice(t, ctx, client, before.GetName())
			if stored.GetUID() != before.GetUID() || stored.GetResourceVersion() != before.GetResourceVersion() {
				t.Fatal("restart did not retain the original ResourceSlice UID and resourceVersion")
			}
			want := sharingAffinitySlice().Spec.SharingAffinity
			assertSharingAffinity(t, stored, want)
			stored.SetLabels(map[string]string{"unrelated-update": "accepted"})
			updated, err := client.Update(ctx, stored, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("update existing sharing affinity with gate disabled: %v", err)
			}
			assertSharingAffinity(t, updated, want)
			stored = getSharingAffinitySlice(t, ctx, client, before.GetName())
			assertSharingAffinity(t, stored, want)
			if stored.GetLabels()["unrelated-update"] != "accepted" {
				t.Fatal("unrelated label update was not persisted")
			}
		})
	})
}

func TestSharingAffinity_AdmissionRejects(t *testing.T) {
	server := startSharingAffinityServer(t, true, framework.SharedEtcd())
	forSharingAffinityVersions(t, server, func(t *testing.T, ctx context.Context, gv schema.GroupVersion, client dynamic.ResourceInterface) {
		for _, tc := range []struct {
			name   string
			mutate func(*resourceapi.ResourceSlice)
			field  string
			cause  metav1.CauseType
		}{
			{
				name: "invalid-cel",
				mutate: func(slice *resourceapi.ResourceSlice) {
					slice.Spec.SharingAffinity[0].Expression = "parameters."
				},
				field: "spec.sharingAffinity[0].expression",
				cause: metav1.CauseTypeFieldValueInvalid,
			},
			{
				name: "duplicate-name",
				mutate: func(slice *resourceapi.ResourceSlice) {
					slice.Spec.SharingAffinity = append(slice.Spec.SharingAffinity, slice.Spec.SharingAffinity[0])
				},
				field: "spec.sharingAffinity[1]",
				cause: metav1.CauseTypeFieldValueDuplicate,
			},
			{
				name: "empty-api-group",
				mutate: func(slice *resourceapi.ResourceSlice) {
					slice.Spec.SharingAffinity[0].APIGroup = ""
				},
				field: "spec.sharingAffinity[0].apiGroup",
				cause: metav1.CauseTypeFieldValueRequired,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Run("create", func(t *testing.T) {
					slice := sharingAffinitySlice()
					slice.GenerateName = ""
					slice.Name = tc.name + "-" + gv.Version
					tc.mutate(slice)
					_, err := client.Create(ctx, sharingAffinityObject(t, gv, slice), metav1.CreateOptions{})
					assertSharingAffinityInvalid(t, err, tc.field, tc.cause)
					if _, err := client.Get(ctx, slice.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
						t.Fatalf("rejected create must not persist an object, got: %v", err)
					}
				})
				t.Run("update", func(t *testing.T) {
					slice := sharingAffinitySlice()
					created := createSharingAffinitySlice(t, ctx, client, sharingAffinityObject(t, gv, slice))
					slice.Name = created.GetName()
					slice.ResourceVersion = created.GetResourceVersion()
					tc.mutate(slice)
					_, err := client.Update(ctx, sharingAffinityObject(t, gv, slice), metav1.UpdateOptions{})
					assertSharingAffinityInvalid(t, err, tc.field, tc.cause)
					stored := getSharingAffinitySlice(t, ctx, client, created.GetName())
					assertSharingAffinity(t, stored, sharingAffinitySlice().Spec.SharingAffinity)
					if stored.GetResourceVersion() != created.GetResourceVersion() {
						t.Fatal("rejected update changed the stored resourceVersion")
					}
				})
			})
		}
	})
}

func startSharingAffinityServer(t *testing.T, enabled bool, etcdOptions *storagebackend.Config) *kubeapiservertesting.TestServer {
	t.Helper()
	// These gates are process-global; callers and version subtests must stay serial.
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRASharingAffinity:    enabled,
	})
	server := kubeapiservertesting.StartTestServerOrDie(t, kubeapiservertesting.NewDefaultTestServerOptions(), framework.DefaultTestServerFlags(), etcdOptions)
	t.Cleanup(server.TearDownFn)
	return server
}

func forSharingAffinityVersions(t *testing.T, server *kubeapiservertesting.TestServer, test func(*testing.T, context.Context, schema.GroupVersion, dynamic.ResourceInterface)) {
	t.Helper()
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("create discovery client: %v", err)
	}
	client, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("create ResourceSlice client: %v", err)
	}
	// Do not enable retired or non-default APIs just to exercise conversions.
	for _, gv := range []schema.GroupVersion{resourceapi.SchemeGroupVersion, resourcev1beta2.SchemeGroupVersion} {
		t.Run(gv.Version, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			resources, err := discoveryClient.ServerResourcesForGroupVersion(gv.String())
			if apierrors.IsNotFound(err) && gv == resourcev1beta2.SchemeGroupVersion {
				t.Skipf("%s is not served by default", gv)
			}
			if err != nil {
				t.Fatalf("discover %s: %v", gv, err)
			}
			served := false
			for _, resource := range resources.APIResources {
				if resource.Name == "resourceslices" {
					served = true
					break
				}
			}
			if !served {
				if gv == resourcev1beta2.SchemeGroupVersion {
					t.Skipf("ResourceSlices are not served by default in %s", gv)
				}
				t.Fatalf("ResourceSlices must be served in %s", gv)
			}
			logger.Info("Testing sharing affinity admission and storage", "apiVersion", gv.String())
			test(t, ctx, gv, client.Resource(gv.WithResource("resourceslices")))
		})
	}
}

func sharingAffinitySlice() *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "sharing-affinity-"},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   "sharing.example.com",
			Pool:     resourceapi.ResourcePool{Name: "pool", Generation: 1, ResourceSliceCount: 1},
			AllNodes: ptr.To(true),
			SharingAffinity: []resourceapi.SharingAffinityExtractor{{
				Name:       "subnet",
				APIGroup:   "networking.example.com",
				Kind:       "NICConfig",
				Expression: `parameters.?subnetID.orValue("")`,
			}},
		},
	}
}

func sharingAffinityObject(t *testing.T, gv schema.GroupVersion, slice *resourceapi.ResourceSlice) *unstructured.Unstructured {
	t.Helper()
	// A typed fixture keeps the proposed API a compile-time contract. Sending
	// its JSON shape to each served endpoint exercises server-side conversion.
	object, err := runtime.DefaultUnstructuredConverter.ToUnstructured(slice)
	if err != nil {
		t.Fatalf("encode ResourceSlice: %v", err)
	}
	result := &unstructured.Unstructured{Object: object}
	result.SetAPIVersion(gv.String())
	result.SetKind("ResourceSlice")
	return result
}

func createSharingAffinitySlice(t *testing.T, ctx context.Context, client dynamic.ResourceInterface, slice *unstructured.Unstructured) *unstructured.Unstructured {
	t.Helper()
	created, err := client.Create(ctx, slice, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create ResourceSlice: %v", err)
	}
	return created
}

func getSharingAffinitySlice(t *testing.T, ctx context.Context, client dynamic.ResourceInterface, name string) *unstructured.Unstructured {
	t.Helper()
	stored, err := client.Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get ResourceSlice %q: %v", name, err)
	}
	return stored
}

func assertSharingAffinity(t *testing.T, object *unstructured.Unstructured, want []resourceapi.SharingAffinityExtractor) {
	t.Helper()
	var slice resourceapi.ResourceSlice
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, &slice); err != nil {
		t.Fatalf("decode ResourceSlice %q: %v", object.GetName(), err)
	}
	if !reflect.DeepEqual(slice.Spec.SharingAffinity, want) {
		t.Fatalf("ResourceSlice %q sharingAffinity = %#v, want %#v", object.GetName(), slice.Spec.SharingAffinity, want)
	}
}

func assertSharingAffinityInvalid(t *testing.T, err error, field string, cause metav1.CauseType) {
	t.Helper()
	var statusError *apierrors.StatusError
	if !apierrors.IsInvalid(err) || !errors.As(err, &statusError) {
		t.Fatalf("expected Invalid API status for %s, got: %v", field, err)
	}
	status := statusError.ErrStatus
	if status.Code != http.StatusUnprocessableEntity || status.Reason != metav1.StatusReasonInvalid {
		t.Fatalf("expected HTTP 422 Invalid, got: %#v", status)
	}
	if status.Details == nil || len(status.Details.Causes) == 0 {
		t.Fatalf("expected field causes for %s, got: %#v", field, status)
	}
	for _, got := range status.Details.Causes {
		if got.Field != field || got.Type != cause {
			t.Errorf("expected flat field cause %s (%s), got: %#v", field, cause, got)
		}
	}
}
