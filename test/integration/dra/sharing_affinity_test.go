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
	"fmt"
	"net/http"
	"reflect"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/ktesting"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	draapi "k8s.io/dynamic-resource-allocation/api"
	dracel "k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	klogtesting "k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
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
			logger, ctx := klogtesting.NewTestContext(t)
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

const publicationTimeout = 30 * time.Second

type publisher struct {
	controller *resourceslice.Controller
	client     kubernetes.Interface
	driver     string
	// Sized so the error handler never blocks a worker.
	errs chan error
}

func startPublisher(t *testing.T, server *kubeapiservertesting.TestServer, driver string, resources *resourceslice.DriverResources) *publisher {
	t.Helper()
	_, ctx := klogtesting.NewTestContext(t)
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("create client: %v", err)
	}
	p := &publisher{client: client, driver: driver, errs: make(chan error, 1000)}
	ctx, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)
	p.controller, err = resourceslice.StartController(ctx, resourceslice.Options{
		DriverName: driver,
		KubeClient: client,
		Resources:  resources,
		ErrorHandler: func(_ context.Context, err error, _ string) {
			select {
			case p.errs <- err:
			default:
			}
		},
	})
	if err != nil {
		t.Fatalf("start ResourceSlice controller: %v", err)
	}
	t.Cleanup(p.controller.Stop)
	return p
}

// list returns the driver's slices sorted by name, which encodes the slice index.
func (p *publisher) list(t *testing.T) []resourceapi.ResourceSlice {
	t.Helper()
	list, err := p.client.ResourceV1().ResourceSlices().List(context.Background(), metav1.ListOptions{
		FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + p.driver,
	})
	if err != nil {
		t.Fatalf("list ResourceSlices: %v", err)
	}
	items := list.Items
	sort.Slice(items, func(i, j int) bool { return items[i].Name < items[j].Name })
	return items
}

func (p *publisher) waitFor(t *testing.T, what string, cond func([]resourceapi.ResourceSlice) bool) []resourceapi.ResourceSlice {
	t.Helper()
	var items []resourceapi.ResourceSlice
	err := wait.PollUntilContextTimeout(context.Background(), 100*time.Millisecond, publicationTimeout, true, func(context.Context) (bool, error) {
		items = p.list(t)
		return cond(items), nil
	})
	if err != nil {
		t.Fatalf("waiting for %s: %v; last state: %s", what, err, describeSlices(items))
	}
	return items
}

func (p *publisher) waitForError(t *testing.T, what string, match func(error) bool) error {
	t.Helper()
	deadline := time.After(publicationTimeout)
	for {
		select {
		case err := <-p.errs:
			if match(err) {
				return err
			}
			t.Logf("ignoring unrelated controller error: %v", err)
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

// stopAndDrain is the completion barrier for negative assertions: after
// Stop every sync and error handler call has returned.
func (p *publisher) stopAndDrain() []error {
	p.controller.Stop()
	var errs []error
	for {
		select {
		case err := <-p.errs:
			errs = append(errs, err)
		default:
			return errs
		}
	}
}

func describeSlices(items []resourceapi.ResourceSlice) string {
	var parts []string
	for _, s := range items {
		parts = append(parts, fmt.Sprintf("%s{gen=%d count=%d devices=%d affinity=%v}", s.Name, s.Spec.Pool.Generation, s.Spec.Pool.ResourceSliceCount, len(s.Spec.Devices), s.Spec.SharingAffinity))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func affinity(name, expression string) []resourceapi.SharingAffinityExtractor {
	return []resourceapi.SharingAffinityExtractor{{
		Name:       name,
		APIGroup:   "networking.example.com",
		Kind:       "NICConfig",
		Expression: expression,
	}}
}

func publicationResources(generation int64, affinities ...[]resourceapi.SharingAffinityExtractor) *resourceslice.DriverResources {
	pool := resourceslice.Pool{AllNodes: true, Generation: generation}
	for _, a := range affinities {
		pool.Slices = append(pool.Slices, resourceslice.Slice{SharingAffinity: a})
	}
	pool.Slices = append(pool.Slices, resourceslice.Slice{Devices: []resourceapi.Device{{Name: "dev-0"}}})
	return &resourceslice.DriverResources{Pools: map[string]resourceslice.Pool{"pool": pool}}
}

// matches reports whether the stored pool equals the desired affinities in
// index order, followed by the single device slice, at one generation.
func matches(items []resourceapi.ResourceSlice, affinities ...[]resourceapi.SharingAffinityExtractor) bool {
	if len(items) != len(affinities)+1 {
		return false
	}
	for i, s := range items {
		if s.Spec.Pool.Generation != items[0].Spec.Pool.Generation || s.Spec.Pool.ResourceSliceCount != int64(len(items)) {
			return false
		}
		if i < len(affinities) {
			if !reflect.DeepEqual(s.Spec.SharingAffinity, affinities[i]) || len(s.Spec.Devices) != 0 {
				return false
			}
		} else if len(s.Spec.Devices) != 1 || s.Spec.SharingAffinity != nil {
			return false
		}
	}
	return true
}

func isDropped(err error) bool {
	var dropped *resourceslice.DroppedFieldsError
	return errors.As(err, &dropped) && slices.Contains(dropped.DisabledFeatures(), "DRASharingAffinity")
}

func TestSharingAffinityPublication(t *testing.T) {
	t.Run("gate-enabled-multi-slice-create-update", func(t *testing.T) {
		server := startSharingAffinityServer(t, true, framework.SharedEtcd())
		a, b := affinity("subnet", `parameters.?subnetID.orValue("")`), affinity("vlan", `parameters.?vlan.orValue("")`)
		p := startPublisher(t, server, "multi.sharing.example.com", publicationResources(0, a, b))
		created := p.waitFor(t, "initial publication", func(i []resourceapi.ResourceSlice) bool { return matches(i, a, b) })
		uids := []string{string(created[0].UID), string(created[1].UID)}

		a2 := affinity("subnet", `parameters.?subnet.orValue("x")`)
		p.controller.Update(publicationResources(0, a2, b))
		updated := p.waitFor(t, "one changed declaration", func(i []resourceapi.ResourceSlice) bool { return matches(i, a2, b) })
		if string(updated[0].UID) != uids[0] || string(updated[1].UID) != uids[1] {
			t.Errorf("metadata slices were replaced instead of updated: %s", describeSlices(updated))
		}
		if updated[0].Spec.Pool.Generation < created[0].Spec.Pool.Generation {
			t.Errorf("generation went backwards: %d -> %d", created[0].Spec.Pool.Generation, updated[0].Spec.Pool.Generation)
		}

		a3, b3 := affinity("subnet", `parameters.?s.orValue("y")`), affinity("vlan", `parameters.?v.orValue("y")`)
		p.controller.Update(publicationResources(updated[0].Spec.Pool.Generation+1, a3, b3))
		final := p.waitFor(t, "two changed declarations", func(i []resourceapi.ResourceSlice) bool {
			return matches(i, a3, b3) && i[0].Spec.Pool.Generation > updated[0].Spec.Pool.Generation
		})
		if errs := p.stopAndDrain(); len(errs) > 0 {
			t.Errorf("unexpected controller errors: %v (state %s)", errs, describeSlices(final))
		}
	})

	t.Run("gate-enabled-invalid-pool-not-published", func(t *testing.T) {
		server := startSharingAffinityServer(t, true, framework.SharedEtcd())
		dup := affinity("subnet", `parameters.?subnetID.orValue("")`)
		p := startPublisher(t, server, "dup.sharing.example.com", publicationResources(0, dup, dup))
		err := p.waitForError(t, "duplicate key validation error", func(err error) bool {
			return strings.Contains(err.Error(), `duplicate sharing affinity key "subnet"`)
		})
		t.Logf("got expected error: %v", err)
		p.stopAndDrain()
		if items := p.list(t); len(items) != 0 {
			t.Fatalf("invalid pool was published: %s", describeSlices(items))
		}
	})

	t.Run("gate-disabled-create-reports-dropped", func(t *testing.T) {
		server := startSharingAffinityServer(t, false, framework.SharedEtcd())
		a := affinity("subnet", `parameters.?subnetID.orValue("")`)
		p := startPublisher(t, server, "create-off.sharing.example.com", publicationResources(0, a))
		p.waitForError(t, "DroppedFieldsError", isDropped)
		items := p.waitFor(t, "slices without affinity", func(i []resourceapi.ResourceSlice) bool { return matches(i, nil) })
		p.stopAndDrain()
		if items[0].Spec.SharingAffinity != nil {
			t.Fatalf("affinity stored with gate disabled: %s", describeSlices(items))
		}
	})

	t.Run("gate-disabled-update-preserves-existing", func(t *testing.T) {
		etcd := framework.SharedEtcd()
		driver := "preserve.sharing.example.com"
		a := affinity("subnet", `parameters.?subnetID.orValue("")`)
		var before resourceapi.ResourceSlice
		if !t.Run("enabled", func(t *testing.T) {
			p := startPublisher(t, startSharingAffinityServer(t, true, etcd), driver, publicationResources(0, a))
			before = p.waitFor(t, "enabled publication", func(i []resourceapi.ResourceSlice) bool { return matches(i, a) })[0]
			if errs := p.stopAndDrain(); len(errs) > 0 {
				t.Fatalf("unexpected controller errors: %v", errs)
			}
		}) {
			return
		}
		t.Run("disabled", func(t *testing.T) {
			p := startPublisher(t, startSharingAffinityServer(t, false, etcd), driver, publicationResources(before.Spec.Pool.Generation+1, a))
			after := p.waitFor(t, "generation bump", func(i []resourceapi.ResourceSlice) bool {
				return len(i) == 2 && i[0].Spec.Pool.Generation > before.Spec.Pool.Generation
			})[0]
			errs := p.stopAndDrain()
			if after.UID != before.UID || after.ResourceVersion == before.ResourceVersion {
				t.Errorf("expected in-place update of %s: uid %s->%s rv %s->%s", before.Name, before.UID, after.UID, before.ResourceVersion, after.ResourceVersion)
			}
			if !reflect.DeepEqual(after.Spec.SharingAffinity, a) {
				t.Errorf("stored affinity not preserved: %v", after.Spec.SharingAffinity)
			}
			for _, err := range errs {
				var dropped *resourceslice.DroppedFieldsError
				if errors.As(err, &dropped) {
					t.Errorf("unexpected DroppedFieldsError for preserved affinity: %v", err)
				}
			}
		})
	})

	t.Run("gate-disabled-update-introducing-affinity-dropped", func(t *testing.T) {
		etcd := framework.SharedEtcd()
		driver := "introduce.sharing.example.com"
		var before resourceapi.ResourceSlice
		if !t.Run("enabled", func(t *testing.T) {
			p := startPublisher(t, startSharingAffinityServer(t, true, etcd), driver, publicationResources(0, nil))
			before = p.waitFor(t, "empty metadata slice", func(i []resourceapi.ResourceSlice) bool { return matches(i, nil) })[0]
			p.stopAndDrain()
		}) {
			return
		}
		t.Run("disabled", func(t *testing.T) {
			a := affinity("subnet", `parameters.?subnetID.orValue("")`)
			p := startPublisher(t, startSharingAffinityServer(t, false, etcd), driver, publicationResources(before.Spec.Pool.Generation+1, a))
			p.waitForError(t, "DroppedFieldsError", isDropped)
			after := p.waitFor(t, "generation bump", func(i []resourceapi.ResourceSlice) bool {
				return len(i) == 2 && i[0].Spec.Pool.Generation > before.Spec.Pool.Generation
			})[0]
			p.stopAndDrain()
			if after.UID != before.UID || after.ResourceVersion == before.ResourceVersion {
				t.Errorf("expected in-place update: uid %s->%s rv %s->%s", before.UID, after.UID, before.ResourceVersion, after.ResourceVersion)
			}
			if after.Spec.SharingAffinity != nil {
				t.Errorf("affinity stored with gate disabled: %v", after.Spec.SharingAffinity)
			}
		})
	})

	t.Run("remove-last-declaration", func(t *testing.T) {
		server := startSharingAffinityServer(t, true, framework.SharedEtcd())
		a := affinity("subnet", `parameters.?subnetID.orValue("")`)
		p := startPublisher(t, server, "remove.sharing.example.com", publicationResources(0, a))
		before := p.waitFor(t, "publication", func(i []resourceapi.ResourceSlice) bool { return matches(i, a) })[0]
		p.controller.Update(publicationResources(0, nil))
		after := p.waitFor(t, "affinity removed", func(i []resourceapi.ResourceSlice) bool { return matches(i, nil) })[0]
		if after.UID != before.UID {
			t.Errorf("metadata slice replaced instead of updated")
		}
		p.controller.Update(publicationResources(0))
		p.waitFor(t, "metadata slice deleted", func(i []resourceapi.ResourceSlice) bool { return matches(i) })
		if errs := p.stopAndDrain(); len(errs) > 0 {
			t.Errorf("unexpected controller errors: %v", errs)
		}
	})
}

// These tests flip process-global feature gates and each owns its
// apiserver and scheduler, so none of them may run in parallel.

const (
	saGateName     = "SharingAffinityGate"
	saGateLabel    = "sharing-affinity-gate"
	saGateWait     = "wait"
	saGateReject   = "reject"
	saGatePreBind  = "prebind-fail-once"
	saNode         = "worker-0"
	saMemory       = resourceapi.QualifiedName("memory")
	saSubnetExpr   = `parameters.?subnetID.orValue("")`
	saConsistently = 5 * time.Second
)

var saSchedulerConfig = `
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
podInitialBackoffSeconds: 1
podMaxBackoffSeconds: 1
profiles:
- schedulerName: default-scheduler
  plugins:
    permit:
      enabled:
      - name: ` + saGateName + `
    preBind:
      enabled:
      - name: ` + saGateName + `
`

// sharingAffinityGate gives tests control over the window after Reserve:
// Permit runs before any PreBind, so Wait holds a tentative lock and Reject
// drives Unreserve; the PreBind hook runs after DynamicResources persisted.
type sharingAffinityGate struct {
	handle fwk.Handle

	mutex      sync.Mutex
	waiting    map[string]bool
	rejected   map[string]int
	preBindHit map[string]int
}

var (
	_ fwk.PermitPlugin  = &sharingAffinityGate{}
	_ fwk.PreBindPlugin = &sharingAffinityGate{}
)

func (g *sharingAffinityGate) Name() string { return saGateName }

func (g *sharingAffinityGate) Permit(_ context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) (*fwk.Status, time.Duration) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	switch pod.Labels[saGateLabel] {
	case saGateWait:
		g.waiting[pod.Name] = true
		return fwk.NewStatus(fwk.Wait), 2 * time.Minute
	case saGateReject:
		g.rejected[pod.Name]++
		return fwk.NewStatus(fwk.Unschedulable, "rejected by sharing affinity test gate"), 0
	}
	return nil, 0
}

func (g *sharingAffinityGate) PreBindPreFlight(_ context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) (*fwk.PreBindPreFlightResult, *fwk.Status) {
	if pod.Labels[saGateLabel] != saGatePreBind {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	return nil, nil
}

func (g *sharingAffinityGate) PreBind(_ context.Context, _ fwk.CycleState, pod *v1.Pod, _ string) *fwk.Status {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	g.preBindHit[pod.Name]++
	if g.preBindHit[pod.Name] == 1 {
		return fwk.NewStatus(fwk.Error, "injected PreBind failure after DynamicResources")
	}
	return nil
}

func (g *sharingAffinityGate) count(m map[string]int, name string) int {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return m[name]
}

func (g *sharingAffinityGate) isWaiting(name string) bool {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g.waiting[name]
}

type saEnv struct {
	tCtx ktesting.TContext
	gate *sharingAffinityGate
}

// newSAEnv starts an apiserver with sharing affinity stored, plus nodes and
// the claim controller (needed to release claim finalizers on deletion).
func newSAEnv(t *testing.T) *saEnv {
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRASharingAffinity:    true,
	})
	server, err := kubeapiservertesting.StartTestServer(tCtx.WithoutCancel(), kubeapiservertesting.NewDefaultTestServerOptions(), framework.DefaultTestServerFlags(), framework.SharedEtcd())
	tCtx.ExpectNoError(err, "start apiserver")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) { server.TearDownFn() })
	tCtx = tCtx.WithRESTConfig(server.ClientConfig)
	createNodes(tCtx)
	tCtx = prepareClaimController(tCtx)
	startClaimController(tCtx)
	return &saEnv{tCtx: tCtx}
}

// startScheduler reads the current scheduler-side gates, so callers flip
// DRASharingAffinity before calling it.
func (e *saEnv) startScheduler() func() {
	gate := &sharingAffinityGate{waiting: map[string]bool{}, rejected: map[string]int{}, preBindHit: map[string]int{}}
	e.gate = gate
	registry := frameworkruntime.Registry{
		saGateName: func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			gate.handle = h
			return gate, nil
		},
	}
	schedCtx := e.tCtx.WithCancel()
	cfg := newSchedulerComponentConfig(schedCtx, saSchedulerConfig)
	_, informerFactory, done := util.StartSchedulerWithDone(schedCtx, cfg, registry)
	var once sync.Once
	stop := func() {
		once.Do(func() {
			schedCtx.Cancel("stop sharing affinity scheduler")
			select {
			case <-done:
			case <-time.After(30 * time.Second):
				e.tCtx.Fatal("sharing affinity scheduler did not stop within 30s")
			}
			informerFactory.Shutdown()
		})
	}
	e.tCtx.CleanupCtx(func(ktesting.TContext) { stop() })
	return stop
}

// scope is a per-subtest namespace with its own driver and class.
type saScope struct {
	tCtx   ktesting.TContext
	ns     string
	driver string
	class  *resourceapi.DeviceClass
}

func (e *saEnv) run(name string, cb func(s *saScope)) {
	e.tCtx.Run(name, func(tCtx ktesting.TContext) {
		ns := createTestNamespace(tCtx, nil)
		class, driver := createTestClass(tCtx, ns)
		cb(&saScope{tCtx: tCtx, ns: ns, driver: driver, class: class})
	})
}

type saDevice struct {
	name       string
	memory     string
	multiple   bool
	attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute
}

func shared(name, memory string) saDevice {
	return saDevice{name: name, memory: memory, multiple: true}
}

// saSlices is every slice a helper call created for one pool.
type saSlices []*resourceapi.ResourceSlice

// slice publishes a pool. Validation keeps sharingAffinity in a dedicated
// metadata slice, so extractors plus devices become two slices and count is
// raised by one; count is otherwise the caller's ResourceSliceCount.
func (s *saScope) slice(name, pool string, generation, count int64, extractors []resourceapi.SharingAffinityExtractor, devices ...saDevice) saSlices {
	if len(extractors) > 0 && len(devices) > 0 {
		count++
		return append(s.slice(name+"-dev", pool, generation, count, nil, devices...),
			s.slice(name+"-meta", pool, generation, count, extractors)...)
	}
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: s.ns + "-" + name},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:          s.driver,
			Pool:            resourceapi.ResourcePool{Name: pool, Generation: generation, ResourceSliceCount: count},
			NodeName:        ptr.To(saNode),
			SharingAffinity: extractors,
		},
	}
	for _, d := range devices {
		device := resourceapi.Device{Name: d.name, Attributes: d.attributes}
		if d.multiple {
			device.AllowMultipleAllocations = ptr.To(true)
		}
		if d.memory != "" {
			device.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{saMemory: {Value: resource.MustParse(d.memory)}}
		}
		slice.Spec.Devices = append(slice.Spec.Devices, device)
	}
	return saSlices{createSlice(s.tCtx, slice)}
}

func subnetExtractor(expr string) []resourceapi.SharingAffinityExtractor {
	return []resourceapi.SharingAffinityExtractor{{Name: "subnet", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: expr}}
}

func nicParams(fields string) string {
	if fields != "" {
		fields = "," + fields
	}
	return `{"apiVersion":"networking.example.com/v1","kind":"NICConfig"` + fields + `}`
}

func subnet(id string) string { return nicParams(fmt.Sprintf(`"subnetID":%q`, id)) }

type saRequest struct {
	name        string
	memory      string
	subRequests []string
	selectors   []resourceapi.DeviceSelector
}

type saConfig struct {
	requests []string
	params   string
}

func (s *saScope) claim(name string, requests []saRequest, configs ...saConfig) *resourceapi.ResourceClaim {
	claim := &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: s.ns}}
	capacity := func(memory string) *resourceapi.CapacityRequirements {
		if memory == "" {
			return nil
		}
		return &resourceapi.CapacityRequirements{Requests: map[resourceapi.QualifiedName]resource.Quantity{saMemory: resource.MustParse(memory)}}
	}
	for _, r := range requests {
		request := resourceapi.DeviceRequest{Name: r.name}
		if len(r.subRequests) == 0 {
			request.Exactly = &resourceapi.ExactDeviceRequest{
				DeviceClassName: s.class.Name, AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
				Capacity:  capacity(r.memory),
				Selectors: r.selectors,
			}
		}
		for _, sub := range r.subRequests {
			request.FirstAvailable = append(request.FirstAvailable, resourceapi.DeviceSubRequest{
				Name: sub, DeviceClassName: s.class.Name, AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
				Capacity:  capacity(r.memory),
				Selectors: r.selectors,
			})
		}
		claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, request)
	}
	for _, c := range configs {
		claim.Spec.Devices.Config = append(claim.Spec.Devices.Config, resourceapi.DeviceClaimConfiguration{
			Requests: c.requests,
			DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: s.driver, Parameters: runtime.RawExtension{Raw: []byte(c.params)},
			}},
		})
	}
	claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Create(s.tCtx, claim, metav1.CreateOptions{})
	s.tCtx.ExpectNoError(err, "create claim "+name)
	return claim
}

// nicClaim is a single shared-NIC request carrying one config.
func (s *saScope) nicClaim(name, memory, params string) *resourceapi.ResourceClaim {
	var configs []saConfig
	if params != "" {
		configs = append(configs, saConfig{requests: []string{"nic"}, params: params})
	}
	return s.claim(name, []saRequest{{name: "nic", memory: memory}}, configs...)
}

func (s *saScope) pod(name, gate string, claims ...*resourceapi.ResourceClaim) *v1.Pod {
	w := st.MakePod().Name(name).Container("c")
	if gate != "" {
		w = w.Label(saGateLabel, gate)
	}
	return createPod(s.tCtx, s.ns, "", w.Obj(), claims...)
}

func (s *saScope) scheduled(pod *v1.Pod) *v1.Pod {
	s.tCtx.Helper()
	pod = waitForPodScheduled(s.tCtx, s.ns, pod.Name)
	if pod.Spec.NodeName != saNode {
		s.tCtx.Fatalf("pod %s bound to %q, want %q", pod.Name, pod.Spec.NodeName, saNode)
	}
	return pod
}

// notScheduled waits for a PodScheduled=False condition with the given
// reason and message fragments; absent fragments are asserted too.
func (s *saScope) notScheduled(pod *v1.Pod, reason string, contains []string, absent ...string) *v1.Pod {
	s.tCtx.Helper()
	var got *v1.Pod
	s.tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		p, err := tCtx.Client().CoreV1().Pods(s.ns).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		got = p
		for _, c := range p.Status.Conditions {
			if c.Type == v1.PodScheduled && c.Status == v1.ConditionFalse && c.Reason == reason {
				return c.Message, nil
			}
		}
		return "", nil
	}).WithTimeout(schedulingTimeout).Should(gomega.And(append([]gomega.OmegaMatcher{gomega.Not(gomega.BeEmpty())},
		saContainAll(contains)...)...), "pod %s should be %s", pod.Name, reason)
	for _, a := range absent {
		for _, c := range got.Status.Conditions {
			if strings.Contains(c.Message, a) {
				s.tCtx.Fatalf("pod %s condition leaks %q: %s", pod.Name, a, c.Message)
			}
		}
	}
	if got.Spec.NodeName != "" {
		s.tCtx.Fatalf("pod %s unexpectedly bound to %s", pod.Name, got.Spec.NodeName)
	}
	return got
}

func saContainAll(fragments []string) []gomega.OmegaMatcher {
	var m []gomega.OmegaMatcher
	for _, f := range fragments {
		m = append(m, gomega.ContainSubstring(f))
	}
	return m
}

// staysPending asserts a bounded window in which the pod is not bound and
// its claim stays unallocated.
func (s *saScope) staysPending(pod *v1.Pod, claims ...*resourceapi.ResourceClaim) {
	s.tCtx.Helper()
	s.tCtx.Consistently(func(tCtx ktesting.TContext) (string, error) {
		p, err := tCtx.Client().CoreV1().Pods(s.ns).Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return "", err
		}
		for _, c := range claims {
			c, err := tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(tCtx, c.Name, metav1.GetOptions{})
			if err != nil {
				return "", err
			}
			if c.Status.Allocation != nil {
				return "claim " + c.Name + " allocated", nil
			}
		}
		return p.Spec.NodeName, nil
	}).WithTimeout(saConsistently).WithPolling(250*time.Millisecond).Should(gomega.BeEmpty(), "pod %s must stay pending", pod.Name)
}

func (s *saScope) allocated(claim *resourceapi.ResourceClaim) []resourceapi.DeviceRequestAllocationResult {
	s.tCtx.Helper()
	c := waitForClaimAllocatedToDevice(s.tCtx, s.ns, claim.Name, schedulingTimeout)
	return c.Status.Allocation.Devices.Results
}

func (s *saScope) unallocated(claim *resourceapi.ResourceClaim) {
	s.tCtx.Helper()
	c, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(s.tCtx, claim.Name, metav1.GetOptions{})
	s.tCtx.ExpectNoError(err, "get claim")
	if c.Status.Allocation != nil {
		s.tCtx.Fatalf("claim %s unexpectedly allocated: %+v", claim.Name, c.Status.Allocation.Devices.Results)
	}
}

func (s *saScope) device(claim *resourceapi.ResourceClaim) string {
	s.tCtx.Helper()
	results := s.allocated(claim)
	if len(results) != 1 {
		s.tCtx.Fatalf("claim %s: want one result, got %+v", claim.Name, results)
	}
	return results[0].Pool + "/" + results[0].Device
}

func (s *saScope) release(pod *v1.Pod, claims ...*resourceapi.ResourceClaim) {
	s.tCtx.Helper()
	deleteAndWait(s.tCtx, s.tCtx.Client().CoreV1().Pods(s.ns).Delete, s.tCtx.Client().CoreV1().Pods(s.ns).Get, pod.Name)
	for _, c := range claims {
		deleteAndWait(s.tCtx, s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Delete, s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get, c.Name)
	}
}

func (s *saScope) updateSlice(slices saSlices, mutate func(*resourceapi.ResourceSlice)) {
	s.tCtx.Helper()
	for _, slice := range slices {
		current, err := s.tCtx.Client().ResourceV1().ResourceSlices().Get(s.tCtx, slice.Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "get slice")
		mutate(current)
		_, err = s.tCtx.Client().ResourceV1().ResourceSlices().Update(s.tCtx, current, metav1.UpdateOptions{})
		s.tCtx.ExpectNoError(err, "update slice")
	}
}

func (s *saScope) deleteSlice(slices saSlices) {
	s.tCtx.Helper()
	for _, slice := range slices {
		deleteAndWait(s.tCtx, s.tCtx.Client().ResourceV1().ResourceSlices().Delete, s.tCtx.Client().ResourceV1().ResourceSlices().Get, slice.Name)
	}
}

func TestSharingAffinitySchedulingCompatible(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("same-device", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		s.scheduled(s.pod("pb", "", b))
		ra, rb := s.allocated(a), s.allocated(b)
		if ra[0].Device != "nic" || rb[0].Device != "nic" {
			s.tCtx.Fatalf("compatible claims not co-allocated: %+v / %+v", ra, rb)
		}
		if ra[0].ShareID == nil || rb[0].ShareID == nil || *ra[0].ShareID == *rb[0].ShareID {
			s.tCtx.Fatalf("co-allocated claims need distinct ShareIDs: %v / %v", ra[0].ShareID, rb[0].ShareID)
		}
	})
}

func TestSharingAffinitySchedulingMismatch(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("one-device", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
	})
}

func TestSharingAffinitySchedulingReleaseAndRollback(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("release", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.release(pa, a)
		s.scheduled(pb)
		if d := s.device(b); d != "pool/nic" {
			s.tCtx.Fatalf("released device not reused: %s", d)
		}
	})
	env.run("permit-reject-unreserve", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", saGateReject, a)
		s.tCtx.Eventually(func() int { return env.gate.count(env.gate.rejected, pa.Name) }).
			WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">=", 1), "Permit should reject pa")
		s.unallocated(a)
		// Deleting the pod cannot run Unreserve again; a leaked tentative
		// lock from the rejected attempt would keep blocking subnet-b.
		deleteAndWait(s.tCtx, s.tCtx.Client().CoreV1().Pods(s.ns).Delete, s.tCtx.Client().CoreV1().Pods(s.ns).Get, pa.Name)
		s.unallocated(a)
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.scheduled(s.pod("pb", "", b))
		if d := s.device(b); d != "pool/nic" {
			s.tCtx.Fatalf("want pool/nic after rollback, got %s", d)
		}
	})
	env.run("prebind-failure-after-persist", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", saGatePreBind, a)
		pa = s.scheduled(pa)
		if hits := env.gate.count(env.gate.preBindHit, pa.Name); hits < 2 {
			s.tCtx.Fatalf("PreBind failure path not exercised: %d calls", hits)
		}
		s.allocated(a)
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.release(pa, a)
		s.scheduled(pb)
	})
}

func TestSharingAffinitySchedulingCapacity(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("exhausted-lock", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "4Gi"), shared("nic-1", "4Gi"))
		a := s.nicClaim("a", "3Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "3Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pb", "", b))
		if da, db := s.device(a), s.device(b); da == db {
			s.tCtx.Fatalf("capacity exceeded: both 3Gi claims on %s", da)
		}
		// 1Gi remains on each device; observed-only accounting must admit it.
		c := s.nicClaim("c", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pc", "", c))
		d := s.nicClaim("d", "2Gi", subnet("subnet-a"))
		s.notScheduled(s.pod("pd", "", d), v1.PodReasonUnschedulable, nil)
		s.unallocated(d)
	})
}

func TestSharingAffinitySchedulingConcurrent(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("overlapping-attempts", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", saGateWait, a)
		var waiting fwk.WaitingPod
		s.tCtx.Eventually(func() bool {
			if !env.gate.isWaiting(pa.Name) {
				return false
			}
			waiting = env.gate.handle.GetWaitingPod(pa.UID)
			return waiting != nil
		}).WithTimeout(schedulingTimeout).Should(gomega.BeTrue(), "pa should hold its Reserve at Permit")
		// The lock is only in-flight: nothing has been written to the API.
		s.unallocated(a)
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
		waiting.Allow(saGateName)
		s.scheduled(pa)
		if d := s.device(a); d != "pool/nic" {
			s.tCtx.Fatalf("pa allocated %s", d)
		}
		s.staysPending(pb, b)
	})
	env.run("same-pod-one-device", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.notScheduled(s.pod("p", "", a, b), v1.PodReasonUnschedulable, nil)
		s.unallocated(a)
		s.unallocated(b)
	})
	env.run("same-pod-two-devices", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "8Gi"), shared("nic-1", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.scheduled(s.pod("p", "", a, b))
		if da, db := s.device(a), s.device(b); da == db {
			s.tCtx.Fatalf("one allocation co-located incompatible claims on %s", da)
		}
	})
}

func TestSharingAffinitySchedulingGateDisabled(t *testing.T) {
	env := newSAEnv(t)
	type pending struct {
		s     *saScope
		check func()
	}
	// Slices with affinity must be stored while the apiserver gate is on;
	// only the scheduler then runs with the gate off.
	var scopes []saScope
	var declaring, gen2 saSlices
	for _, name := range []string{"declaring", "unconstrained", "older-generation"} {
		tCtx := env.tCtx
		ns := createTestNamespace(tCtx.WithStep(name), nil)
		class, driver := createTestClass(tCtx, ns)
		scopes = append(scopes, saScope{tCtx: tCtx, ns: ns, driver: driver, class: class})
	}
	declaring = scopes[0].slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
	scopes[1].slice("s", "pool", 1, 1, nil, shared("nic", "8Gi"))
	scopes[2].slice("gen1", "pool", 1, 1, nil, shared("nic", "8Gi"))
	gen2 = scopes[2].slice("gen2", "pool", 2, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRASharingAffinity, false)
	stored, err := env.tCtx.Client().ResourceV1().ResourceSlices().Get(env.tCtx, declaring[1].Name, metav1.GetOptions{})
	env.tCtx.ExpectNoError(err, "get declaring slice")
	if len(stored.Spec.SharingAffinity) != 1 {
		t.Fatalf("apiserver must retain sharingAffinity, got %+v", stored.Spec.SharingAffinity)
	}
	env.startScheduler()

	env.tCtx.Run("declaring-pool-unallocatable", func(tCtx ktesting.TContext) {
		s := scopes[0]
		s.tCtx = tCtx
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.notScheduled(s.pod("pa", "", a), v1.PodReasonUnschedulable, nil)
		s.unallocated(a)
	})
	env.tCtx.Run("unconstrained-pool-legacy", func(tCtx ktesting.TContext) {
		s := scopes[1]
		s.tCtx = tCtx
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		s.scheduled(s.pod("pa", "", a))
		s.scheduled(s.pod("pb", "", b))
		if da, db := s.device(a), s.device(b); da != db {
			s.tCtx.Fatalf("unconstrained shared device not shared: %s vs %s", da, db)
		}
	})
	env.tCtx.Run("delete-exposes-older-generation", func(tCtx ktesting.TContext) {
		s := scopes[2]
		s.tCtx = tCtx
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.deleteSlice(gen2)
		s.scheduled(pa)
	})
}

func TestSharingAffinitySchedulingRequestScope(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("request", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		s.slice("plain", "plain", 1, 1, nil, saDevice{name: "dedicated"})
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi"}, {name: "b"}},
			saConfig{requests: []string{"a"}, params: subnet("subnet-a")})
		s.scheduled(s.pod("p", "", c))
		got := map[string]string{}
		for _, r := range s.allocated(c) {
			got[r.Request] = r.Pool + "/" + r.Device
		}
		if got["a"] != "constrained/nic" || got["b"] != "plain/dedicated" {
			s.tCtx.Fatalf("config must scope to request a only, got %v", got)
		}
	})
	env.run("request-without-alternative", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "8Gi"), shared("nic-1", "8Gi"))
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi"}, {name: "b", memory: "1Gi"}},
			saConfig{requests: []string{"a"}, params: subnet("subnet-a")})
		s.notScheduled(s.pod("p", "", c), v1.PodReasonUnschedulable, nil)
		s.unallocated(c)
	})
	env.run("subrequest", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi", subRequests: []string{"x", "y"}}},
			saConfig{requests: []string{"a/y"}, params: subnet("subnet-a")})
		s.scheduled(s.pod("p", "", c))
		results := s.allocated(c)
		if len(results) != 1 || results[0].Request != "a/y" || results[0].Device != "nic" {
			s.tCtx.Fatalf("unconfigured subrequest x must not use the constrained pool: %+v", results)
		}
	})
	env.run("other-driver-config-ignored", func(s *saScope) {
		s.slice("constrained", "constrained", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		c := s.claim("c", []saRequest{{name: "a", memory: "1Gi"}})
		c.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{{
			Requests: []string{"a"},
			DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: "other.example.com", Parameters: runtime.RawExtension{Raw: []byte(subnet("subnet-a"))},
			}},
		}}
		s.tCtx.ExpectNoError(s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Delete(s.tCtx, c.Name, metav1.DeleteOptions{}))
		waitForNotFound(s.tCtx, s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get, c.Name)
		c.ResourceVersion, c.UID = "", ""
		c, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Create(s.tCtx, c, metav1.CreateOptions{})
		s.tCtx.ExpectNoError(err, "create claim with foreign-driver config")
		s.notScheduled(s.pod("p", "", c), v1.PodReasonUnschedulable, nil)
		s.unallocated(c)
	})
}

func TestSharingAffinitySchedulingMetadataChanges(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("completing-slice-single-event", func(s *saScope) {
		s.slice("devices", "pool", 1, 2, nil, shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.slice("metadata", "pool", 1, 2, subnetExtractor(saSubnetExpr))
		// No further writes: one Add event must converge.
		s.scheduled(pa)
	})
	env.run("same-generation-expression-update", func(s *saScope) {
		slice := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.scheduled(s.pod("pa", "", a))
		s.updateSlice(slice, func(sl *resourceapi.ResourceSlice) {
			if len(sl.Spec.SharingAffinity) > 0 {
				sl.Spec.SharingAffinity[0].Expression = `parameters.?vlan.orValue("")`
			}
		})
		// The bound owner was extracted under the old predicate; its device
		// must not be treated as compatible (or clean) under the new one.
		b := s.nicClaim("b", "1Gi", nicParams(`"subnetID":"subnet-a","vlan":"subnet-a"`))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
		s.release(pa, a)
		s.scheduled(pb)
	})
	env.run("same-generation-equal-map-expression-update", func(s *saScope) {
		slices := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
		a := saScheduleOwner(s, "a", subnet("subnet-a"), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", "subnet", "subnet-a"))
		extractors := subnetExtractor(`has(parameters.subnet) ? parameters.subnet : parameters.subnetID`)
		saRepublishAndProbe(env, s, slices, 1, extractors, nicParams(`"subnet":"subnet-a"`),
			saLocked("pool", "nic", "1Gi", "subnet", "subnet-a"))
		saScheduleOwner(s, "b", nicParams(`"subnet":"subnet-a"`), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
		saRetained(s, a)
	})
	env.run("generation-roll", func(s *saScope) {
		slice := s.slice("s", "pool", 1, 1, subnetExtractor(`parameters.?zone.orValue("")`), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.updateSlice(slice, func(sl *resourceapi.ResourceSlice) {
			sl.Spec.Pool.Generation = 2
			if len(sl.Spec.SharingAffinity) > 0 {
				sl.Spec.SharingAffinity[0].Expression = saSubnetExpr
			}
		})
		s.scheduled(pa)
	})
	env.run("stale-generation-deleted", func(s *saScope) {
		s.slice("gen1", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		gen2 := s.slice("gen2", "pool", 2, 2, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.deleteSlice(gen2)
		s.scheduled(pa)
	})
	env.run("metadata-added-to-unconstrained-pool", func(s *saScope) {
		slice := s.slice("s", "pool", 1, 1, nil, shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", "")
		pa := s.scheduled(s.pod("pa", "", a))
		s.updateSlice(slice, func(sl *resourceapi.ResourceSlice) {
			sl.Spec.Pool.Generation = 2
			sl.Spec.Pool.ResourceSliceCount = 2
		})
		s.slice("meta", "pool", 2, 2, subnetExtractor(saSubnetExpr))
		b := s.nicClaim("b", "1Gi", subnet("subnet-a"))
		pb := s.pod("pb", "", b)
		s.notScheduled(pb, v1.PodReasonUnschedulable, nil)
		s.unallocated(b)
		s.release(pa, a)
		s.scheduled(pb)
	})
}

func TestSharingAffinitySchedulingExtractionFailures(t *testing.T) {
	env := newSAEnv(t)
	env.startScheduler()
	env.run("claim-self-inconsistent", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.claim("a", []saRequest{{name: "nic", memory: "1Gi"}},
			saConfig{requests: []string{"nic"}, params: subnet("subnet-a")},
			saConfig{requests: []string{"nic"}, params: subnet("subnet-b")})
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, []string{s.ns + "/a", "self-inconsistent"}, "subnet-b")
		s.staysPending(pa, a)
		s.release(pa, a)
		fixed := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa2", "", fixed))
	})
	env.run("invalid-pool-duplicate-extractor", func(s *saScope) {
		s.slice("devices", "pool", 1, 3, nil, shared("nic", "8Gi"))
		s.slice("meta-1", "pool", 1, 3, subnetExtractor(saSubnetExpr))
		s.slice("meta-2", "pool", 1, 3, subnetExtractor(`parameters.?other.orValue("")`))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.notScheduled(s.pod("pa", "", a), v1.PodReasonUnschedulable, []string{"invalid"})
		s.unallocated(a)
	})
}

// Same as delete-exposes-older-generation, but the scheduler also runs with
// DRAConsumableCapacity disabled.
func TestSharingAffinitySchedulingGateDisabledNoConsumableCapacity(t *testing.T) {
	env := newSAEnv(t)
	tCtx := env.tCtx
	ns := createTestNamespace(tCtx.WithStep("older-generation"), nil)
	class, driver := createTestClass(tCtx, ns)
	s := saScope{tCtx: tCtx, ns: ns, driver: driver, class: class}
	s.slice("gen1", "pool", 1, 1, nil, saDevice{name: "nic"})
	gen2 := s.slice("gen2", "pool", 2, 1, subnetExtractor(saSubnetExpr), saDevice{name: "nic"})

	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRASharingAffinity:    false,
		features.DRAConsumableCapacity: false,
	})
	env.startScheduler()

	env.tCtx.Run("delete-exposes-older-generation", func(tCtx ktesting.TContext) {
		s.tCtx = tCtx
		a := s.claim("a", []saRequest{{name: "a"}})
		pa := s.pod("pa", "", a)
		s.notScheduled(pa, v1.PodReasonUnschedulable, nil)
		s.deleteSlice(gen2)
		s.scheduled(pa)
	})
}

func TestSharingAffinityStateLockLifecycle(t *testing.T) {
	fixture := newSharingAffinityStateFixture(t, true)
	first := fixture.allocateClaim(t, "first", "1Gi", sharingAffinityStateParameters)
	second := fixture.allocateClaim(t, "second", "2Gi", sharingAffinityStateParameters)

	state := fixture.waitForCapacity(t, "3Gi")
	fixture.assertAffinity(t, state, structured.DeviceAffinityState{
		Status:         structured.AffinityStatusLocked,
		LockedAffinity: map[string]string{"subnet": "subnet-a"},
	})

	fixture.deleteClaim(t, first)
	state = fixture.waitForCapacity(t, "2Gi")
	fixture.assertAffinity(t, state, structured.DeviceAffinityState{
		Status:         structured.AffinityStatusLocked,
		LockedAffinity: map[string]string{"subnet": "subnet-a"},
	})

	fixture.deleteClaim(t, second)
	state = fixture.waitForCapacity(t, "")
	if affinity, found := state.DeviceSharingAffinity.DeviceStates[fixture.device]; found {
		t.Fatalf("last claim deleted: affinity state remains: %#v", affinity)
	}
}

func TestSharingAffinityStateUnsatisfiedUnreconstructable(t *testing.T) {
	fixture := newSharingAffinityStateFixture(t, true)
	claim := fixture.allocateClaim(t, "unsatisfied", "1Gi",
		`{"apiVersion":"networking.example.com/v1","kind":"NICConfig"}`)

	// Capacity is published while the owner's record is still pending; only
	// a settled (non-stale) snapshot may be judged for Unreconstructable.
	state := fixture.waitForSettledCapacity(t, "1Gi")
	fixture.assertAffinity(t, state, structured.DeviceAffinityState{
		Status: structured.AffinityStatusUnreconstructable,
	})

	fixture.deleteClaim(t, claim)
	state = fixture.waitForSettledCapacity(t, "")
	if affinity, found := state.DeviceSharingAffinity.DeviceStates[fixture.device]; found {
		t.Fatalf("unreconstructable claim drained: affinity state remains: %#v", affinity)
	}
}

func TestSharingAffinityStateGateDisabled(t *testing.T) {
	fixture := newSharingAffinityStateFixture(t, false)
	fixture.allocateClaim(t, "first", "1Gi", sharingAffinityStateParameters)
	fixture.allocateClaim(t, "second", "2Gi", sharingAffinityStateParameters)

	state := fixture.waitForCapacity(t, "3Gi")
	if state.DeviceSharingAffinity.DeviceStates != nil {
		t.Fatalf("sharing affinity disabled: want nil affinity states, got %#v", state.DeviceSharingAffinity.DeviceStates)
	}
}

const sharingAffinityStateParameters = `{"apiVersion":"networking.example.com/v1","kind":"NICConfig","subnetID":"subnet-a"}`

type sharingAffinityStateFixture struct {
	ctx       context.Context
	client    kubernetes.Interface
	manager   *dynamicresources.DefaultDRAManager
	namespace string
	device    structured.DeviceID
}

func newSharingAffinityStateFixture(t *testing.T, enabled bool) *sharingAffinityStateFixture {
	t.Helper()
	// The server helper scopes process-global gates to this serial test.
	server := startSharingAffinityServer(t, enabled, framework.SharedEtcd())
	logger, ctx := klogtesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	t.Cleanup(cancel)
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("create sharing affinity state client: %v", err)
	}
	namespace, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "sharing-affinity-state-"},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create sharing affinity state namespace: %v", err)
	}

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	sliceInformer := draapi.NewInformerForResourceSlice(informerFactory)
	options := resourceslicetracker.Options{
		EnableDeviceTaintRules:   utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules),
		EnableConsumableCapacity: true,
		SliceLister:              draapi.NewResourceSliceLister(sliceInformer.GetIndexer()),
		SliceInformer:            sliceInformer,
		KubeClient:               client,
	}
	if options.EnableDeviceTaintRules {
		options.TaintInformer = informerFactory.Resource().V1().DeviceTaintRules()
	}
	tracker, err := resourceslicetracker.StartTracker(ctx, options)
	if err != nil {
		t.Fatalf("start sharing affinity ResourceSlice tracker: %v", err)
	}
	t.Cleanup(func() {
		cancel()
		tracker.Stop()
		informerFactory.Shutdown()
	})
	claimsCache := assumecache.NewAssumeCache(logger, informerFactory.Resource().V1().ResourceClaims().Informer(), "resource claim", "", nil)
	manager := dynamicresources.NewDRAManager(ctx, claimsCache, tracker, informerFactory)
	// Informer synchronization alone does not wait for assume-cache handlers.
	claimsSynced := claimsCache.AddEventHandler(cache.ResourceEventHandlerFuncs{}).HasSyncedChecker()
	informerFactory.Start(ctx.Done())
	syncCtx, syncCancel := context.WithTimeout(ctx, 30*time.Second)
	defer syncCancel()
	for informerType, synced := range informerFactory.WaitForCacheSync(syncCtx.Done()) {
		if !synced {
			t.Fatalf("sharing affinity informer %v did not synchronize", informerType)
		}
	}
	if !cache.WaitFor(syncCtx, "sharing affinity cache handlers", tracker.HasSyncedChecker(), claimsSynced) {
		t.Fatal("sharing affinity tracker and assume-cache handlers did not synchronize")
	}

	metadata := sharingAffinitySlice()
	metadata.Spec.Pool.ResourceSliceCount = 2
	devices := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "sharing-affinity-devices-"},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   metadata.Spec.Driver,
			Pool:     metadata.Spec.Pool,
			AllNodes: ptr.To(true),
			Devices: []resourceapi.Device{{
				Name:                     "nic",
				AllowMultipleAllocations: ptr.To(true),
				Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"memory": {Value: resource.MustParse("8Gi")},
				},
			}},
		},
	}
	published := make(map[string]*draapi.ResourceSlice)
	for _, slice := range []*resourceapi.ResourceSlice{metadata, devices} {
		created, err := client.ResourceV1().ResourceSlices().Create(ctx, slice, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("publish sharing affinity pool slice: %v", err)
		}
		var internalCreated draapi.ResourceSlice
		if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(created, &internalCreated, nil); err != nil {
			t.Fatalf("convert sharing affinity pool slice: %v", err)
		}
		published[created.Name] = &internalCreated
	}
	// Claim events must not race ahead of complete pool metadata observations.
	if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		slices, err := tracker.ListPatchedResourceSlices()
		if err != nil {
			return false, err
		}
		observed := 0
		for _, slice := range slices {
			if slice.Spec.Driver.String() != metadata.Spec.Driver || slice.Spec.Pool.Name.String() != metadata.Spec.Pool.Name {
				continue
			}
			want, found := published[slice.Name]
			if !found || slice.UID != want.UID || !reflect.DeepEqual(slice.Spec, want.Spec) {
				return false, nil
			}
			observed++
		}
		return observed == 2, nil
	}); err != nil {
		t.Fatalf("wait for complete sharing affinity metadata/device pool in tracker: %v", err)
	}
	return &sharingAffinityStateFixture{
		ctx:       ctx,
		client:    client,
		manager:   manager,
		namespace: namespace.Name,
		device:    structured.MakeDeviceID(metadata.Spec.Driver, metadata.Spec.Pool.Name, "nic"),
	}
}

func (f *sharingAffinityStateFixture) allocateClaim(t *testing.T, name, capacity, parameters string) *resourceapi.ResourceClaim {
	t.Helper()
	claims := f.client.ResourceV1().ResourceClaims(f.namespace)
	claim, err := claims.Create(f.ctx, &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{{
					Name: "nic",
					Exactly: &resourceapi.ExactDeviceRequest{
						DeviceClassName: "sharing-affinity",
						AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
						Count:           1,
					},
				}},
				Config: []resourceapi.DeviceClaimConfiguration{{
					Requests: []string{"nic"},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver:     f.device.Driver.String(),
							Parameters: runtime.RawExtension{Raw: []byte(parameters)},
						},
					},
				}},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create sharing affinity claim %q: %v", name, err)
	}
	claim.Status.Allocation = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Request: "nic",
				Driver:  f.device.Driver.String(),
				Pool:    f.device.Pool.String(),
				Device:  f.device.Device.String(),
				ShareID: ptr.To(uuid.NewUUID()),
				ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse(capacity),
				},
			}},
		},
	}
	claim, err = claims.UpdateStatus(f.ctx, claim, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("allocate sharing affinity claim %q through status update: %v", name, err)
	}
	return claim
}

func (f *sharingAffinityStateFixture) deleteClaim(t *testing.T, claim *resourceapi.ResourceClaim) {
	t.Helper()
	if err := f.client.ResourceV1().ResourceClaims(f.namespace).Delete(f.ctx, claim.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete sharing affinity claim %q: %v", claim.Name, err)
	}
}

func (f *sharingAffinityStateFixture) waitForCapacity(t *testing.T, capacity string) *structured.AllocatedState {
	t.Helper()
	var want resource.Quantity
	if capacity != "" {
		want = resource.MustParse(capacity)
	}
	key := draapi.MakeFullyQualifiedName("memory", f.device.Driver, draapi.MakeUniqueString)
	var state *structured.AllocatedState
	var gatherErr error
	err := wait.PollUntilContextTimeout(f.ctx, 10*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		state, gatherErr = f.manager.ResourceClaims().GatherAllocatedState()
		if gatherErr != nil {
			// GatherAllocatedState asks callers to retry concurrent revisions.
			return false, nil
		}
		if state == nil {
			return false, nil
		}
		if capacity == "" {
			_, found := state.AggregatedCapacity[f.device]
			return !found && !state.AllocatedSharedDeviceIDs.Has(f.device), nil
		}
		got := state.AggregatedCapacity[f.device][key]
		return got != nil && got.Cmp(want) == 0, nil
	})
	if err != nil {
		t.Fatalf("wait for device %s capacity %q: %v; last gather error: %v; last state: %#v", f.device, capacity, err, gatherErr, state)
	}
	if len(state.AllocatedDevices) != 0 {
		t.Fatalf("shared claims unexpectedly own dedicated devices: %v", state.AllocatedDevices)
	}
	if capacity != "" && (len(state.AllocatedSharedDeviceIDs) != 1 || !state.AllocatedSharedDeviceIDs.Has(f.device)) {
		t.Fatalf("want one shared device %s, got %v", f.device, state.AllocatedSharedDeviceIDs)
	}
	return state
}

// waitForSettledCapacity waits for exact capacity on a pool that has left
// StalePools. Every pending snapshot seen on the way must fail closed: the pool
// excluded and no ready (Clean/Locked) state exposed for the device.
func (f *sharingAffinityStateFixture) waitForSettledCapacity(t *testing.T, capacity string) *structured.AllocatedState {
	t.Helper()
	pool := schedulerapi.PoolID{Driver: f.device.Driver.String(), Pool: f.device.Pool.String()}
	var state *structured.AllocatedState
	err := wait.PollUntilContextTimeout(f.ctx, 10*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
		state = f.waitForCapacity(t, capacity)
		if !state.DeviceSharingAffinity.StalePools.Has(pool) {
			return true, nil
		}
		if affinity, found := state.DeviceSharingAffinity.DeviceStates[f.device]; found && affinity.Status != structured.AffinityStatusUnreconstructable {
			t.Fatalf("pending pool %v exposes non-fenced affinity %#v", pool, affinity)
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("wait for pool %v to settle with capacity %q: %v; last state: %#v", pool, capacity, err, state)
	}
	return state
}

func (f *sharingAffinityStateFixture) assertAffinity(t *testing.T, state *structured.AllocatedState, want structured.DeviceAffinityState) {
	t.Helper()
	got, found := state.DeviceSharingAffinity.DeviceStates[f.device]
	if !found || len(state.DeviceSharingAffinity.DeviceStates) != 1 || !reflect.DeepEqual(got, want) {
		t.Fatalf("device %s: want affinity %#v, got %#v (present: %t); all states: %#v", f.device, want, got, found, state.DeviceSharingAffinity.DeviceStates)
	}
}

// Keep these tests serial: the apiserver and scheduler share process-global gates.
func TestSharingAffinityRestart(t *testing.T) {
	env := newSAEnv(t)
	env.run("retained-owners", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
		stop := env.startScheduler()
		firstManager := env.gate.handle.SharedDRAManager()
		a := saScheduleOwner(s, "a", subnet("subnet-a"), "nic")
		b := saScheduleOwner(s, "b", subnet("subnet-a"), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))

		stop()
		saRetained(s, a, b)
		stop = env.startScheduler()
		defer stop()
		if firstManager == env.gate.handle.SharedDRAManager() {
			s.tCtx.Fatal("restart reused the first scheduler's DRA manager")
		}
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
		saRetained(s, a, b)
		c := saScheduleOwner(s, "c", subnet("subnet-a"), "nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "3Gi", "subnet", "subnet-a"))
		d := saRestartClaim(s, "d", subnet("subnet-b"), "nic")
		pd := s.pod("pd", "", d)
		s.notScheduled(pd, v1.PodReasonUnschedulable, nil)
		s.staysPending(pd, d)
		s.release(a.pod, a.claim)
		saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
		saRetained(s, b, c)
		s.staysPending(pd, d)
		s.release(b.pod, b.claim)
		s.release(c.pod, c.claim)
		s.scheduled(pd)
		saOnDevice(s, d, "pool/nic")
		saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", "subnet", "subnet-b"))
	})
}

func TestSharingAffinityReconstructionConflict(t *testing.T) {
	env := newSAEnv(t)
	for _, first := range []string{"subnet-a"} {
		env.run("conflicting-seeded-owners/"+first+"-first", func(s *saScope) {
			s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
			second := "subnet-b"
			if first == second {
				second = "subnet-a"
			}
			// Fixture-seeded historical allocations: a correct scheduler cannot
			// create conflicting owners in the first place.
			a := saSeedOwner(s, "a", "pool", "nic", subnet(first))
			b := saSeedOwner(s, "b", "pool", "nic", subnet(second))
			stop := env.startScheduler()
			defer stop()
			saWaitDevices(env, s, saUnknown("pool", "nic", "2Gi"))
			saRetained(s, a, b)
			c := saRestartClaim(s, "c", subnet(second), "nic")
			pc := s.pod("pc", "", c)
			s.notScheduled(pc, v1.PodReasonUnschedulable, nil)
			s.staysPending(pc, c)
			s.release(a.pod, a.claim)
			saWaitDevices(env, s, saUnknown("pool", "nic", "1Gi"))
			saRetained(s, b)
			s.staysPending(pc, c)
			s.release(b.pod, b.claim)
			s.scheduled(pc)
			saOnDevice(s, c, "pool/nic")
			saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", "subnet", second))
		})
	}
}

func TestSharingAffinityLegacyMigration(t *testing.T) {
	env := newSAEnv(t)
	env.run("mixed-owners-and-unaffected-devices", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("dev1"), saRestartDevice("dev2"))
		s.slice("other", "other", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("other"))
		// Fixture-seeded legacy/historical allocations, all with bound consumers.
		l1 := saSeedOwner(s, "l1", "pool", "dev1", "")
		l2 := saSeedOwner(s, "l2", "pool", "dev1", "")
		r1 := saSeedOwner(s, "r1", "pool", "dev1", subnet("subnet-a"))
		r2 := saSeedOwner(s, "r2", "pool", "dev2", subnet("subnet-a"))
		stop := env.startScheduler()
		defer stop()
		saWaitDevices(env, s, saUnknown("pool", "dev1", "3Gi"),
			saLocked("pool", "dev2", "1Gi", "subnet", "subnet-a"), saClean("other", "other"))
		saRetained(s, l1, l2, r1, r2)
		healthy := saScheduleOwner(s, "healthy", subnet("subnet-a"), "dev2")
		other := saScheduleOwner(s, "other", subnet("subnet-b"), "other")
		saOnDevice(s, healthy.claim, "pool/dev2")
		saOnDevice(s, other.claim, "other/other")
		saWaitDevices(env, s, saUnknown("pool", "dev1", "3Gi"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		next := saRestartClaim(s, "next", subnet("subnet-a"), "dev1")
		pnext := s.pod("pnext", "", next)
		s.notScheduled(pnext, v1.PodReasonUnschedulable, nil)
		s.staysPending(pnext, next)
		s.release(l1.pod, l1.claim)
		saWaitDevices(env, s, saUnknown("pool", "dev1", "2Gi"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		saRetained(s, l2, r1, r2, healthy, other)
		s.staysPending(pnext, next)
		s.release(l2.pod, l2.claim)
		saWaitDevices(env, s, saUnknown("pool", "dev1", "1Gi"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		saRetained(s, r1, r2, healthy, other)
		// Even removing every legacy owner cannot clear a genuine fence
		// while the reconstructable owner remains on the device.
		s.staysPending(pnext, next)
		s.release(r1.pod, r1.claim)
		s.scheduled(pnext)
		saOnDevice(s, next, "pool/dev1")
		saWaitDevices(env, s, saLocked("pool", "dev1", "1Gi", "subnet", "subnet-a"),
			saLocked("pool", "dev2", "2Gi", "subnet", "subnet-a"),
			saLocked("other", "other", "1Gi", "subnet", "subnet-b"))
		saRetained(s, r2, healthy, other)
	})
}

func TestSharingAffinityMetadataReconciliation(t *testing.T) {
	env := newSAEnv(t)
	stop := env.startScheduler()
	defer stop()
	for _, tc := range []struct {
		name       string
		key        string
		expression string
		newParams  string
		equal      bool
	}{
		{
			name: "equal-map-dual-path", key: "subnet",
			expression: `has(parameters.subnet) ? parameters.subnet : parameters.subnetID`,
			newParams:  nicParams(`"subnet":"subnet-a"`), equal: true,
		},
		{
			name: "renamed-key", key: "vpc-subnet", expression: saSubnetExpr,
			newParams: subnet("subnet-a"),
		},
		{
			name: "changed-value", key: "subnet", expression: `parameters.?vlan.orValue("")`,
			newParams: nicParams(`"subnetID":"subnet-a","vlan":"subnet-b"`),
		},
	} {
		env.run(tc.name, func(s *saScope) {
			slices := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("nic"))
			oldParams := nicParams(`"subnetID":"subnet-a","vlan":"subnet-b"`)
			a := saScheduleOwner(s, "a", oldParams, "nic")
			b := saScheduleOwner(s, "b", oldParams, "nic")
			saWaitDevices(env, s, saLocked("pool", "nic", "2Gi", "subnet", "subnet-a"))
			extractors := subnetExtractor(tc.expression)
			extractors[0].Name = tc.key
			want := saUnknown("pool", "nic", "2Gi")
			if tc.equal {
				want = saLocked("pool", "nic", "2Gi", "subnet", "subnet-a")
			}
			saRepublishAndProbe(env, s, slices, 2, extractors, tc.newParams, want)
			saRetained(s, a, b)
			c := saRestartClaim(s, "c", tc.newParams, "nic")
			pc := s.pod("pc", "", c)
			if tc.equal {
				s.scheduled(pc)
				saOnDevice(s, c, "pool/nic")
				saWaitDevices(env, s, saLocked("pool", "nic", "3Gi", "subnet", "subnet-a"))
				saRetained(s, a, b)
				return
			}
			s.notScheduled(pc, v1.PodReasonUnschedulable, nil)
			s.staysPending(pc, c)
			// Returning to the old map must not erase an established fence.
			saRepublishAndProbe(env, s, slices, 3, subnetExtractor(saSubnetExpr), oldParams, want)
			saRetained(s, a, b)
			s.staysPending(pc, c)
			saRepublishAndProbe(env, s, slices, 4, extractors, tc.newParams, want)
			s.release(a.pod, a.claim)
			saWaitDevices(env, s, saUnknown("pool", "nic", "1Gi"))
			saRetained(s, b)
			s.staysPending(pc, c)
			s.release(b.pod, b.claim)
			s.scheduled(pc)
			saOnDevice(s, c, "pool/nic")
			value := "subnet-a"
			if tc.name == "changed-value" {
				value = "subnet-b"
			}
			saWaitDevices(env, s, saLocked("pool", "nic", "1Gi", tc.key, value))
		})
	}
}

func TestSharingAffinityGateReenable(t *testing.T) {
	env := newSAEnv(t)
	env.run("retained-reconstructable-and-legacy-owners", func(s *saScope) {
		slices := s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("dev-r"), saRestartDevice("dev-l"))
		s.slice("fallback", "fallback", 1, 1, nil, saRestartDevice("fallback"))
		// Fixture-seeded legacy/historical allocations. A gate-off scheduler
		// cannot establish these owners in an affinity-declaring pool.
		r := saSeedOwner(s, "r", "pool", "dev-r", subnet("subnet-a"))
		l := saSeedOwner(s, "l", "pool", "dev-l", "")
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRASharingAffinity, false)
		stored, err := s.tCtx.Client().ResourceV1().ResourceSlices().Get(s.tCtx, slices[1].Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "read persisted affinity metadata with scheduler gate off")
		if !reflect.DeepEqual(stored.Spec.SharingAffinity, subnetExtractor(saSubnetExpr)) {
			s.tCtx.Fatalf("affinity metadata was not retained: %#v", stored.Spec.SharingAffinity)
		}
		stop := env.startScheduler()
		firstManager := env.gate.handle.SharedDRAManager()
		saWaitDevices(env, s, saWantDevice{pool: "pool", device: "dev-r", capacity: "1Gi"},
			saWantDevice{pool: "pool", device: "dev-l", capacity: "1Gi"})
		state, err := firstManager.ResourceClaims().GatherAllocatedState()
		s.tCtx.ExpectNoError(err, "gather gate-disabled accounting")
		if state.DeviceSharingAffinity.DeviceStates != nil {
			s.tCtx.Fatalf("gate-disabled scheduler populated affinity: %#v", state.DeviceSharingAffinity.DeviceStates)
		}
		fallback := saScheduleOwner(s, "fallback", subnet("subnet-b"), "")
		saOnDevice(s, fallback.claim, "fallback/fallback")
		blocked := saRestartClaim(s, "blocked", subnet("subnet-a"), "dev-r")
		pblocked := s.pod("pblocked", "", blocked)
		s.notScheduled(pblocked, v1.PodReasonUnschedulable, nil)
		s.staysPending(pblocked, blocked)
		saWaitDevices(env, s, saWantDevice{pool: "pool", device: "dev-r", capacity: "1Gi"},
			saWantDevice{pool: "pool", device: "dev-l", capacity: "1Gi"},
			saWantDevice{pool: "fallback", device: "fallback", capacity: "1Gi"})
		saRetained(s, r, l, fallback)
		// Restart without a pending placement, so the first post-restart
		// snapshot proves reconstruction of exactly the retained owners.
		s.release(pblocked, blocked)
		stop()
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRASharingAffinity, true)
		stop = env.startScheduler()
		defer stop()
		if firstManager == env.gate.handle.SharedDRAManager() {
			s.tCtx.Fatal("gate reenable reused the gate-disabled DRA manager")
		}
		saWaitDevices(env, s, saLocked("pool", "dev-r", "1Gi", "subnet", "subnet-a"),
			saUnknown("pool", "dev-l", "1Gi"),
			saWantDevice{pool: "fallback", device: "fallback", capacity: "1Gi"})
		saRetained(s, r, l, fallback)
		saRepublishAndProbe(env, s, slices, 2, subnetExtractor(saSubnetExpr), subnet("subnet-a"),
			saLocked("pool", "dev-r", "1Gi", "subnet", "subnet-a"), saUnknown("pool", "dev-l", "1Gi"))
		compatible := saScheduleOwner(s, "compatible", subnet("subnet-a"), "dev-r")
		saOnDevice(s, compatible.claim, "pool/dev-r")
		incompatible := saRestartClaim(s, "incompatible", subnet("subnet-b"), "dev-r")
		pincompatible := s.pod("pincompatible", "", incompatible)
		s.notScheduled(pincompatible, v1.PodReasonUnschedulable, nil)
		s.staysPending(pincompatible, incompatible)
		legacy := saRestartClaim(s, "legacy", subnet("subnet-a"), "dev-l")
		plegacy := s.pod("plegacy", "", legacy)
		s.notScheduled(plegacy, v1.PodReasonUnschedulable, nil)
		s.staysPending(plegacy, legacy)
		saWaitDevices(env, s, saLocked("pool", "dev-r", "2Gi", "subnet", "subnet-a"),
			saUnknown("pool", "dev-l", "1Gi"),
			saWantDevice{pool: "fallback", device: "fallback", capacity: "1Gi"})
		saRetained(s, r, l, fallback, compatible)
	})
}

func saRestartDevice(name string) saDevice {
	d := shared(name, "16Gi")
	d.attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{"slot": {StringValue: ptr.To(name)}}
	return d
}

func saNonMarkerSelector(driver string) string {
	return fmt.Sprintf(`!("marker" in device.attributes[%q])`, driver)
}

func saMarkerSelector(driver, marker string) string {
	return fmt.Sprintf(`"marker" in device.attributes[%q] && device.attributes[%q].marker == %q`, driver, driver, marker)
}

func saRestartClaim(s *saScope, name, params, slot string) *resourceapi.ResourceClaim {
	s.tCtx.Helper()
	expression := saNonMarkerSelector(s.driver)
	if slot != "" {
		expression += fmt.Sprintf(` && "slot" in device.attributes[%q] && device.attributes[%q].slot == %q`, s.driver, s.driver, slot)
	}
	return saSelectedClaim(s, name, params, expression)
}

func saSelectedClaim(s *saScope, name, params, expression string) *resourceapi.ResourceClaim {
	s.tCtx.Helper()
	var configs []saConfig
	if params != "" {
		configs = []saConfig{{requests: []string{"nic"}, params: params}}
	}
	return s.claim(name, []saRequest{{
		name: "nic", memory: "1Gi",
		selectors: []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{Expression: expression}}},
	}}, configs...)
}

type saOwner struct {
	pod   *v1.Pod
	claim *resourceapi.ResourceClaim
}

func saScheduleOwner(s *saScope, name, params, slot string) saOwner {
	s.tCtx.Helper()
	claim := saRestartClaim(s, name, params, slot)
	pod := s.scheduled(s.pod("p"+name, "", claim))
	claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(s.tCtx, claim.Name, metav1.GetOptions{})
	s.tCtx.ExpectNoError(err, "capture scheduled claim")
	if claim.Status.Allocation == nil {
		s.tCtx.Fatalf("scheduled pod %s has no claim allocation", pod.Name)
	}
	if slot != "" {
		results := claim.Status.Allocation.Devices.Results
		if len(results) != 1 || results[0].Device != slot || results[0].ShareID == nil {
			s.tCtx.Fatalf("claim %s: expected shared allocation on %s, got %+v", claim.Name, slot, results)
		}
	}
	return saOwner{pod: pod, claim: claim}
}

// saSeedOwner is only for fixture-seeded legacy/historical allocations which
// cannot be produced by a correctly enforcing scheduler.
func saSeedOwner(s *saScope, name, pool, device, params string) saOwner {
	s.tCtx.Helper()
	claim := saRestartClaim(s, name, params, device)
	pod := st.MakePod().Name("p" + name).Container("c").Obj()
	pod.Spec.NodeName = saNode
	pod = createPod(s.tCtx, s.ns, "", pod, claim)
	claim.Status.Allocation = &resourceapi.AllocationResult{
		NodeSelector: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{
			MatchFields: []v1.NodeSelectorRequirement{{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{saNode}}},
		}}},
		Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
			Request: "nic", Driver: s.driver, Pool: pool, Device: device,
			ShareID:          ptr.To(uuid.NewUUID()),
			ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{saMemory: resource.MustParse("1Gi")},
		}}},
	}
	for _, config := range claim.Spec.Devices.Config {
		claim.Status.Allocation.Devices.Config = append(claim.Status.Allocation.Devices.Config, resourceapi.DeviceAllocationConfiguration{
			Source: resourceapi.AllocationConfigSourceClaim, Requests: config.Requests, DeviceConfiguration: config.DeviceConfiguration,
		})
	}
	claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{{Resource: "pods", Name: pod.Name, UID: pod.UID}}
	claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).UpdateStatus(s.tCtx, claim, metav1.UpdateOptions{})
	s.tCtx.ExpectNoError(err, "fixture-seed historical allocation and bound consumer")
	return saOwner{pod: pod, claim: claim}
}

func saRetained(s *saScope, owners ...saOwner) {
	s.tCtx.Helper()
	for _, owner := range owners {
		pod, err := s.tCtx.Client().CoreV1().Pods(s.ns).Get(s.tCtx, owner.pod.Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "get retained bound pod")
		if pod.UID != owner.pod.UID || pod.Spec.NodeName != saNode || pod.DeletionTimestamp != nil {
			s.tCtx.Fatalf("retained pod %s was replaced, unbound, or deleted: %+v", pod.Name, pod)
		}
		claim, err := s.tCtx.Client().ResourceV1().ResourceClaims(s.ns).Get(s.tCtx, owner.claim.Name, metav1.GetOptions{})
		s.tCtx.ExpectNoError(err, "get retained allocated claim")
		if claim.UID != owner.claim.UID || claim.ResourceVersion != owner.claim.ResourceVersion ||
			claim.DeletionTimestamp != nil || !reflect.DeepEqual(claim.Status.Allocation, owner.claim.Status.Allocation) ||
			!reflect.DeepEqual(claim.Status.ReservedFor, owner.claim.Status.ReservedFor) {
			s.tCtx.Fatalf("retained claim %s changed: before=%+v, after=%+v", claim.Name, owner.claim, claim)
		}
	}
}

func saOnDevice(s *saScope, claim *resourceapi.ResourceClaim, want string) {
	s.tCtx.Helper()
	if got := s.device(claim); got != want {
		s.tCtx.Fatalf("claim %s allocated on %s, want %s", claim.Name, got, want)
	}
}

type saWantDevice struct {
	pool, device, capacity string
	affinity               structured.DeviceAffinityState
}

func saLocked(pool, device, capacity, key, value string) saWantDevice {
	return saWantDevice{pool: pool, device: device, capacity: capacity,
		affinity: structured.DeviceAffinityState{Status: structured.AffinityStatusLocked, LockedAffinity: map[string]string{key: value}}}
}

func saUnknown(pool, device, capacity string) saWantDevice {
	return saWantDevice{pool: pool, device: device, capacity: capacity,
		affinity: structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}}
}

func saClean(pool, device string) saWantDevice {
	return saWantDevice{pool: pool, device: device}
}

// saCheckDevices compares all requested devices in one immutable snapshot.
func saCheckDevices(s *saScope, state *schedulerapi.AllocatedState, wants ...saWantDevice) error {
	if state == nil {
		return fmt.Errorf("allocated state is nil")
	}
	for _, want := range wants {
		id := structured.MakeDeviceID(s.driver, want.pool, want.device)
		if state.DeviceSharingAffinity.StalePools.Has(schedulerapi.PoolID{Driver: s.driver, Pool: want.pool}) {
			return fmt.Errorf("pool %s remains stale", want.pool)
		}
		got, found := state.DeviceSharingAffinity.DeviceStates[id]
		if !reflect.DeepEqual(got, want.affinity) || (want.affinity.Status == structured.AffinityStatusClean && found) {
			return fmt.Errorf("device %s: affinity=%+v (present=%t), want %+v", id, got, found, want.affinity)
		}
		if state.AllocatedDevices.Has(id) {
			return fmt.Errorf("device %s unexpectedly has a dedicated allocation", id)
		}
		capacity := state.AggregatedCapacity[id][draapi.MakeFullyQualifiedName(saMemory, draapi.MakeUniqueString(s.driver), draapi.MakeUniqueString)]
		if want.capacity == "" {
			if _, found := state.AggregatedCapacity[id]; found || state.AllocatedSharedDeviceIDs.Has(id) {
				return fmt.Errorf("device %s has not drained: capacity=%v shared=%t", id, capacity, state.AllocatedSharedDeviceIDs.Has(id))
			}
		} else if capacity == nil || capacity.Cmp(resource.MustParse(want.capacity)) != 0 || !state.AllocatedSharedDeviceIDs.Has(id) {
			return fmt.Errorf("device %s: capacity=%v shared=%t, want %s shared", id, capacity, state.AllocatedSharedDeviceIDs.Has(id), want.capacity)
		}
	}
	return nil
}

func saWaitDevices(env *saEnv, s *saScope, wants ...saWantDevice) {
	s.tCtx.Helper()
	s.tCtx.Eventually(func() error {
		state, err := env.gate.handle.SharedDRAManager().ResourceClaims().GatherAllocatedState()
		if err != nil {
			return err
		}
		return saCheckDevices(s, state, wants...)
	}).WithTimeout(schedulingTimeout).Should(gomega.Succeed(), "wait for reconstructed device state")
}

// Binding a generation-specific marker proves both metadata publication and
// pool reconciliation. A bare !StalePools check can still see the old generation.
func saRepublishAndProbe(env *saEnv, s *saScope, slices saSlices, generation int64,
	extractors []resourceapi.SharingAffinityExtractor, params string, wants ...saWantDevice) {
	s.tCtx.Helper()
	marker := fmt.Sprintf("m%d", generation)
	device := fmt.Sprintf("dev-marker-%d", generation)
	s.updateSlice(slices, func(slice *resourceapi.ResourceSlice) {
		slice.Spec.Pool.Generation = generation
		if len(slice.Spec.SharingAffinity) != 0 {
			slice.Spec.SharingAffinity = extractors
			return
		}
		// Ordinary devices deliberately precede the marker: a missing marker
		// attribute must be false, not a CEL error aborting the allocation.
		slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{
			Name: device, AllowMultipleAllocations: ptr.To(true),
			Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
				resourceapi.QualifiedName(s.driver + "/marker"): {StringValue: ptr.To(marker)},
			},
			Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{saMemory: {Value: resource.MustParse("16Gi")}},
		})
	})
	claim := saSelectedClaim(s, "probe-"+marker, params, saMarkerSelector(s.driver, marker))
	pod := s.scheduled(s.pod("probe-"+marker, "", claim))
	saOnDevice(s, claim, slices[0].Spec.Pool.Name+"/"+device)
	state, err := env.gate.handle.SharedDRAManager().ResourceClaims().GatherAllocatedState()
	s.tCtx.ExpectNoError(err, "gather state after generation-specific marker bound")
	s.tCtx.ExpectNoError(saCheckDevices(s, state, wants...), "reconciled owner state after marker bound")
	s.release(pod, claim)
	saWaitDevices(env, s, append(wants, saClean(slices[0].Spec.Pool.Name, device))...)
}

func TestSharingAffinityMarkerSelectorRejectsOrdinaryDevice(t *testing.T) {
	const driver = "sharing-affinity.example.com"
	for _, selector := range []struct {
		name       string
		expression string
		want       []bool
	}{
		{name: "probe", expression: saMarkerSelector(driver, "m2"), want: []bool{false, true, false}},
		{name: "workload", expression: saNonMarkerSelector(driver), want: []bool{true, false, false}},
	} {
		t.Run(selector.name, func(t *testing.T) {
			compiled := dracel.GetCompiler(dracel.Features{EnableConsumableCapacity: true}).CompileCELExpression(selector.expression, dracel.Options{})
			if compiled.Error != nil {
				t.Fatalf("compile marker selector: %v", compiled.Error)
			}
			for i, marker := range []string{"", "m2", "m1"} {
				var attributes draapi.DeviceAttributes
				if marker != "" {
					attributes = draapi.DeviceAttributes{
						Nested: map[draapi.UniqueString]map[draapi.UniqueString]any{
							draapi.MakeUniqueString(driver): {
								draapi.MakeUniqueString("marker"): marker,
							},
						},
					}
				}
				got, _, err := compiled.DeviceMatches(context.Background(), dracel.Device{
					Driver: driver, AllowMultipleAllocations: ptr.To(true), LookupUniqueString: draapi.MakeUniqueString, Attributes: attributes,
				})
				if err != nil || got != selector.want[i] {
					t.Fatalf("marker %q: matches=%t err=%v, want %t without evaluation error", marker, got, err, selector.want[i])
				}
			}
		})
	}
}

const (
	saDiagMismatch    = "incompatible with the device's sharing affinity lock"
	saDiagUnsatisfied = "does not produce a value for every sharing affinity extractor"
	saDiagUnknown     = "device sharing affinity is unknown because of legacy or inconsistent allocations"
	saDiagPending     = "not yet reconciled"
	saDiagRejectLog   = "Sharing affinity rejected candidate devices"
	saDiagExtractLog  = "Sharing affinity extraction failed"
)

// saEventRecord is one pre-coalescing Eventf call.
type saEventRecord struct {
	uid     types.UID
	reason  string
	message string
}

// saRecorderStore is shared by every recorder the spy hands out, including
// WithLogger derivatives.
type saRecorderStore struct {
	mutex    sync.Mutex
	inFlight int
	records  []saEventRecord
}

type saRecorderSpy struct {
	store    *saRecorderStore
	delegate events.EventRecorderLogger
}

var _ events.EventRecorderLogger = &saRecorderSpy{}

func (r *saRecorderSpy) Eventf(regarding runtime.Object, related runtime.Object, eventtype, reason, action, note string, args ...interface{}) {
	r.store.mutex.Lock()
	r.store.inFlight++
	r.store.mutex.Unlock()
	defer func() {
		r.store.mutex.Lock()
		r.store.inFlight--
		r.store.mutex.Unlock()
	}()
	var uid types.UID
	if obj, err := meta.Accessor(regarding); err == nil {
		uid = obj.GetUID()
	}
	r.store.mutex.Lock()
	r.store.records = append(r.store.records, saEventRecord{uid: uid, reason: reason, message: fmt.Sprintf(note, args...)})
	r.store.mutex.Unlock()
	r.delegate.Eventf(regarding, related, eventtype, reason, action, note, args...)
}

func (r *saRecorderSpy) WithLogger(logger klog.Logger) events.EventRecorderLogger {
	return &saRecorderSpy{store: r.store, delegate: r.delegate.WithLogger(logger)}
}

func (s *saRecorderStore) reset() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.records = nil
}

// failed returns the FailedScheduling records for uid and the in-flight count
// from one critical section, so callers get a consistent snapshot.
func (s *saRecorderStore) failed(uid types.UID) ([]string, int) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	var msgs []string
	for _, r := range s.records {
		if r.uid == uid && r.reason == "FailedScheduling" {
			msgs = append(msgs, r.message)
		}
	}
	return msgs, s.inFlight
}

// startSchedulerWithRecorderSpy mirrors util.StartSchedulerWithDone, but
// wraps the recorder factory so Eventf calls are counted before the
// broadcaster coalesces them, and runs with the given logger.
func (e *saEnv) startSchedulerWithRecorderSpy(logger klog.Logger) *saRecorderStore {
	gate := &sharingAffinityGate{waiting: map[string]bool{}, rejected: map[string]int{}, preBindHit: map[string]int{}}
	e.gate = gate
	registry := frameworkruntime.Registry{
		saGateName: func(_ context.Context, _ runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
			gate.handle = h
			return gate, nil
		},
	}
	tCtx := e.tCtx.WithLogger(logger).WithCancel()
	cfg := newSchedulerComponentConfig(tCtx, saSchedulerConfig)
	clientSet := tCtx.Client()
	informerFactory := scheduler.NewInformerFactory(clientSet, 0, nil)
	evtBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: clientSet.EventsV1()})
	go func() {
		<-tCtx.Done()
		evtBroadcaster.Shutdown()
	}()
	evtBroadcaster.StartRecordingToSink(tCtx.Done())
	store := &saRecorderStore{}
	realFactory := profile.NewRecorderFactory(evtBroadcaster)
	sched, err := scheduler.New(
		tCtx,
		clientSet,
		informerFactory,
		nil,
		func(name string) events.EventRecorderLogger {
			return &saRecorderSpy{store: store, delegate: realFactory(name)}
		},
		scheduler.WithKubeConfig(tCtx.RESTConfig()),
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithPercentageOfNodesToScore(cfg.PercentageOfNodesToScore),
		scheduler.WithPercentageOfPlacementsToScore(cfg.PercentageOfPlacementsToScore),
		scheduler.WithPodMaxBackoffSeconds(cfg.PodMaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(cfg.PodInitialBackoffSeconds),
		scheduler.WithExtenders(cfg.Extenders...),
		scheduler.WithParallelism(cfg.Parallelism),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
	)
	tCtx.ExpectNoError(err, "creating scheduler")
	informerFactory.Start(tCtx.Done())
	informerFactory.WaitForCacheSync(tCtx.Done())
	tCtx.ExpectNoError(sched.WaitForHandlersSync(tCtx), "waiting for handlers to sync")
	done := make(chan struct{})
	go func() {
		sched.Run(tCtx)
		close(done)
	}()
	e.tCtx.CleanupCtx(func(ktesting.TContext) {
		tCtx.Cancel("stop sharing affinity diagnostics scheduler")
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			e.tCtx.Fatal("sharing affinity scheduler did not stop within 30s")
		}
		informerFactory.Shutdown()
	})
	return store
}

// failedSchedulingEvents waits until the API has a FailedScheduling Event for
// the pod containing all fragments and checks none of them leak secrets.
// Persisted Events may be coalesced, so this is message evidence only.
func (s *saScope) failedSchedulingEvents(pod *v1.Pod, contains []string, absent ...string) []v1.Event {
	s.tCtx.Helper()
	var events []v1.Event
	s.tCtx.Eventually(func(tCtx ktesting.TContext) (string, error) {
		list, err := tCtx.Client().CoreV1().Events(s.ns).List(tCtx, metav1.ListOptions{
			FieldSelector: "involvedObject.kind=Pod,involvedObject.name=" + pod.Name + ",reason=FailedScheduling",
		})
		if err != nil {
			return "", err
		}
		events = list.Items
		var msgs []string
		for _, e := range list.Items {
			msgs = append(msgs, e.Message)
		}
		return strings.Join(msgs, "\n"), nil
	}).WithTimeout(schedulingTimeout).Should(gomega.And(append([]gomega.OmegaMatcher{gomega.Not(gomega.BeEmpty())},
		saContainAll(contains)...)...), "FailedScheduling event for pod %s", pod.Name)
	for _, e := range events {
		for _, a := range absent {
			if strings.Contains(e.Message, a) {
				s.tCtx.Fatalf("event for pod %s leaks %q: %s", pod.Name, a, e.Message)
			}
		}
	}
	return events
}

// saLogs gives access to the scheduler's buffered log output.
type saLogs struct {
	logger klog.Logger
}

func (l saLogs) text() string {
	return l.logger.GetSink().(klogtesting.Underlier).GetBuffer().String()
}

// expect waits for matching log entries (logging can race API status writes).
// Exclusions apply only to those entries, not verbose object dumps.
func (l saLogs) expect(s *saScope, pattern string, absent ...string) {
	s.tCtx.Helper()
	re := regexp.MustCompile(pattern)
	var matching []string
	s.tCtx.Eventually(func(ktesting.TContext) bool {
		matching = nil
		for _, entry := range saLogEntries(l.text()) {
			if re.MatchString(entry) {
				matching = append(matching, entry)
			}
		}
		return len(matching) > 0
	}).WithTimeout(schedulingTimeout).Should(gomega.BeTrueBecause("scheduler log should match %q", pattern))
	for _, entry := range matching {
		for _, a := range absent {
			if strings.Contains(entry, a) {
				s.tCtx.Fatalf("scheduler log entry contains %q:\n%s", a, entry)
			}
		}
	}
}

func saLogEntries(text string) []string {
	var entries []string
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "INFO ") || strings.HasPrefix(line, "ERROR ") || len(entries) == 0 {
			entries = append(entries, line)
			continue
		}
		entries[len(entries)-1] += "\n" + line
	}
	return entries
}

// saBudgetFixture is the admissible shared-runtime-budget fixture from the
// design (§5.1): constant cost per evaluation, admission-accepted, exhausted
// by the fifth claim-wide config because one budget covers all configs.
func saBudgetExpression() string {
	l := "["
	for i := range 16 {
		if i > 0 {
			l += ","
		}
		l += fmt.Sprint(i)
	}
	l += "]"
	return `has(parameters.kind) && ` + l + `.all(i, ` + l + `.all(j, "` + strings.Repeat("a", 4000) + `".matches("a{1,3}"))) ? "x" : "x"`
}

func saAttempts(s *saScope, result string) float64 {
	s.tCtx.Helper()
	values, err := testutil.GetCounterValuesFromGatherer(legacyregistry.DefaultGatherer, "scheduler_schedule_attempts_total",
		map[string]string{"profile": "default-scheduler"}, "result")
	s.tCtx.ExpectNoError(err, "read scheduler_schedule_attempts_total")
	return values[result]
}

// TestSharingAffinityDiagnostics drives the real scheduler and checks that
// each KEP diagnostic category reaches the pod condition, the
// FailedScheduling Event and the scheduler log, without leaking claim
// configuration values.
func TestSharingAffinityDiagnostics(t *testing.T) {
	env := newSAEnv(t)
	logs := saLogs{logger: klogtesting.NewLogger(t, klogtesting.NewConfig(klogtesting.Verbosity(7), klogtesting.BufferLogs(true)))}
	spy := env.startSchedulerWithRecorderSpy(logs.logger)

	// Runs first: the attempt metric is global, so this window must be the
	// only pending pod in the scheduler.
	env.run("mismatch", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr),
			shared("nic-0", "8Gi"), shared("nic-1", "8Gi"), shared("nic-2", "8Gi"))
		var owners []*v1.Pod
		var ownerClaims []*resourceapi.ResourceClaim
		for _, n := range []string{"a", "b", "c"} {
			c := s.nicClaim(n, "1Gi", subnet("subnet-"+n))
			owners = append(owners, s.scheduled(s.pod("p"+n, "", c)))
			ownerClaims = append(ownerClaims, c)
		}
		diagnosticExclusions := []string{saSubnetExpr, "secret-subnet-d", "subnet-a", "subnet-b", "subnet-c"}

		baseline := saAttempts(s, "unschedulable")
		spy.reset()
		d := s.nicClaim("d", "1Gi", subnet("secret-subnet-d"))
		pd := s.pod("pd", "", d)
		s.notScheduled(pd, v1.PodReasonUnschedulable, []string{saDiagMismatch}, diagnosticExclusions...)

		count := func() int {
			msgs, _ := spy.failed(pd.UID)
			return len(msgs)
		}
		s.tCtx.Eventually(func(ktesting.TContext) int { return count() }).WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">=", 1))
		// Node label updates requeue the unschedulable pod for new attempts.
		for i := range 2 {
			before := count()
			node, err := s.tCtx.Client().CoreV1().Nodes().Get(s.tCtx, "worker-1", metav1.GetOptions{})
			s.tCtx.ExpectNoError(err, "get unrelated node")
			node.Labels["sharing-affinity-diag-requeue"] = fmt.Sprintf("%s-%d", s.ns, i)
			_, err = s.tCtx.Client().CoreV1().Nodes().Update(s.tCtx, node, metav1.UpdateOptions{})
			s.tCtx.ExpectNoError(err, "update unrelated node label")
			s.tCtx.Eventually(func(ktesting.TContext) int { return count() }).WithTimeout(schedulingTimeout).Should(gomega.BeNumerically(">", before))
		}

		// Quiescent window: three consecutive equal reads spanning >=2s with
		// metric delta == recorder count and nothing inside Eventf.
		var attempts float64
		var msgs []string
		stable := 0
		var last time.Time
		var prevAttempts float64 = -1
		prevCount := -1
		err := wait.PollUntilContextTimeout(s.tCtx, 100*time.Millisecond, 30*time.Second, true, func(context.Context) (bool, error) {
			a := saAttempts(s, "unschedulable") - baseline
			m, inFlight := spy.failed(pd.UID)
			if inFlight != 0 || a != float64(len(m)) || a != prevAttempts || len(m) != prevCount {
				prevAttempts, prevCount, stable, last = a, len(m), 0, time.Now()
				return false, nil
			}
			stable++
			if stable >= 3 && time.Since(last) >= 2*time.Second {
				attempts, msgs = a, m
				return true, nil
			}
			return false, nil
		})
		s.tCtx.ExpectNoError(err, "scheduler did not quiesce with matching attempt metric and FailedScheduling emissions")
		if attempts < 3 {
			s.tCtx.Fatalf("want >=3 unschedulable attempts for pod %s, got %v", pd.Name, attempts)
		}
		if float64(len(msgs)) != attempts {
			s.tCtx.Fatalf("FailedScheduling emissions %d != attempts %v: three rejected devices must not add per-device events", len(msgs), attempts)
		}
		for _, m := range msgs {
			if !strings.Contains(m, saDiagMismatch) {
				s.tCtx.Fatalf("emission lacks mismatch fragment: %s", m)
			}
			for _, a := range append([]string{"nic-0", "nic-1", "nic-2", saDiagUnknown, saDiagUnsatisfied}, diagnosticExclusions...) {
				if strings.Contains(m, a) {
					s.tCtx.Fatalf("emission contains %q: %s", a, m)
				}
			}
		}
		s.unallocated(d)
		s.failedSchedulingEvents(pd, []string{saDiagMismatch}, append([]string{"nic-0", "nic-1", "nic-2"}, diagnosticExclusions...)...)
		logs.expect(s, saDiagRejectLog+`.*"`+s.ns+`/pd".*mismatch=[1-9]`, diagnosticExclusions...)

		s.release(pd, d)
		for i, o := range owners {
			s.release(o, ownerClaims[i])
		}
	})

	env.run("alternative-candidate-succeeds", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic-0", "8Gi"), shared("nic-1", "8Gi"))
		a := s.nicClaim("a", "1Gi", subnet("subnet-a"))
		s.scheduled(s.pod("pa", "", a))
		b := s.nicClaim("b", "1Gi", subnet("subnet-b"))
		pb := s.scheduled(s.pod("pb", "", b))
		logs.expect(s, `(?s)Reserved resource in allocation result.*claim="`+s.ns+`/b".*"subnetID":\s*"subnet-b"`)
		if s.device(a) == s.device(b) {
			s.tCtx.Fatalf("mismatching claims co-allocated on %s", s.device(a))
		}
		if msgs, _ := spy.failed(pb.UID); len(msgs) > 0 {
			s.tCtx.Fatalf("rejected sibling candidate produced FailedScheduling for schedulable pod: %v", msgs)
		}
		list, err := s.tCtx.Client().CoreV1().Events(s.ns).List(s.tCtx, metav1.ListOptions{})
		s.tCtx.ExpectNoError(err, "list events")
		for _, e := range list.Items {
			if e.InvolvedObject.Name == pb.Name && e.Reason == "FailedScheduling" {
				s.tCtx.Fatalf("rejected sibling candidate produced FailedScheduling for schedulable pod: %s", e.Message)
			}
		}
	})

	env.run("unsatisfied", func(s *saScope) {
		vlanExpr := `parameters.?vlanID.orValue("")`
		extractors := append(subnetExtractor(saSubnetExpr),
			resourceapi.SharingAffinityExtractor{Name: "vlan", APIGroup: "networking.example.com", Kind: "NICConfig", Expression: vlanExpr})
		s.slice("s", "pool", 1, 1, extractors, shared("nic", "8Gi"))
		// subnet extracts successfully; its vlan sibling yields "".
		a := s.nicClaim("a", "1Gi", subnet("secret-subnet-u"))
		pa := s.pod("pa", "", a)
		diagnosticExclusions := []string{saSubnetExpr, vlanExpr, "secret-subnet-u"}
		s.notScheduled(pa, v1.PodReasonUnschedulable, []string{saDiagUnsatisfied}, diagnosticExclusions...)
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{saDiagUnsatisfied}, diagnosticExclusions...)
		logs.expect(s, saDiagRejectLog+`.*"`+s.ns+`/pa".*unsatisfied=[1-9]`, diagnosticExclusions...)
	})

	env.run("evaluation-error/non-string", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", nicParams(`"subnetID":987654321`))
		pa := s.pod("pa", "", a)
		cond := s.notScheduled(pa, v1.PodReasonSchedulerError, []string{`extractor "subnet"`, "non-string"}, "987654321")
		for _, c := range cond.Status.Conditions {
			if strings.Contains(c.Message, saDiagUnsatisfied) {
				s.tCtx.Fatalf("evaluation error reported as unsatisfied: %s", c.Message)
			}
		}
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{"non-string"}, "987654321")
		logs.expect(s, saDiagExtractLog+`.*non-string`, "987654321")
	})

	env.run("evaluation-error/missing-field", func(s *saScope) {
		expr := `parameters.subnetID`
		s.slice("s", "pool", 1, 1, subnetExtractor(expr), shared("nic", "8Gi"))
		a := s.nicClaim("a", "1Gi", nicParams(`"vlanID":"secret-vlan-m"`))
		pa := s.pod("pa", "", a)
		diagnosticExclusions := []string{expr, "secret-vlan-m"}
		cond := s.notScheduled(pa, v1.PodReasonSchedulerError, []string{`extractor "subnet"`, "missing field"}, diagnosticExclusions...)
		for _, c := range cond.Status.Conditions {
			if strings.Contains(c.Message, saDiagUnsatisfied) {
				s.tCtx.Fatalf("missing field reported as unsatisfied: %s", c.Message)
			}
		}
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{"missing field"}, diagnosticExclusions...)
		logs.expect(s, saDiagExtractLog+`.*missing field`, diagnosticExclusions...)
	})

	env.run("evaluation-error/cost-exhausted", func(s *saScope) {
		expr := saBudgetExpression()
		if len(expr) > resourceapi.SharingAffinityCELExpressionMaxLength {
			s.tCtx.Fatalf("fixture expression length %d exceeds limit", len(expr))
		}
		extractors := []resourceapi.SharingAffinityExtractor{{Name: "b0", APIGroup: "example.com", Kind: "Config", Expression: expr}}
		// Creation through the API proves the fixture is admissible.
		s.slice("s", "pool", 1, 1, extractors, shared("nic", "8Gi"))
		const privateValue = "secret-cost-config"
		var configs []saConfig
		for range 8 {
			configs = append(configs, saConfig{params: fmt.Sprintf(`{"apiVersion":"example.com/v1","kind":"Config","privateValue":%q}`, privateValue)})
		}
		a := s.claim("a", []saRequest{{name: "nic", memory: "1Gi"}}, configs...)
		pa := s.pod("pa", "", a)
		re := regexp.MustCompile(`config [0-7]\b`)
		cond := s.notScheduled(pa, v1.PodReasonSchedulerError, []string{`extractor "b0"`, "cost exceeded"}, privateValue, strings.Repeat("a", 64))
		for _, c := range cond.Status.Conditions {
			if c.Type == v1.PodScheduled && !re.MatchString(c.Message) {
				s.tCtx.Fatalf("cost exhaustion should name a config index < 8 (shared budget): %s", c.Message)
			}
		}
		s.unallocated(a)
		s.failedSchedulingEvents(pa, []string{"cost exceeded"}, privateValue, strings.Repeat("a", 64))
		logs.expect(s, saDiagExtractLog+`.*cost`, privateValue, strings.Repeat("a", 64))
	})

	env.run("unknown-legacy-owner", func(s *saScope) {
		s.slice("s", "pool", 1, 1, subnetExtractor(saSubnetExpr), saRestartDevice("dev1"))
		legacy := saSeedOwner(s, "legacy", "pool", "dev1", "")
		next := saRestartClaim(s, "next", subnet("secret-next"), "dev1")
		pnext := s.pod("pnext", "", next)
		s.notScheduled(pnext, v1.PodReasonUnschedulable, []string{saDiagUnknown}, "secret-next", saDiagPending)
		s.staysPending(pnext, next)
		s.failedSchedulingEvents(pnext, []string{saDiagUnknown}, "secret-next")
		logs.expect(s, saDiagRejectLog+`.*"`+s.ns+`/pnext".*unknown=[1-9]`, "secret-next")
		// Releasing the legacy owner clears the unknown state.
		s.release(legacy.pod, legacy.claim)
		s.scheduled(pnext)
		saOnDevice(s, next, "pool/dev1")
	})
}
