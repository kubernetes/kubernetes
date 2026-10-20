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
	"reflect"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

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
	_, ctx := ktesting.NewTestContext(t)
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
