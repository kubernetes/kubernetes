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

package apiserver

import (
	"context"
	"fmt"
	"math/big"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	apiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

// calculateShardRange divides the 64-bit hash space evenly across shards and
// returns the hex-encoded start (inclusive) and end (exclusive) for the given index.
func calculateShardRange(index, total int) (start, end string) {
	maxVal := new(big.Int).Lsh(big.NewInt(1), 64) // 2^64
	span := new(big.Int).Div(maxVal, big.NewInt(int64(total)))

	startVal := new(big.Int).Mul(span, big.NewInt(int64(index)))
	endVal := new(big.Int).Mul(span, big.NewInt(int64(index+1)))
	if index == total-1 {
		endVal = maxVal // last shard covers remainder
	}

	start = fmt.Sprintf("0x%016x", startVal)
	end = fmt.Sprintf("0x%016x", endVal)
	return start, end
}

func shardSelectorString(index, total int) string {
	return shardSelectorStringForField("object.metadata.uid", index, total)
}

func shardSelectorStringForField(field string, index, total int) string {
	start, end := calculateShardRange(index, total)
	return fmt.Sprintf("shardRange(%s, '%s', '%s')", field, start, end)
}

// objectInShard returns true if the object's UID hash falls within the given shard range.
func objectInShard(uid string, index, total int) bool {
	return valueInShard(uid, index, total)
}

// valueInShard returns true if the hash of value falls within the given shard range.
func valueInShard(value string, index, total int) bool {
	start, end := calculateShardRange(index, total)
	hash := "0x" + sharding.HashField(value)
	return !sharding.HexLess(hash, start) && sharding.HexLess(hash, end)
}

type shardingResource struct {
	name   string
	gvr    schema.GroupVersionResource
	create func(ctx context.Context, namespace, generateName string) (metav1.Object, error)
	list   func(ctx context.Context, namespace string, opts metav1.ListOptions) (*shardedList, error)
	watch  func(ctx context.Context, namespace string, opts metav1.ListOptions) (watch.Interface, error)
	update func(ctx context.Context, obj metav1.Object) error
	delete func(ctx context.Context, namespace, name string) error
}

type shardedList struct {
	resourceVersion string
	shardInfo       *metav1.ShardInfo
	items           []metav1.Object
}

func setupSharding(t *testing.T, flags ...string) (context.Context, clientset.Interface, dynamic.Interface, []shardingResource) {
	t.Helper()
	ctx := ktesting.Init(t)

	server, err := apiservertesting.StartTestServer(t, apiservertesting.NewDefaultTestServerOptions(), flags, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(server.TearDownFn)

	client, err := clientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	apiExtensionsClient, err := apiextensionsclientset.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	crd, err := fixtures.CreateNewV1CustomResourceDefinition(fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped), apiExtensionsClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}
	crGVR := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural}

	resources := []shardingResource{
		{
			name: "configmaps",
			gvr:  v1.SchemeGroupVersion.WithResource("configmaps"),
			create: func(ctx context.Context, namespace, generateName string) (metav1.Object, error) {
				return client.CoreV1().ConfigMaps(namespace).Create(ctx, &v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{GenerateName: generateName},
				}, metav1.CreateOptions{})
			},
			list: func(ctx context.Context, namespace string, opts metav1.ListOptions) (*shardedList, error) {
				list, err := client.CoreV1().ConfigMaps(namespace).List(ctx, opts)
				if err != nil {
					return nil, err
				}
				items := make([]metav1.Object, 0, len(list.Items))
				for i := range list.Items {
					items = append(items, &list.Items[i])
				}
				return &shardedList{resourceVersion: list.ResourceVersion, shardInfo: list.ShardInfo, items: items}, nil
			},
			watch: func(ctx context.Context, namespace string, opts metav1.ListOptions) (watch.Interface, error) {
				return client.CoreV1().ConfigMaps(namespace).Watch(ctx, opts)
			},
			update: func(ctx context.Context, obj metav1.Object) error {
				_, err := client.CoreV1().ConfigMaps(obj.GetNamespace()).Update(ctx, obj.(*v1.ConfigMap), metav1.UpdateOptions{})
				return err
			},
			delete: func(ctx context.Context, namespace, name string) error {
				return client.CoreV1().ConfigMaps(namespace).Delete(ctx, name, metav1.DeleteOptions{})
			},
		},
		{
			name: "customresources",
			gvr:  crGVR,
			create: func(ctx context.Context, namespace, generateName string) (metav1.Object, error) {
				obj := fixtures.NewVersionedNoxuInstance(namespace, "", crGVR.Version)
				obj.SetGenerateName(generateName)
				return dynamicClient.Resource(crGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
			},
			list: func(ctx context.Context, namespace string, opts metav1.ListOptions) (*shardedList, error) {
				list, err := dynamicClient.Resource(crGVR).Namespace(namespace).List(ctx, opts)
				if err != nil {
					return nil, err
				}
				items := make([]metav1.Object, 0, len(list.Items))
				for i := range list.Items {
					items = append(items, &list.Items[i])
				}
				return &shardedList{resourceVersion: list.GetResourceVersion(), shardInfo: list.GetShardInfo(), items: items}, nil
			},
			watch: func(ctx context.Context, namespace string, opts metav1.ListOptions) (watch.Interface, error) {
				return dynamicClient.Resource(crGVR).Namespace(namespace).Watch(ctx, opts)
			},
			update: func(ctx context.Context, obj metav1.Object) error {
				_, err := dynamicClient.Resource(crGVR).Namespace(obj.GetNamespace()).Update(ctx, obj.(*unstructured.Unstructured), metav1.UpdateOptions{})
				return err
			},
			delete: func(ctx context.Context, namespace, name string) error {
				return dynamicClient.Resource(crGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
			},
		},
	}
	return ctx, client, dynamicClient, resources
}

func TestShardedList(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, true)

	ctx, client, _, resources := setupSharding(t)

	for _, r := range resources {
		t.Run(r.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(client, "shard-list-"+r.name, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			// Create a batch of objects.
			const numObjects = 20
			created := make([]metav1.Object, 0, numObjects)
			for range numObjects {
				obj, err := r.create(ctx, ns.Name, "shard-test-")
				if err != nil {
					t.Fatalf("failed to create object: %v", err)
				}
				created = append(created, obj)
			}

			const numShards = 3
			allFound := make(map[string]bool)

			for shard := range numShards {
				selector := shardSelectorString(shard, numShards)
				list, err := r.list(ctx, ns.Name, metav1.ListOptions{
					ShardSelector: selector,
				})
				if err != nil {
					t.Fatalf("shard %d: failed to list: %v", shard, err)
				}

				// The response must include shard info with the selector echoed back.
				if shardInfo := list.shardInfo; shardInfo == nil {
					t.Errorf("shard %d: expected shardInfo to be set", shard)
				} else if shardInfo.Selector != selector {
					t.Errorf("shard %d: expected shardInfo.selector=%q, got %q", shard, selector, shardInfo.Selector)
				}

				for _, obj := range list.items {
					uid := string(obj.GetUID())
					if !objectInShard(uid, shard, numShards) {
						t.Errorf("shard %d: object %s (UID %s) should not be in this shard", shard, obj.GetName(), uid)
					}
					if allFound[uid] {
						t.Errorf("shard %d: object %s (UID %s) appeared in multiple shards", shard, obj.GetName(), uid)
					}
					allFound[uid] = true
				}
			}

			// Every created object must appear in exactly one shard.
			for _, obj := range created {
				if !allFound[string(obj.GetUID())] {
					t.Errorf("object %s (UID %s) was not returned by any shard", obj.GetName(), obj.GetUID())
				}
			}
		})
	}
}

func TestShardedWatch(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, true)

	ctx, client, _, resources := setupSharding(t)

	for _, r := range resources {
		t.Run(r.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(client, "shard-watch-"+r.name, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			// Get initial resource version.
			list, err := r.list(ctx, ns.Name, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("initial list failed: %v", err)
			}
			rv := list.resourceVersion

			const numShards = 2

			// Start a watch per shard.
			watchers := make([]watch.Interface, numShards)
			for shard := range numShards {
				w, err := r.watch(ctx, ns.Name, metav1.ListOptions{
					ResourceVersion: rv,
					ShardSelector:   shardSelectorString(shard, numShards),
				})
				if err != nil {
					t.Fatalf("shard %d: failed to start watch: %v", shard, err)
				}
				defer w.Stop()
				watchers[shard] = w
			}

			// Create objects and track which shard should see each one.
			const numObjects = 10
			expectedShard := make(map[string]int) // UID -> shard index
			created := make([]metav1.Object, 0, numObjects)

			for range numObjects {
				obj, err := r.create(ctx, ns.Name, "shard-watch-")
				if err != nil {
					t.Fatalf("failed to create object: %v", err)
				}
				created = append(created, obj)
				for shard := range numShards {
					if objectInShard(string(obj.GetUID()), shard, numShards) {
						expectedShard[string(obj.GetUID())] = shard
						break
					}
				}
			}

			// Multiplex all watcher channels into a single channel for easy consumption.
			type shardEvent struct {
				shard     int
				eventType watch.EventType
				uid       string
			}
			eventCh := make(chan shardEvent, 100)
			for shard, w := range watchers {
				go func(shard int, w watch.Interface) {
					for evt := range w.ResultChan() {
						obj, err := meta.Accessor(evt.Object)
						if err != nil {
							continue
						}
						eventCh <- shardEvent{shard: shard, eventType: evt.Type, uid: string(obj.GetUID())}
					}
				}(shard, w)
			}

			// waitForEvents drains the multiplexed channel until count events of the
			// given type are collected, returning per-shard UID sets.
			waitForEvents := func(eventType watch.EventType, count int) []map[string]bool {
				t.Helper()
				perShard := make([]map[string]bool, numShards)
				for i := range perShard {
					perShard[i] = make(map[string]bool)
				}
				timeout := time.After(30 * time.Second)
				collected := 0
				for collected < count {
					select {
					case evt := <-eventCh:
						if evt.eventType != eventType {
							continue
						}
						perShard[evt.shard][evt.uid] = true
						collected++
					case <-timeout:
						t.Fatalf("timed out waiting for %s events: got %d/%d", eventType, collected, count)
					}
				}
				return perShard
			}

			// Collect ADDED events.
			added := waitForEvents(watch.Added, numObjects)
			verifyShardEvents(t, "ADDED", added, expectedShard, numShards)

			// Update each object by setting an annotation, then collect MODIFIED events.
			for _, obj := range created {
				obj.SetAnnotations(map[string]string{"updated": "true"})
				if err := r.update(ctx, obj); err != nil {
					t.Fatalf("failed to update object %s: %v", obj.GetName(), err)
				}
			}
			modified := waitForEvents(watch.Modified, numObjects)
			verifyShardEvents(t, "MODIFIED", modified, expectedShard, numShards)

			// Delete each object, then collect DELETED events.
			for _, obj := range created {
				if err := r.delete(ctx, ns.Name, obj.GetName()); err != nil {
					t.Fatalf("failed to delete object %s: %v", obj.GetName(), err)
				}
			}
			deleted := waitForEvents(watch.Deleted, numObjects)
			verifyShardEvents(t, "DELETED", deleted, expectedShard, numShards)
		})
	}
}

// verifyShardEvents checks that each UID was seen by exactly the expected shard.
func verifyShardEvents(t *testing.T, eventType string, perShard []map[string]bool, expectedShard map[string]int, numShards int) {
	t.Helper()
	for uid, expectedIdx := range expectedShard {
		if !perShard[expectedIdx][uid] {
			t.Errorf("%s: UID %s expected in shard %d but not received", eventType, uid, expectedIdx)
		}
		for other := range numShards {
			if other != expectedIdx && perShard[other][uid] {
				t.Errorf("%s: UID %s received in shard %d but expected only in shard %d", eventType, uid, other, expectedIdx)
			}
		}
	}
}

func TestShardedListFeatureGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, false)

	ctx, client, _, resources := setupSharding(t)

	for _, r := range resources {
		t.Run(r.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(client, "shard-disabled-"+r.name, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			// Create an object so the namespace is non-empty.
			if _, err := r.create(ctx, ns.Name, "test-"); err != nil {
				t.Fatalf("failed to create object: %v", err)
			}

			// List with a shard selector — when the gate is off, the selector should be
			// ignored and all objects returned (not sharded).
			list, err := r.list(ctx, ns.Name, metav1.ListOptions{
				ShardSelector: shardSelectorString(0, 2),
			})
			if err != nil {
				t.Fatalf("list failed: %v", err)
			}

			if list.shardInfo != nil {
				t.Errorf("expected shardInfo=nil when feature gate is disabled")
			}
			if len(list.items) != 1 {
				t.Errorf("expected 1 item (selector ignored), got %d", len(list.items))
			}
		})
	}
}

func TestShardedListComplete(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, true)

	ctx, client, _, resources := setupSharding(t)

	for _, r := range resources {
		t.Run(r.name, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(client, "shard-complete-"+r.name, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			// An empty/everything selector should return all objects without sharded=true.
			const numObjects = 5
			for range numObjects {
				if _, err := r.create(ctx, ns.Name, "complete-"); err != nil {
					t.Fatalf("failed to create object: %v", err)
				}
			}

			// A single shard covering the full range (shard 0 of 1) has empty start and end,
			// which means no selector string — just use a normal list.
			list, err := r.list(ctx, ns.Name, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("list failed: %v", err)
			}
			if list.shardInfo != nil {
				t.Errorf("expected shardInfo=nil for unsharded list")
			}
			if len(list.items) != numObjects {
				t.Errorf("expected %d items, got %d", numObjects, len(list.items))
			}
		})
	}
}

func TestShardedListByNamespace(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, true)

	ctx, client, _, resources := setupSharding(t)

	for _, r := range resources {
		t.Run(r.name, func(t *testing.T) {
			// Create multiple namespaces with one object each.
			const numNamespaces = 10
			type nsObj struct {
				namespace string
				uid       string
			}
			var objects []nsObj
			for i := range numNamespaces {
				nsName := fmt.Sprintf("shard-ns-%s-%d", r.name, i)
				ns := framework.CreateNamespaceOrDie(client, nsName, t)
				defer framework.DeleteNamespaceOrDie(client, ns, t)

				obj, err := r.create(ctx, nsName, "test-")
				if err != nil {
					t.Fatalf("failed to create object in %s: %v", nsName, err)
				}
				objects = append(objects, nsObj{namespace: nsName, uid: string(obj.GetUID())})
			}

			const numShards = 3
			allFound := make(map[string]bool) // UID -> found

			for shard := range numShards {
				selector := shardSelectorStringForField("object.metadata.namespace", shard, numShards)
				// List across all namespaces.
				list, err := r.list(ctx, metav1.NamespaceAll, metav1.ListOptions{
					ShardSelector: selector,
				})
				if err != nil {
					t.Fatalf("shard %d: failed to list: %v", shard, err)
				}

				if list.shardInfo == nil {
					t.Errorf("shard %d: expected shardInfo to be set", shard)
				}

				for _, obj := range list.items {
					if !valueInShard(obj.GetNamespace(), shard, numShards) {
						t.Errorf("shard %d: object %s/%s namespace hash should not be in this shard", shard, obj.GetNamespace(), obj.GetName())
					}
					allFound[string(obj.GetUID())] = true
				}
			}

			// Every object we created must appear in exactly one shard.
			for _, obj := range objects {
				if !allFound[obj.uid] {
					t.Errorf("object in namespace %s (UID %s) was not returned by any shard", obj.namespace, obj.uid)
				}
			}
		})
	}
}

// TestShardedListAllResources verifies that sharding is wired for every API
// resource. It follows the same discovery+etcd pattern used by the dryrun
// integration tests.
func TestShardedListAllResources(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, true)

	ctx, client, dynamicClient, resources := setupSharding(t, "--disable-admission-plugins=ServiceAccount")

	testNamespace := "shard-allresources"
	ns := framework.CreateNamespaceOrDie(client, testNamespace, t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	storageData := etcd.GetEtcdStorageDataForNamespace(testNamespace)

	_, serverResources, err := client.Discovery().ServerGroupsAndResources()
	if err != nil {
		t.Fatalf("failed to get ServerGroupsAndResources: %v", err)
	}

	for _, resourceToTest := range etcd.GetResources(t, serverResources) {
		mapping := resourceToTest.Mapping
		gvr := mapping.Resource

		testData, hasData := storageData[gvr]
		if !hasData {
			continue
		}

		// Check that this resource supports list.
		hasListVerb := false
		for _, discoveryGroup := range serverResources {
			for _, r := range discoveryGroup.APIResources {
				gv, _ := schema.ParseGroupVersion(discoveryGroup.GroupVersion)
				if gv.WithResource(r.Name) == gvr {
					for _, verb := range r.Verbs {
						if verb == "list" {
							hasListVerb = true
						}
					}
				}
			}
		}
		if !hasListVerb {
			continue
		}

		t.Run(gvr.String(), func(t *testing.T) {
			res, obj, err := etcd.JSONToUnstructured(testData.Stub, testNamespace, mapping, dynamicClient)
			if err != nil {
				t.Fatalf("failed to unmarshal stub: %v", err)
			}
			verifyObjectInOneShard(ctx, t, res, obj)
		})
	}

	for _, r := range resources {
		if r.gvr.Group == "" {
			continue
		}
		t.Run(r.gvr.String(), func(t *testing.T) {
			verifyObjectInOneShard(ctx, t, dynamicClient.Resource(r.gvr).Namespace(testNamespace), fixtures.NewVersionedNoxuInstance(testNamespace, "test", r.gvr.Version))
		})
	}
}

// verifyObjectInOneShard creates obj, lists with 2 shards and verifies the
// object appears in exactly one of them and that shardInfo is set.
func verifyObjectInOneShard(ctx context.Context, t *testing.T, res dynamic.ResourceInterface, obj *unstructured.Unstructured) {
	t.Helper()
	created, err := res.Create(ctx, obj, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create object: %v", err)
	}
	uid := string(created.GetUID())

	const numShards = 2
	foundInShard := -1
	for shard := range numShards {
		selector := shardSelectorString(shard, numShards)
		listResult, err := res.List(ctx, metav1.ListOptions{
			ShardSelector: selector,
		})
		if err != nil {
			t.Fatalf("shard %d: list failed: %v", shard, err)
		}
		if shardInfo := listResult.GetShardInfo(); shardInfo == nil || shardInfo.Selector != selector {
			t.Errorf("shard %d: expected shardInfo.selector=%q, got %v", shard, selector, shardInfo)
		}
		for _, item := range listResult.Items {
			if string(item.GetUID()) == uid {
				if foundInShard >= 0 {
					t.Errorf("object %s appeared in shard %d and %d", uid, foundInShard, shard)
				}
				foundInShard = shard
			}
		}
	}
	if foundInShard < 0 {
		t.Errorf("object %s not found in any shard", uid)
	}

	if err := res.Delete(context.TODO(), created.GetName(), *metav1.NewDeleteOptions(0)); err != nil {
		t.Logf("cleanup delete failed: %v", err)
	}
}
