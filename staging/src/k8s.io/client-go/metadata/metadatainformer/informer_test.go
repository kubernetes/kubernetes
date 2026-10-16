/*
Copyright 2018 The Kubernetes Authors.

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

package metadatainformer

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/fake"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

func init() {
	klog.InitFlags(flag.CommandLine)
	flag.CommandLine.Lookup("v").Value.Set("5")
	flag.CommandLine.Lookup("alsologtostderr").Value.Set("true")
}

func TestWatchListSemanticsSimple(t *testing.T) {
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, true)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if _, ok := req.URL.Query()["watch"]; !ok {
			t.Errorf("expected a watch request, params: %v", req.URL.Query())
			http.Error(w, fmt.Errorf("unexpected request").Error(), http.StatusInternalServerError)
			return
		}

		obj := &metav1.PartialObjectMetadata{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "meta.k8s.io/v1",
				Kind:       "PartialObjectMetadata",
			},
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					metav1.InitialEventsAnnotationKey: "true",
				},
			},
		}
		rawObj, err := json.Marshal(obj)
		if err != nil {
			t.Errorf("failed to marshal rawObj: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		watchEvent := &metav1.WatchEvent{
			Type:   string(watch.Bookmark),
			Object: runtime.RawExtension{Raw: rawObj},
		}
		rawRsp, err := json.Marshal(watchEvent)
		if err != nil {
			t.Errorf("failed to marshal watchEvent: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, err = w.Write(rawRsp)
		if err != nil {
			t.Fatalf("failed to write response: %v", err)
		}
	}))
	defer server.Close()

	cfg := &rest.Config{Host: server.URL}
	client, err := metadata.NewForConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}

	factory := NewFilteredSharedInformerFactory(client, 0, "ns", nil)
	target := factory.ForResource(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	factory.Start(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), target.Informer().HasSynced) {
		t.Fatalf("failed to wait for caches to sync")
	}
}

func TestUnSupportWatchListSemantics(t *testing.T) {
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, true)
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add appsv1 to scheme: %v", err)
	}
	// The fake client doesn’t support WatchList semantics,
	// so we don’t need to prepare a response.
	fakeClient := fake.NewSimpleMetadataClient(scheme)
	factory := NewFilteredSharedInformerFactory(fakeClient, 0, "ns", nil)
	target := factory.ForResource(schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	factory.Start(ctx.Done())

	if !cache.WaitForCacheSync(ctx.Done(), target.Informer().HasSynced) {
		t.Fatalf("failed to wait for caches to sync")
	}
}

func TestMetadataSharedInformerFactory(t *testing.T) {
	scenarios := []struct {
		name        string
		existingObj *metav1.PartialObjectMetadata
		gvr         schema.GroupVersionResource
		ns          string
		trigger     func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeMetadataClient, testObject *metav1.PartialObjectMetadata) *metav1.PartialObjectMetadata
		handler     func(rcvCh chan<- *metav1.PartialObjectMetadata) *cache.ResourceEventHandlerFuncs
	}{
		// scenario 1
		{
			name: "scenario 1: test if adding an object triggers AddFunc",
			ns:   "ns-foo",
			gvr:  schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
			trigger: func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeMetadataClient, _ *metav1.PartialObjectMetadata) *metav1.PartialObjectMetadata {
				testObject := newPartialObjectMetadata("extensions/v1beta1", "Deployment", "ns-foo", "name-foo")
				createdObj, err := fakeClient.Resource(gvr).Namespace(ns).(fake.MetadataClient).CreateFake(testObject, metav1.CreateOptions{})
				if err != nil {
					t.Error(err)
				}
				return createdObj
			},
			handler: func(rcvCh chan<- *metav1.PartialObjectMetadata) *cache.ResourceEventHandlerFuncs {
				return &cache.ResourceEventHandlerFuncs{
					AddFunc: func(obj interface{}) {
						rcvCh <- obj.(*metav1.PartialObjectMetadata)
					},
				}
			},
		},

		// scenario 2
		{
			name:        "scenario 2: tests if updating an object triggers UpdateFunc",
			ns:          "ns-foo",
			gvr:         schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
			existingObj: newPartialObjectMetadata("extensions/v1beta1", "Deployment", "ns-foo", "name-foo"),
			trigger: func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeMetadataClient, testObject *metav1.PartialObjectMetadata) *metav1.PartialObjectMetadata {
				if testObject.Annotations == nil {
					testObject.Annotations = make(map[string]string)
				}
				testObject.Annotations["test"] = "updatedName"
				updatedObj, err := fakeClient.Resource(gvr).Namespace(ns).(fake.MetadataClient).UpdateFake(testObject, metav1.UpdateOptions{})
				if err != nil {
					t.Error(err)
				}
				return updatedObj
			},
			handler: func(rcvCh chan<- *metav1.PartialObjectMetadata) *cache.ResourceEventHandlerFuncs {
				return &cache.ResourceEventHandlerFuncs{
					UpdateFunc: func(old, updated interface{}) {
						rcvCh <- updated.(*metav1.PartialObjectMetadata)
					},
				}
			},
		},

		// scenario 3
		{
			name:        "scenario 3: test if deleting an object triggers DeleteFunc",
			ns:          "ns-foo",
			gvr:         schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
			existingObj: newPartialObjectMetadata("extensions/v1beta1", "Deployment", "ns-foo", "name-foo"),
			trigger: func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeMetadataClient, testObject *metav1.PartialObjectMetadata) *metav1.PartialObjectMetadata {
				err := fakeClient.Resource(gvr).Namespace(ns).Delete(context.TODO(), testObject.GetName(), metav1.DeleteOptions{})
				if err != nil {
					t.Error(err)
				}
				return testObject
			},
			handler: func(rcvCh chan<- *metav1.PartialObjectMetadata) *cache.ResourceEventHandlerFuncs {
				return &cache.ResourceEventHandlerFuncs{
					DeleteFunc: func(obj interface{}) {
						rcvCh <- obj.(*metav1.PartialObjectMetadata)
					},
				}
			},
		},
	}

	for _, ts := range scenarios {
		t.Run(ts.name, func(t *testing.T) {
			// test data
			timeout := time.Duration(3 * time.Second)
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			scheme := fake.NewTestScheme()
			metav1.AddMetaToScheme(scheme)
			informerReciveObjectCh := make(chan *metav1.PartialObjectMetadata, 1)
			objs := []runtime.Object{}
			if ts.existingObj != nil {
				objs = append(objs, ts.existingObj)
			}
			fakeClient := fake.NewSimpleMetadataClient(scheme, objs...)
			target := NewSharedInformerFactory(fakeClient, 0)

			// act
			informerListerForGvr := target.ForResource(ts.gvr)
			informerListerForGvr.Informer().AddEventHandler(ts.handler(informerReciveObjectCh))
			target.Start(ctx.Done())
			if synced := target.WaitForCacheSync(ctx.Done()); !synced[ts.gvr] {
				t.Fatalf("informer for %s hasn't synced", ts.gvr)
			}

			testObject := ts.trigger(ts.gvr, ts.ns, fakeClient, ts.existingObj)
			select {
			case objFromInformer := <-informerReciveObjectCh:
				if !equality.Semantic.DeepEqual(testObject, objFromInformer) {
					t.Fatalf("%v", cmp.Diff(testObject, objFromInformer))
				}
			case <-ctx.Done():
				t.Errorf("tested informer haven't received an object, waited %v", timeout)
			}
		})
	}
}

func newPartialObjectMetadata(apiVersion, kind, namespace, name string) *metav1.PartialObjectMetadata {
	return &metav1.PartialObjectMetadata{
		TypeMeta: metav1.TypeMeta{
			APIVersion: apiVersion,
			Kind:       kind,
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
	}
}

func TestMetadataSharedInformerFactoryWithShardSelector(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	gvr := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	const initialObjects = 12
	objs := make([]runtime.Object, 0, initialObjects)
	for i := range initialObjects {
		obj := newPartialObjectMetadata("apps/v1", "Deployment", "ns-foo", fmt.Sprintf("deploy-%02d", i))
		obj.UID = types.UID(fmt.Sprintf("%d-uid-deploy", i))
		objs = append(objs, obj)
	}

	scheme := fake.NewTestScheme()
	if err := metav1.AddMetaToScheme(scheme); err != nil {
		t.Fatalf("AddMetaToScheme: %v", err)
	}
	fakeClient := fake.NewSimpleMetadataClient(scheme, objs...)

	const totalShards = 3
	factories := make([]SharedInformerFactory, totalShards)
	selectors := make([]sharding.Selector, totalShards)
	seenInitial := make(map[string]int, initialObjects)

	for shard := range totalShards {
		sel, err := sharding.NewShardRangeSelector("object.metadata.uid", shard, totalShards)
		if err != nil {
			t.Fatalf("NewShardRangeSelector: %v", err)
		}
		selectors[shard] = sel

		factories[shard] = NewSharedInformerFactoryWithOptions(fakeClient, 0, WithShardSelector(sel))
		informer := factories[shard].ForResource(gvr)
		_ = informer.Informer()
		factories[shard].Start(ctx.Done())
		t.Cleanup(factories[shard].Shutdown)
		if synced := factories[shard].WaitForCacheSync(ctx.Done()); !synced[gvr] {
			t.Fatalf("shard %d informer for %s failed to sync", shard, gvr)
		}

		items, err := informer.Lister().ByNamespace("ns-foo").List(labels.Everything())
		if err != nil {
			t.Fatalf("shard %d lister failed: %v", shard, err)
		}
		for _, item := range items {
			metaObj := item.(*metav1.PartialObjectMetadata)
			uid := string(metaObj.UID)
			if matched, err := sel.Matches(metaObj); err != nil || !matched {
				t.Errorf("shard %d cached out-of-shard object %s (uid %s, err %v)", shard, metaObj.Name, uid, err)
			}
			if prev, dup := seenInitial[uid]; dup {
				t.Errorf("object %s (uid %s) seen in both shard %d and %d", metaObj.Name, uid, prev, shard)
			}
			seenInitial[uid] = shard
		}
	}
	if len(seenInitial) != initialObjects {
		t.Errorf("expected %d total initial objects across shards, got %d", initialObjects, len(seenInitial))
	}

	const watchObjects = 6
	for offset := range watchObjects {
		i := initialObjects + offset
		obj := newPartialObjectMetadata("apps/v1", "Deployment", "ns-foo", fmt.Sprintf("deploy-%02d", i))
		obj.UID = types.UID(fmt.Sprintf("%d-uid-deploy", i))
		if _, err := fakeClient.Resource(gvr).Namespace("ns-foo").(fake.MetadataClient).CreateFake(obj, metav1.CreateOptions{}); err != nil {
			t.Fatalf("CreateFake: %v", err)
		}
	}

	if err := wait.PollUntilContextCancel(ctx, 10*time.Millisecond, true, func(_ context.Context) (bool, error) {
		total := 0
		for shard := range totalShards {
			items, err := factories[shard].ForResource(gvr).Lister().ByNamespace("ns-foo").List(labels.Everything())
			if err != nil {
				return false, err
			}
			for _, item := range items {
				metaObj := item.(*metav1.PartialObjectMetadata)
				if matched, err := selectors[shard].Matches(metaObj); err != nil || !matched {
					return false, fmt.Errorf("shard %d cached out-of-shard object %s", shard, metaObj.Name)
				}
			}
			total += len(items)
		}
		return total == initialObjects+watchObjects, nil
	}); err != nil {
		t.Fatalf("failed waiting for sharded watch events across informers: %v", err)
	}
}
