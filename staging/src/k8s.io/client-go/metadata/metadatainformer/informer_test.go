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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	clientgotesting "k8s.io/client-go/testing"
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
	gvr := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	const numObjects = 12
	objs := make([]runtime.Object, 0, numObjects)
	for i := range numObjects {
		obj := newPartialObjectMetadata("apps/v1", "Deployment", "ns-foo", fmt.Sprintf("deploy-%02d", i))
		obj.UID = types.UID(fmt.Sprintf("%d-uid-deploy", i))
		objs = append(objs, obj)
	}

	const totalShards = 3
	seen := make(map[string]int, numObjects)
	for shard := range totalShards {
		func() {
			sel, err := sharding.NewShardRangeSelector("object.metadata.uid", shard, totalShards)
			if err != nil {
				t.Fatalf("NewShardRangeSelector: %v", err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			scheme := fake.NewTestScheme()
			if err := metav1.AddMetaToScheme(scheme); err != nil {
				t.Fatalf("AddMetaToScheme: %v", err)
			}
			fakeClient := fake.NewSimpleMetadataClient(scheme, objs...)

			factory := NewSharedInformerFactoryWithOptions(fakeClient, 0, WithShardSelector(sel))
			defer func() {
				cancel()
				factory.Shutdown()
			}()
			informer := factory.ForResource(gvr)

			addedCh := make(chan *metav1.PartialObjectMetadata, 2)
			_, err = informer.Informer().AddEventHandler(&cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					metaObj := obj.(*metav1.PartialObjectMetadata)
					if metaObj.Name == "dynamic-in-shard" || metaObj.Name == "dynamic-out-of-shard" {
						addedCh <- metaObj
					}
				},
			})
			if err != nil {
				t.Fatalf("AddEventHandler: %v", err)
			}

			factory.Start(ctx.Done())
			if synced := factory.WaitForCacheSync(ctx.Done()); !synced[gvr] {
				t.Fatalf("shard %d informer for %s failed to sync", shard, gvr)
			}
			var gotWatchSelector string
			if err := wait.PollUntilContextCancel(ctx, 10*time.Millisecond, true, func(_ context.Context) (bool, error) {
				for _, action := range fakeClient.Actions() {
					if watchAction, ok := action.(clientgotesting.WatchActionImpl); ok {
						gotWatchSelector = watchAction.ListOptions.ShardSelector
						return true, nil
					}
				}
				return false, nil
			}); err != nil {
				t.Fatalf("shard %d informer failed to start watch: %v", shard, err)
			}
			if gotWatchSelector != sel.String() {
				t.Errorf("shard %d watch action ShardSelector = %q, want %q", shard, gotWatchSelector, sel.String())
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
				if prev, dup := seen[uid]; dup {
					t.Errorf("object %s (uid %s) seen in both shard %d and %d", metaObj.Name, uid, prev, shard)
				}
				seen[uid] = shard
			}

			// Verify watch filtering: find one UID inside this shard and one UID outside it.
			var inShardUID, outOfShardUID types.UID
			for i := range 100 {
				candidate := types.UID(fmt.Sprintf("%d-dynamic-uid-deploy", i))
				probe := &metav1.PartialObjectMetadata{ObjectMeta: metav1.ObjectMeta{UID: candidate}}
				matched, err := sel.Matches(probe)
				if err != nil {
					t.Fatalf("Matches: %v", err)
				}
				if matched && inShardUID == "" {
					inShardUID = candidate
				} else if !matched && outOfShardUID == "" {
					outOfShardUID = candidate
				}
				if inShardUID != "" && outOfShardUID != "" {
					break
				}
			}

			outObj := newPartialObjectMetadata("apps/v1", "Deployment", "ns-foo", "dynamic-out-of-shard")
			outObj.UID = outOfShardUID
			if _, err := fakeClient.Resource(gvr).Namespace("ns-foo").(fake.MetadataClient).CreateFake(outObj, metav1.CreateOptions{}); err != nil {
				t.Fatalf("CreateFake out-of-shard: %v", err)
			}

			inObj := newPartialObjectMetadata("apps/v1", "Deployment", "ns-foo", "dynamic-in-shard")
			inObj.UID = inShardUID
			if _, err := fakeClient.Resource(gvr).Namespace("ns-foo").(fake.MetadataClient).CreateFake(inObj, metav1.CreateOptions{}); err != nil {
				t.Fatalf("CreateFake in-shard: %v", err)
			}

			select {
			case got := <-addedCh:
				if got.Name != "dynamic-in-shard" {
					t.Errorf("shard %d expected dynamic-in-shard watch event, got %s", shard, got.Name)
				}
			case <-ctx.Done():
				t.Errorf("shard %d timed out waiting for dynamic-in-shard watch event", shard)
			}
			if _, err := informer.Lister().ByNamespace("ns-foo").Get("dynamic-out-of-shard"); !apierrors.IsNotFound(err) {
				t.Errorf("shard %d expected dynamic-out-of-shard to be absent from store, got err: %v", shard, err)
			}
		}()
	}

	if len(seen) != numObjects {
		t.Errorf("expected %d total objects across shards, got %d", numObjects, len(seen))
	}
}
