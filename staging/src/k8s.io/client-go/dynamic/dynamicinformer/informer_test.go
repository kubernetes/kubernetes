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

package dynamicinformer_test

import (
	"context"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/tools/cache"
)

type triggerFunc func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, testObject *unstructured.Unstructured) *unstructured.Unstructured

func triggerFactory(t *testing.T) triggerFunc {
	return func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, _ *unstructured.Unstructured) *unstructured.Unstructured {
		testObject := newUnstructured("apps/v1", "Deployment", "ns-foo", "name-foo")
		createdObj, err := fakeClient.Resource(gvr).Namespace(ns).Create(context.TODO(), testObject, metav1.CreateOptions{})
		if err != nil {
			t.Error(err)
		}
		return createdObj
	}
}

func handler(rcvCh chan<- *unstructured.Unstructured) *cache.ResourceEventHandlerFuncs {
	return &cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			rcvCh <- obj.(*unstructured.Unstructured)
		},
	}
}

func TestFilteredDynamicSharedInformerFactory(t *testing.T) {
	scenarios := []struct {
		name        string
		existingObj *unstructured.Unstructured
		gvr         schema.GroupVersionResource
		informNS    string
		ns          string
		trigger     func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, testObject *unstructured.Unstructured) *unstructured.Unstructured
		handler     func(rcvCh chan<- *unstructured.Unstructured) *cache.ResourceEventHandlerFuncs
	}{
		// scenario 1
		{
			name:     "scenario 1: test adding an object in different namespace should not trigger AddFunc",
			informNS: "ns-bar",
			ns:       "ns-foo",
			gvr:      schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
			trigger:  triggerFactory(t),
			handler:  handler,
		},
		// scenario 2
		{
			name:     "scenario 2: test adding an object should trigger AddFunc",
			informNS: "ns-foo",
			ns:       "ns-foo",
			gvr:      schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"},
			trigger:  triggerFactory(t),
			handler:  handler,
		},
	}

	for _, ts := range scenarios {
		t.Run(ts.name, func(t *testing.T) {
			// test data
			timeout := time.Duration(3 * time.Second)
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			scheme := runtime.NewScheme()
			informerReciveObjectCh := make(chan *unstructured.Unstructured, 1)
			objs := []runtime.Object{}
			if ts.existingObj != nil {
				objs = append(objs, ts.existingObj)
			}
			// don't adjust the scheme to include deploymentlist. This is testing whether an informer can be created against using
			// a client that doesn't have a type registered in the scheme.
			gvrToListKind := map[schema.GroupVersionResource]string{
				{Group: "apps", Version: "v1", Resource: "deployments"}: "DeploymentList",
			}
			fakeClient := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, gvrToListKind, objs...)
			target := dynamicinformer.NewFilteredDynamicSharedInformerFactory(fakeClient, 0, ts.informNS, nil)

			// act
			informerListerForGvr := target.ForResource(ts.gvr)
			informerListerForGvr.Informer().AddEventHandler(ts.handler(informerReciveObjectCh))
			target.Start(ctx.Done())
			if synced := target.WaitForCacheSync(ctx.Done()); !synced[ts.gvr] {
				t.Errorf("informer for %s hasn't synced", ts.gvr)
			}

			testObject := ts.trigger(ts.gvr, ts.ns, fakeClient, ts.existingObj)
			select {
			case objFromInformer := <-informerReciveObjectCh:
				if ts.ns != ts.informNS {
					t.Errorf("informer received an object for namespace %s when watching namespace %s", ts.ns, ts.informNS)
				}
				if !equality.Semantic.DeepEqual(testObject, objFromInformer) {
					t.Fatalf("%v", diff.ObjectDiff(testObject, objFromInformer))
				}
			case <-ctx.Done():
				if ts.ns == ts.informNS {
					t.Errorf("tested informer haven't received an object, waited %v", timeout)
				}
			}
		})
	}

}

func TestDynamicSharedInformerFactory(t *testing.T) {
	scenarios := []struct {
		name        string
		existingObj *unstructured.Unstructured
		gvr         schema.GroupVersionResource
		ns          string
		trigger     func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, testObject *unstructured.Unstructured) *unstructured.Unstructured
		handler     func(rcvCh chan<- *unstructured.Unstructured) *cache.ResourceEventHandlerFuncs
	}{
		// scenario 1
		{
			name: "scenario 1: test if adding an object triggers AddFunc",
			ns:   "ns-foo",
			gvr:  schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
			trigger: func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, _ *unstructured.Unstructured) *unstructured.Unstructured {
				testObject := newUnstructured("extensions/v1beta1", "Deployment", "ns-foo", "name-foo")
				createdObj, err := fakeClient.Resource(gvr).Namespace(ns).Create(context.TODO(), testObject, metav1.CreateOptions{})
				if err != nil {
					t.Error(err)
				}
				return createdObj
			},
			handler: func(rcvCh chan<- *unstructured.Unstructured) *cache.ResourceEventHandlerFuncs {
				return &cache.ResourceEventHandlerFuncs{
					AddFunc: func(obj interface{}) {
						rcvCh <- obj.(*unstructured.Unstructured)
					},
				}
			},
		},

		// scenario 2
		{
			name:        "scenario 2: tests if updating an object triggers UpdateFunc",
			ns:          "ns-foo",
			gvr:         schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
			existingObj: newUnstructured("extensions/v1beta1", "Deployment", "ns-foo", "name-foo"),
			trigger: func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, testObject *unstructured.Unstructured) *unstructured.Unstructured {
				testObject.Object["spec"] = "updatedName"
				updatedObj, err := fakeClient.Resource(gvr).Namespace(ns).Update(context.TODO(), testObject, metav1.UpdateOptions{})
				if err != nil {
					t.Error(err)
				}
				return updatedObj
			},
			handler: func(rcvCh chan<- *unstructured.Unstructured) *cache.ResourceEventHandlerFuncs {
				return &cache.ResourceEventHandlerFuncs{
					UpdateFunc: func(old, updated interface{}) {
						rcvCh <- updated.(*unstructured.Unstructured)
					},
				}
			},
		},

		// scenario 3
		{
			name:        "scenario 3: test if deleting an object triggers DeleteFunc",
			ns:          "ns-foo",
			gvr:         schema.GroupVersionResource{Group: "extensions", Version: "v1beta1", Resource: "deployments"},
			existingObj: newUnstructured("extensions/v1beta1", "Deployment", "ns-foo", "name-foo"),
			trigger: func(gvr schema.GroupVersionResource, ns string, fakeClient *fake.FakeDynamicClient, testObject *unstructured.Unstructured) *unstructured.Unstructured {
				err := fakeClient.Resource(gvr).Namespace(ns).Delete(context.TODO(), testObject.GetName(), metav1.DeleteOptions{})
				if err != nil {
					t.Error(err)
				}
				return testObject
			},
			handler: func(rcvCh chan<- *unstructured.Unstructured) *cache.ResourceEventHandlerFuncs {
				return &cache.ResourceEventHandlerFuncs{
					DeleteFunc: func(obj interface{}) {
						rcvCh <- obj.(*unstructured.Unstructured)
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
			scheme := runtime.NewScheme()
			informerReciveObjectCh := make(chan *unstructured.Unstructured, 1)
			objs := []runtime.Object{}
			if ts.existingObj != nil {
				objs = append(objs, ts.existingObj)
			}
			// don't adjust the scheme to include deploymentlist. This is testing whether an informer can be created against using
			// a client that doesn't have a type registered in the scheme.
			gvrToListKind := map[schema.GroupVersionResource]string{
				{Group: "extensions", Version: "v1beta1", Resource: "deployments"}: "DeploymentList",
			}
			fakeClient := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, gvrToListKind, objs...)
			target := dynamicinformer.NewDynamicSharedInformerFactory(fakeClient, 0)

			// act
			informerListerForGvr := target.ForResource(ts.gvr)
			informerListerForGvr.Informer().AddEventHandler(ts.handler(informerReciveObjectCh))
			target.Start(ctx.Done())
			if synced := target.WaitForCacheSync(ctx.Done()); !synced[ts.gvr] {
				t.Errorf("informer for %s hasn't synced", ts.gvr)
			}

			testObject := ts.trigger(ts.gvr, ts.ns, fakeClient, ts.existingObj)
			select {
			case objFromInformer := <-informerReciveObjectCh:
				if !equality.Semantic.DeepEqual(testObject, objFromInformer) {
					t.Fatalf("%v", diff.ObjectDiff(testObject, objFromInformer))
				}
			case <-ctx.Done():
				t.Errorf("tested informer haven't received an object, waited %v", timeout)
			}
		})
	}
}

func newUnstructured(apiVersion, kind, namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": apiVersion,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"spec": name,
		},
	}
}
