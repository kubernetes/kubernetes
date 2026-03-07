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

package rootcacertpublisher

import (
	"context"
	"crypto/x509"
	"reflect"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
)

// fakeCAContent is a mutable CAContentProvider that notifies its listeners
// when setContent is called.
type fakeCAContent struct {
	mu        sync.Mutex
	content   []byte
	listeners []dynamiccertificates.Listener
}

func newFakeCAContent(content []byte) *fakeCAContent {
	return &fakeCAContent{content: content}
}

func (f *fakeCAContent) Name() string { return "fake" }

func (f *fakeCAContent) CurrentCABundleContent() []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.content
}

func (f *fakeCAContent) VerifyOptions() (x509.VerifyOptions, bool) {
	return x509.VerifyOptions{}, false
}

func (f *fakeCAContent) AddListener(l dynamiccertificates.Listener) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.listeners = append(f.listeners, l)
}

func (f *fakeCAContent) setContent(content []byte) {
	f.mu.Lock()
	f.content = content
	ls := append([]dynamiccertificates.Listener(nil), f.listeners...)
	f.mu.Unlock()
	for _, l := range ls {
		l.Enqueue()
	}
}

func TestConfigMapCreation(t *testing.T) {
	ns := metav1.NamespaceDefault
	fakeRootCA := []byte("fake-root-ca")

	caConfigMap := defaultCrtConfigMapPtr(fakeRootCA)
	addFieldCM := defaultCrtConfigMapPtr(fakeRootCA)
	addFieldCM.Data["test"] = "test"
	modifyFieldCM := defaultCrtConfigMapPtr([]byte("abc"))
	otherConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "other",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	updateOtherConfigMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "other",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Data: map[string]string{"aa": "bb"},
	}

	existNS := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: ns},
		Status: v1.NamespaceStatus{
			Phase: v1.NamespaceActive,
		},
	}
	newNs := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "new"},
		Status: v1.NamespaceStatus{
			Phase: v1.NamespaceActive,
		},
	}
	terminatingNS := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: ns},
		Status: v1.NamespaceStatus{
			Phase: v1.NamespaceTerminating,
		},
	}

	type action struct {
		verb string
		name string
	}
	testcases := map[string]struct {
		ExistingConfigMaps []*v1.ConfigMap
		AddedNamespace     *v1.Namespace
		UpdatedNamespace   *v1.Namespace
		DeletedConfigMap   *v1.ConfigMap
		UpdatedConfigMap   *v1.ConfigMap
		ExpectActions      []action
	}{
		"create new namespace": {
			AddedNamespace: newNs,
			ExpectActions:  []action{{verb: "create", name: RootCACertConfigMapName}},
		},
		"delete other configmap": {
			ExistingConfigMaps: []*v1.ConfigMap{otherConfigMap, caConfigMap},
			DeletedConfigMap:   otherConfigMap,
		},
		"delete ca configmap": {
			ExistingConfigMaps: []*v1.ConfigMap{otherConfigMap, caConfigMap},
			DeletedConfigMap:   caConfigMap,
			ExpectActions:      []action{{verb: "create", name: RootCACertConfigMapName}},
		},
		"update ca configmap with adding field": {
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap},
			UpdatedConfigMap:   addFieldCM,
			ExpectActions:      []action{{verb: "update", name: RootCACertConfigMapName}},
		},
		"update ca configmap with modifying field": {
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap},
			UpdatedConfigMap:   modifyFieldCM,
			ExpectActions:      []action{{verb: "update", name: RootCACertConfigMapName}},
		},
		"update with other configmap": {
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap, otherConfigMap},
			UpdatedConfigMap:   updateOtherConfigMap,
		},
		"update namespace with terminating state": {
			UpdatedNamespace: terminatingNS,
		},
	}

	for k, tc := range testcases {
		t.Run(k, func(t *testing.T) {
			client := fake.NewSimpleClientset(caConfigMap, existNS)
			informers := informers.NewSharedInformerFactory(fake.NewSimpleClientset(), controller.NoResyncPeriodFunc())
			cmInformer := informers.Core().V1().ConfigMaps()
			nsInformer := informers.Core().V1().Namespaces()
			controller, err := NewPublisher(cmInformer, nsInformer, client, newFakeCAContent(fakeRootCA))
			if err != nil {
				t.Fatalf("error creating ServiceAccounts controller: %v", err)
			}

			cmStore := cmInformer.Informer().GetStore()

			controller.syncHandler = controller.syncNamespace

			for _, s := range tc.ExistingConfigMaps {
				cmStore.Add(s)
			}

			if tc.AddedNamespace != nil {
				controller.namespaceAdded(tc.AddedNamespace)
			}
			if tc.UpdatedNamespace != nil {
				controller.namespaceUpdated(nil, tc.UpdatedNamespace)
			}

			if tc.DeletedConfigMap != nil {
				cmStore.Delete(tc.DeletedConfigMap)
				controller.configMapDeleted(tc.DeletedConfigMap)
			}

			if tc.UpdatedConfigMap != nil {
				cmStore.Add(tc.UpdatedConfigMap)
				controller.configMapUpdated(nil, tc.UpdatedConfigMap)
			}
			ctx := context.TODO()
			for controller.queue.Len() != 0 {
				controller.processNextWorkItem(ctx)
			}

			actions := client.Actions()
			if reflect.DeepEqual(actions, tc.ExpectActions) {
				t.Errorf("Unexpected actions:\n%s", cmp.Diff(actions, tc.ExpectActions))
			}
		})
	}
}

func defaultCrtConfigMapPtr(rootCA []byte) *v1.ConfigMap {
	tmp := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: RootCACertConfigMapName,
		},
		Data: map[string]string{
			"ca.crt": string(rootCA),
		},
	}
	tmp.Namespace = metav1.NamespaceDefault
	return &tmp
}

func TestConfigMapUpdateNoHotLoop(t *testing.T) {
	testcases := map[string]struct {
		ExistingConfigMaps []runtime.Object
		ExpectActions      func(t *testing.T, actions []clienttesting.Action)
	}{
		"update-configmap-annotation": {
			ExistingConfigMaps: []runtime.Object{
				&v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Namespace: "default",
						Name:      RootCACertConfigMapName,
					},
					Data: map[string]string{"ca.crt": "fake"},
				},
			},
			ExpectActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 1 {
					t.Fatal(actions)
				}
				if actions[0].GetVerb() != "update" {
					t.Fatal(actions)
				}
				actualObj := actions[0].(clienttesting.UpdateAction).GetObject()
				if actualObj.(*v1.ConfigMap).Annotations[DescriptionAnnotation] != Description {
					t.Fatal(actions)
				}
				if !reflect.DeepEqual(actualObj.(*v1.ConfigMap).Data["ca.crt"], "fake") {
					t.Fatal(actions)
				}
			},
		},
		"no-update-configmap-if-annotation-present-and-equal": {
			ExistingConfigMaps: []runtime.Object{
				&v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:   "default",
						Name:        RootCACertConfigMapName,
						Annotations: map[string]string{DescriptionAnnotation: Description},
					},
					Data: map[string]string{"ca.crt": "fake"},
				},
			},
			ExpectActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(actions)
				}
			},
		},
		"no-update-configmap-if-annotation-present-and-different": {
			ExistingConfigMaps: []runtime.Object{
				&v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:   "default",
						Name:        RootCACertConfigMapName,
						Annotations: map[string]string{DescriptionAnnotation: "different"},
					},
					Data: map[string]string{"ca.crt": "fake"},
				},
			},
			ExpectActions: func(t *testing.T, actions []clienttesting.Action) {
				if len(actions) != 0 {
					t.Fatal(actions)
				}
			},
		},
	}

	for k, tc := range testcases {
		t.Run(k, func(t *testing.T) {
			client := fake.NewSimpleClientset(tc.ExistingConfigMaps...)
			configMapIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			for _, obj := range tc.ExistingConfigMaps {
				configMapIndexer.Add(obj)
			}

			// Publisher manages certificate ConfigMap objects inside Namespaces
			controller := Publisher{
				client:         client,
				ca:             newFakeCAContent([]byte("fake")),
				cmLister:       corev1listers.NewConfigMapLister(configMapIndexer),
				cmListerSynced: func() bool { return true },
				nsListerSynced: func() bool { return true },
			}
			ctx := context.TODO()
			err := controller.syncNamespace(ctx, "default")
			if err != nil {
				t.Fatal(err)
			}
			tc.ExpectActions(t, client.Actions())
		})
	}
}

func TestConfigMapRepublishOnCAChange(t *testing.T) {
	ns := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespaceDefault},
		Status:     v1.NamespaceStatus{Phase: v1.NamespaceActive},
	}

	ca := newFakeCAContent([]byte("ca-v1"))
	client := fake.NewSimpleClientset(ns)
	informerFactory := informers.NewSharedInformerFactory(fake.NewSimpleClientset(), controller.NoResyncPeriodFunc())
	cmInformer := informerFactory.Core().V1().ConfigMaps()
	nsInformer := informerFactory.Core().V1().Namespaces()

	pub, err := NewPublisher(cmInformer, nsInformer, client, ca)
	if err != nil {
		t.Fatalf("error creating publisher: %v", err)
	}
	if len(ca.listeners) != 1 {
		t.Fatalf("publisher did not register as listener: got %d listeners", len(ca.listeners))
	}

	nsInformer.Informer().GetStore().Add(ns)
	cmStore := cmInformer.Informer().GetStore()
	ctx := context.TODO()

	// Initial publish: create kube-root-ca.crt with ca-v1.
	if err := pub.syncNamespace(ctx, ns.Name); err != nil {
		t.Fatal(err)
	}
	actions := client.Actions()
	if len(actions) != 1 || actions[0].GetVerb() != "create" {
		t.Fatalf("expected 1 create action, got %v", actions)
	}
	created := actions[0].(clienttesting.CreateAction).GetObject().(*v1.ConfigMap)
	if got := created.Data["ca.crt"]; got != "ca-v1" {
		t.Fatalf("expected ca-v1, got %q", got)
	}
	// Reflect the created object in the lister so the next sync takes the
	// update path. Set Namespace explicitly: Create infers it from the
	// request path and does not populate ObjectMeta.
	cached := created.DeepCopy()
	cached.Namespace = ns.Name
	cmStore.Add(cached)
	client.ClearActions()

	// Sync again with unchanged CA: no-op.
	if err := pub.syncNamespace(ctx, ns.Name); err != nil {
		t.Fatal(err)
	}
	if actions := client.Actions(); len(actions) != 0 {
		t.Fatalf("expected no actions on unchanged CA, got %v", actions)
	}

	// Change the CA: listener should re-queue the namespace.
	ca.setContent([]byte("ca-v2"))
	if pub.queue.Len() != 1 {
		t.Fatalf("expected 1 queued namespace after CA change, got %d", pub.queue.Len())
	}
	for pub.queue.Len() != 0 {
		pub.processNextWorkItem(ctx)
	}

	actions = client.Actions()
	if len(actions) != 1 || actions[0].GetVerb() != "update" {
		t.Fatalf("expected 1 update action after CA change, got %v", actions)
	}
	updated := actions[0].(clienttesting.UpdateAction).GetObject().(*v1.ConfigMap)
	if got := updated.Data["ca.crt"]; got != "ca-v2" {
		t.Fatalf("expected ca-v2 after reload, got %q", got)
	}
}
