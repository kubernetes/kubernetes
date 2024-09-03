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
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	certalpha1listers "k8s.io/client-go/listers/certificates/v1alpha1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller"
)

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
			controller, err := NewPublisher(cmInformer, nsInformer, client, fakeRootCA, "testSigner")
			if err != nil {
				t.Fatalf("error creating ServiceAccounts controller: %v", err)
			}

			cmStore := cmInformer.Informer().GetStore()

			controller.configMapsSyncer = controller.syncNamespace

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
			for controller.configMapQueue.Len() != 0 {
				controller.processNextWorkItem(ctx, controller.configMapQueue, controller.configMapsSyncer)
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
				rootCA:         []byte("fake"),
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

const testSigner = "testSigner"

func TestCTBCreation(t *testing.T) {
	checkCreatedTestSignerBundle := func(t *testing.T, actions []clienttesting.Action) {
		createAction := expectAction[clienttesting.CreateAction](t, actions, "create")

		ctb, ok := createAction.GetObject().(*certificatesv1alpha1.ClusterTrustBundle)
		if !ok {
			t.Fatalf("expected ClusterTrustBundle create, got %v", createAction.GetObject())
		}

		if ctb.Spec.SignerName != testSigner {
			t.Fatalf("expected signer name %q, got %q", testSigner, ctb.Spec.SignerName)
		}
	}

	checkUpdatedTestSignerBundle := func(t *testing.T, actions []clienttesting.Action) {
		updateAction := expectAction[clienttesting.UpdateAction](t, actions, "update")

		ctb, ok := updateAction.GetObject().(*certificatesv1alpha1.ClusterTrustBundle)
		if !ok {
			t.Fatalf("expected ClusterTrustBundle update, got %v", updateAction.GetObject())
		}

		if ctb.Spec.SignerName != testSigner {
			t.Fatalf("expected signer name %q, got %q", testSigner, ctb.Spec.SignerName)
		}

		if ctb.Spec.TrustBundle != "rootcapemdata" {
			t.Fatalf("expected trust bundle payload 'rootcapemdata', got %q", ctb.Spec.TrustBundle)
		}
	}

	checkDeletedTestSignerBundle := func(t *testing.T, actions []clienttesting.Action) {
		deleteAction := expectAction[clienttesting.DeleteCollectionAction](t, actions, "delete-collection")

		if fieldRestrictions := deleteAction.GetListRestrictions().Fields.String(); fieldRestrictions != "spec.signerName=testSigner" {
			t.Fatalf("expected field selector 'spec.signerName=testSigner', got %v", fieldRestrictions)
		}
	}

	for _, tt := range []struct {
		name          string
		existingCTBs  []runtime.Object
		expectActions func(t *testing.T, actions []clienttesting.Action)
		wantErr       bool
	}{
		{
			name:          "no CTBs exist",
			expectActions: checkCreatedTestSignerBundle,
		},
		{
			name: "no CTBs for the current signer exist",
			existingCTBs: []runtime.Object{
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "nosigner",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						TrustBundle: "somedatahere",
					},
				},
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "signer:one",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						SignerName:  "signer",
						TrustBundle: "signerdata",
					},
				},
			},
			expectActions: checkCreatedTestSignerBundle,
		},
		{
			name: "CTB for the signer exists with different content",
			existingCTBs: []runtime.Object{
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "testSigner:name",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						SignerName:  testSigner,
						TrustBundle: "olddata",
					},
				},
			},
			expectActions: checkUpdatedTestSignerBundle,
		},
		{
			name: "multiple CTBs for the signer",
			existingCTBs: []runtime.Object{
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "testSigner:name",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						SignerName:  testSigner,
						TrustBundle: "rootcapemdata",
					},
				},
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "testSigner:name2",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						SignerName:  testSigner,
						TrustBundle: "rootcapemdata",
					},
				},
			},
			expectActions: checkDeletedTestSignerBundle,
		},
		{
			name: "CTB at the correct state - noop",
			existingCTBs: []runtime.Object{
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "nosigner",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						TrustBundle: "somedatahere",
					},
				},
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "signer:one",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						SignerName:  "signer",
						TrustBundle: "signerdata",
					},
				},
				&certificatesv1alpha1.ClusterTrustBundle{
					ObjectMeta: metav1.ObjectMeta{
						Name: "testSigner:name",
					},
					Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
						SignerName:  testSigner,
						TrustBundle: "rootcapemdata",
					},
				},
			},
			expectActions: func(t *testing.T, actions []clienttesting.Action) {
				actions = filterOutListWatch(actions)
				if len(actions) != 0 {
					t.Fatalf("expected no actions, got %v", actions)
				}
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			testCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			fakeClient := fakeKubeClientSetWithCTBList(t, tt.existingCTBs...)

			// TODO: move the below into a construct function?
			ctbInformer := setupSignerNameFilteredCTBInformer(fakeClient, testSigner)
			go ctbInformer.Run(testCtx.Done())

			syncWait := make(chan struct{})
			go func() {
				for !ctbInformer.HasSynced() {
					select {
					case <-testCtx.Done():
						return
					case <-time.After(10 * time.Millisecond):
					}
				}
				syncWait <- struct{}{}
			}()

			select {
			case <-testCtx.Done():
				t.Fatalf("timed out waiting for informer to sync")
			case <-syncWait:
			}

			p := &Publisher{
				client:          fakeClient,
				ctbLister:       certalpha1listers.NewClusterTrustBundleLister(ctbInformer.GetIndexer()),
				ctbListerSynced: func() bool { return true },
				signerName:      testSigner,
				rootCA:          []byte("rootcapemdata"),

				trustBundleQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
					workqueue.DefaultTypedControllerRateLimiter[string](),
					workqueue.TypedRateLimitingQueueConfig[string]{
						Name: "test_root_ca_cert_publisher_cluster_trust_bundles",
					},
				),
			}

			if err := p.syncClusterTrustBundle(testCtx, "testSigner"); (err != nil) != tt.wantErr {
				t.Errorf("syncClusterTrustBundle() error = %v, wantErr %v", err, tt.wantErr)
			}

			tt.expectActions(t, fakeClient.Actions())
		})
	}
}

func fakeKubeClientSetWithCTBList(t *testing.T, ctbs ...runtime.Object) *fake.Clientset {
	fakeClient := fake.NewSimpleClientset(ctbs...)
	fakeClient.PrependReactor("list", "clustertrustbundles", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
		listAction, ok := action.(clienttesting.ListAction)
		if !ok {
			t.Fatalf("expected list action, got %v", action)
		}

		// fakeClient does not implement field selector, we have to do it manually
		listRestrictions := listAction.GetListRestrictions()
		if listRestrictions.Fields == nil || listRestrictions.Fields.String() != "spec.signerName=testSigner" {
			return false, nil, nil
		}

		retList := &certificatesv1alpha1.ClusterTrustBundleList{}
		for _, ctb := range ctbs {
			ctbObj, ok := ctb.(*certificatesv1alpha1.ClusterTrustBundle)
			if !ok {
				continue
			}
			if ctbObj.Spec.SignerName == testSigner {
				retList.Items = append(retList.Items, *ctbObj)
			}
		}

		return true, retList, nil
	})

	return fakeClient
}

func expectAction[A clienttesting.Action](t *testing.T, actions []clienttesting.Action, verb string) A {
	filteredActions := filterOutListWatch(actions)
	if len(filteredActions) != 1 {
		t.Fatalf("expected 1 action, got %v", filteredActions)
	}

	if filteredActions[0].GetVerb() != verb {
		t.Fatalf("expected action with verb %q, got %q", verb, filteredActions[0].GetVerb())
	}

	retAction, ok := filteredActions[0].(A)
	if !ok {
		t.Fatalf("expected %T action, got %v", *new(A), filteredActions[0])
	}

	return retAction
}

func filterOutListWatch(actions []clienttesting.Action) []clienttesting.Action {
	var filtered []clienttesting.Action
	for _, a := range actions {
		if a.Matches("list", "clustertrustbundles") || a.Matches("watch", "clustertrustbundles") {
			continue
		}
		filtered = append(filtered, a)
	}
	return filtered
}
