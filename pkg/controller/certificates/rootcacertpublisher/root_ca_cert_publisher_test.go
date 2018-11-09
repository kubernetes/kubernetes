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
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
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
		ExistingNamespace  *v1.Namespace
		ExistingConfigMaps []*v1.ConfigMap
		AddedNamespace     *v1.Namespace
		UpdatedNamespace   *v1.Namespace
		DeletedConfigMap   *v1.ConfigMap
		UpdatedConfigMap   []*v1.ConfigMap
		ExpectActions      []action
	}{
		"create new namesapce": {
			AddedNamespace: newNs,
			ExpectActions:  []action{{verb: "create", name: RootCACertCofigMapName}},
		},

		"delete other configmap": {
			ExistingNamespace:  existNS,
			ExistingConfigMaps: []*v1.ConfigMap{otherConfigMap, caConfigMap},
			DeletedConfigMap:   otherConfigMap,
		},
		"delete ca configmap": {
			ExistingNamespace:  existNS,
			ExistingConfigMaps: []*v1.ConfigMap{otherConfigMap, caConfigMap},
			DeletedConfigMap:   caConfigMap,
			ExpectActions:      []action{{verb: "create", name: RootCACertCofigMapName}},
		},
		"update ca configmap with adding field": {
			ExistingNamespace:  existNS,
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap},
			UpdatedConfigMap:   []*v1.ConfigMap{caConfigMap, addFieldCM},
			ExpectActions:      []action{{verb: "update", name: RootCACertCofigMapName}},
		},
		"update ca configmap with modifying field": {
			ExistingNamespace:  existNS,
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap},
			UpdatedConfigMap:   []*v1.ConfigMap{caConfigMap, modifyFieldCM},
			ExpectActions:      []action{{verb: "update", name: RootCACertCofigMapName}},
		},
		"update with other configmap": {
			ExistingNamespace:  existNS,
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap, otherConfigMap},
			UpdatedConfigMap:   []*v1.ConfigMap{otherConfigMap, updateOtherConfigMap},
		},
		"update namespace with terminating state": {
			ExistingNamespace: existNS,
			UpdatedNamespace:  terminatingNS,
		},
	}

	for k, tc := range testcases {
		client := fake.NewSimpleClientset(caConfigMap, existNS)
		informers := informers.NewSharedInformerFactory(fake.NewSimpleClientset(), controller.NoResyncPeriodFunc())
		cmInformer := informers.Core().V1().ConfigMaps()
		nsInformer := informers.Core().V1().Namespaces()
		controller, err := NewPublisher(cmInformer, nsInformer, client, fakeRootCA)
		if err != nil {
			t.Fatalf("error creating ServiceAccounts controller: %v", err)
		}
		controller.cmListerSynced = alwaysReady
		controller.nsListerSynced = alwaysReady

		cmStore := cmInformer.Informer().GetStore()
		nsStore := nsInformer.Informer().GetStore()

		syncCalls := make(chan struct{})
		controller.syncHandler = func(key string) error {
			err := controller.syncNamespace(key)
			if err != nil {
				t.Logf("%s: %v", k, err)
			}
			syncCalls <- struct{}{}
			return err
		}
		stopCh := make(chan struct{})
		defer close(stopCh)
		go controller.Run(1, stopCh)

		if tc.ExistingNamespace != nil {
			nsStore.Add(tc.ExistingNamespace)
		}
		for _, s := range tc.ExistingConfigMaps {
			cmStore.Add(s)
		}

		if tc.AddedNamespace != nil {
			nsStore.Add(tc.AddedNamespace)
			controller.namespaceAdded(tc.AddedNamespace)
		}
		if tc.UpdatedNamespace != nil {
			controller.namespaceUpdated(tc.ExistingNamespace, tc.UpdatedNamespace)
		}

		if tc.DeletedConfigMap != nil {
			cmStore.Delete(tc.DeletedConfigMap)
			controller.configMapDeleted(tc.DeletedConfigMap)
		}

		if tc.UpdatedConfigMap != nil {
			old := tc.UpdatedConfigMap[0]
			new := tc.UpdatedConfigMap[1]
			controller.configMapUpdated(old, new)
		}

		// wait to be called
		select {
		case <-syncCalls:
		case <-time.After(5 * time.Second):
		}

		actions := client.Actions()
		if len(tc.ExpectActions) != len(actions) {
			t.Errorf("%s: Expected to create configmap %#v. Actual actions were: %#v", k, tc.ExpectActions, actions)
			continue
		}
		for i, expectAction := range tc.ExpectActions {
			action := actions[i]
			if !action.Matches(expectAction.verb, "configmaps") {
				t.Errorf("%s: Unexpected action %s", k, action)
				break
			}
			cm := action.(core.CreateAction).GetObject().(*v1.ConfigMap)
			if cm.Name != expectAction.name {
				t.Errorf("%s: Expected %s to be %s, got %s be %s", k, expectAction.name, expectAction.verb, cm.Name, action.GetVerb())
			}
		}
	}
}

var alwaysReady = func() bool { return true }

func defaultCrtConfigMapPtr(rootCA []byte) *v1.ConfigMap {
	tmp := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: RootCACertCofigMapName,
		},
		Data: map[string]string{
			"ca.crt": string(rootCA),
		},
	}
	tmp.Namespace = metav1.NamespaceDefault
	return &tmp
}
