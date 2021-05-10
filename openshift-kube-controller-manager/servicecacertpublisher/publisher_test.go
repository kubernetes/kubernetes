package servicecacertpublisher

import (
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller"
)

func TestConfigMapCreation(t *testing.T) {
	ns := metav1.NamespaceDefault

	caConfigMap := defaultCrtConfigMapPtr()
	addAnnotationCM := defaultCrtConfigMapPtr()
	addAnnotationCM.Annotations["test"] = "test"
	modifyAnnotationCM := defaultCrtConfigMapPtr()
	modifyAnnotationCM.Annotations["service.beta.openshift.io/inject-cabundle"] = "no"
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
			Annotations:     map[string]string{"test": "true"},
		},
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
			ExpectActions:  []action{{verb: "create", name: ServiceCACertConfigMapName}},
		},
		"delete other configmap": {
			ExistingConfigMaps: []*v1.ConfigMap{otherConfigMap, caConfigMap},
			DeletedConfigMap:   otherConfigMap,
		},
		"delete ca configmap": {
			ExistingConfigMaps: []*v1.ConfigMap{otherConfigMap, caConfigMap},
			DeletedConfigMap:   caConfigMap,
			ExpectActions:      []action{{verb: "create", name: ServiceCACertConfigMapName}},
		},
		"update ca configmap with adding annotation": {
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap},
			UpdatedConfigMap:   addAnnotationCM,
			ExpectActions:      []action{{verb: "update", name: ServiceCACertConfigMapName}},
		},
		"update ca configmap with modifying annotation": {
			ExistingConfigMaps: []*v1.ConfigMap{caConfigMap},
			UpdatedConfigMap:   modifyAnnotationCM,
			ExpectActions:      []action{{verb: "update", name: ServiceCACertConfigMapName}},
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
			controller, err := NewPublisher(cmInformer, nsInformer, client)
			if err != nil {
				t.Fatalf("error creating controller: %v", err)
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

			for controller.queue.Len() != 0 {
				controller.processNextWorkItem()
			}

			actions := client.Actions()
			if reflect.DeepEqual(actions, tc.ExpectActions) {
				t.Errorf("Unexpected actions:\n%s", diff.ObjectGoPrintDiff(actions, tc.ExpectActions))
			}
		})
	}
}

func defaultCrtConfigMapPtr() *v1.ConfigMap {
	tmp := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: ServiceCACertConfigMapName,
			Annotations: map[string]string{
				"service.beta.openshift.io/inject-cabundle": "true",
			},
		},
	}
	tmp.Namespace = metav1.NamespaceDefault
	return &tmp
}
