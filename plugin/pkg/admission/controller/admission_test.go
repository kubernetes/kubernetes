package controller

import (
	"testing"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"


)

func TestAdmissionNamespaceExistsCreate(t *testing.T) {
	namespace1 := "test1"
	namespace2 := "test2"

	selector1 := map[string]string {
		"key1" : "value1",
		"key2" : "value2",
	}
	selector2 := map[string]string {
		"key2" : "value2",
		"key1" : "value1",
	}
	selector3 := map[string]string {
		"key1" : "kkk",
		"key2" : "hhh",
	}

	testCases := []struct {
		rc 		api.ReplicationController
		fit 	bool
		info 	string
	}{
		{
			rc:  api.ReplicationController {
				ObjectMeta: api.ObjectMeta {
					Name: "rc2",
					Namespace: namespace1,
				},
				Spec: api.ReplicationControllerSpec{
					Selector: selector1,
				},
			},
			fit:  false,
			info: "two same rc in the same namespace, should not fit",
		},
		{
			rc:  api.ReplicationController {
				ObjectMeta: api.ObjectMeta {
					Name: "rc2",
					Namespace: namespace1,
				},
				Spec: api.ReplicationControllerSpec{
					Selector: selector2,
				},
			},
			fit:  false,
			info: "two same rc in the same namespace, should not fit",
		},
		{
			rc:  api.ReplicationController {
				ObjectMeta: api.ObjectMeta {
					Name: "rc2",
					Namespace: namespace2,
				},
				Spec: api.ReplicationControllerSpec{
					Selector: selector1,
				},
			},
			fit:  true,
			info: "two same rc in different namespace, should fit",
		},
		{
			rc:  api.ReplicationController {
				ObjectMeta: api.ObjectMeta {
					Name: "rc2",
					Namespace: namespace2,
				},
				Spec: api.ReplicationControllerSpec{
					Selector: selector2,
				},
			},
			fit:  true,
			info: "two same rc in different namespace, should fit",
		},
		{
			rc:  api.ReplicationController {
				ObjectMeta: api.ObjectMeta {
					Name: "rc2",
					Namespace: namespace2,
				},
				Spec: api.ReplicationControllerSpec{
					Selector: selector3,
				},
			},
			fit:  true,
			info: "two different rc in different namespace, should fit",
		},
	}

	mockClient := &testclient.Fake{}
	store := cache.NewStore(NamespaceSelectorKeyFunc)
	store.Add(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: "rc1",
			Namespace: namespace1,
		},
		Spec: api.ReplicationControllerSpec{
			Selector: selector1,
		},
	})
	handler := &exists{
		client: mockClient,
		store:  store,
	}

	for _, test := range testCases {
		err := handler.Admit(admission.NewAttributesRecord(&test.rc, "ReplicationController", test.rc.Namespace, test.rc.Name, "replicationcontrollers", "", admission.Create, nil))
		if test.fit && err != nil || !test.fit && err == nil {
			t.Errorf("%s", test.info)
		}
	}
}
