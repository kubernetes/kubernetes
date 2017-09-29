/*
Copyright 2017 The Kubernetes Authors.

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

package cloud

import (
	"encoding/json"
	"testing"
	"time"

	"k8s.io/api/core/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"

	fakecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/fake"
)

func TestCreatePatch(t *testing.T) {
	ignoredPV := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "noncloud",
			Initializers: &metav1.Initializers{
				Pending: []metav1.Initializer{
					{
						Name: initializerName,
					},
				},
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/",
				},
			},
		},
	}
	awsPV := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "awsPV",
			Initializers: &metav1.Initializers{
				Pending: []metav1.Initializer{
					{
						Name: initializerName,
					},
				},
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}

	testCases := map[string]struct {
		vol    v1.PersistentVolume
		labels map[string]string
	}{
		"non-cloud PV": {
			vol:    ignoredPV,
			labels: nil,
		},
		"no labels": {
			vol:    awsPV,
			labels: nil,
		},
		"cloudprovider returns nil, nil": {
			vol:    awsPV,
			labels: nil,
		},
		"cloudprovider labels": {
			vol:    awsPV,
			labels: map[string]string{"a": "1", "b": "2"},
		},
	}

	for d, tc := range testCases {
		cloud := &fakecloud.FakeCloud{}
		client := fake.NewSimpleClientset()
		pvlController := NewPersistentVolumeLabelController(client, cloud)
		patch, err := pvlController.createPatch(&tc.vol, tc.labels)
		if err != nil {
			t.Errorf("%s: createPatch returned err: %v", d, err)
		}
		obj := &v1.PersistentVolume{}
		json.Unmarshal(patch, obj)
		if tc.labels != nil {
			for k, v := range tc.labels {
				if obj.ObjectMeta.Labels[k] != v {
					t.Errorf("%s: label %s expected %s got %s", d, k, v, obj.ObjectMeta.Labels[k])
				}
			}
		}
		if obj.ObjectMeta.Initializers != nil {
			t.Errorf("%s: initializer wasn't removed: %v", d, obj.ObjectMeta.Initializers)
		}
	}
}

func TestAddLabelsToVolume(t *testing.T) {
	pv := v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: "awsPV",
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: "123",
				},
			},
		},
	}

	testCases := map[string]struct {
		vol          v1.PersistentVolume
		initializers *metav1.Initializers
		shouldLabel  bool
	}{
		"PV without initializer": {
			vol:          pv,
			initializers: nil,
			shouldLabel:  false,
		},
		"PV with initializer to remove": {
			vol:          pv,
			initializers: &metav1.Initializers{Pending: []metav1.Initializer{{Name: initializerName}}},
			shouldLabel:  true,
		},
		"PV with other initializers": {
			vol:          pv,
			initializers: &metav1.Initializers{Pending: []metav1.Initializer{{Name: "OtherInit"}}},
			shouldLabel:  false,
		},
	}

	for d, tc := range testCases {
		labeledCh := make(chan bool, 1)
		client := fake.NewSimpleClientset()
		client.PrependReactor("patch", "persistentvolumes", func(action core.Action) (handled bool, ret runtime.Object, err error) {
			patch := action.(core.PatchActionImpl).GetPatch()
			obj := &v1.PersistentVolume{}
			json.Unmarshal(patch, obj)
			if obj.ObjectMeta.Labels["a"] != "1" {
				return false, nil, nil
			}
			labeledCh <- true
			return true, nil, nil
		})

		fakeCloud := &fakecloud.FakeCloud{
			VolumeLabelMap: map[string]map[string]string{"awsPV": {"a": "1"}},
		}
		pvlController := &PersistentVolumeLabelController{kubeClient: client, cloud: fakeCloud}
		tc.vol.ObjectMeta.Initializers = tc.initializers
		pvlController.addLabelsToVolume(&tc.vol)

		select {
		case l := <-labeledCh:
			if l != tc.shouldLabel {
				t.Errorf("%s: label of pv failed.  expected %t got %t", d, tc.shouldLabel, l)
			}
		case <-time.After(500 * time.Millisecond):
			if tc.shouldLabel != false {
				t.Errorf("%s: timed out waiting for label notification", d)
			}
		}
	}
}
