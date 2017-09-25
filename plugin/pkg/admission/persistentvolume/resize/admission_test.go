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

package resize

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/storage"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	"k8s.io/kubernetes/pkg/controller"
)

func getResourceList(storage string) api.ResourceList {
	res := api.ResourceList{}
	if storage != "" {
		res[api.ResourceStorage] = resource.MustParse(storage)
	}
	return res
}

func TestPVCResizeAdmission(t *testing.T) {
	goldClassName := "gold"
	trueVal := true
	falseVar := false
	goldClass := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: goldClassName,
		},
		Provisioner:          "kubernetes.io/glusterfs",
		AllowVolumeExpansion: &trueVal,
	}
	silverClassName := "silver"
	silverClass := &storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: silverClassName,
		},
		Provisioner:          "kubernetes.io/glusterfs",
		AllowVolumeExpansion: &falseVar,
	}
	expectNoError := func(err error) bool {
		return err == nil
	}
	expectDynamicallyProvisionedError := func(err error) bool {
		return strings.Contains(err.Error(), "only dynamically provisioned pvc can be resized and "+
			"the storageclass that provisions the pvc must support resize")
	}
	expectVolumePluginError := func(err error) bool {
		return strings.Contains(err.Error(), "volume plugin does not support resize")
	}
	tests := []struct {
		name        string
		resource    schema.GroupVersionResource
		subresource string
		oldObj      runtime.Object
		newObj      runtime.Object

		checkError func(error) bool
	}{
		{
			name:     "pvc-resize, update, no error",
			resource: api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			oldObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume1",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
					StorageClassName: &goldClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("1Gi"),
					Phase:    api.ClaimBound,
				},
			},
			newObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume1",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("2Gi"),
					},
					StorageClassName: &goldClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("2Gi"),
					Phase:    api.ClaimBound,
				},
			},
			checkError: expectNoError,
		},
		{
			name:     "pvc-resize, update, volume plugin error",
			resource: api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			oldObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume2",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
					StorageClassName: &goldClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("1Gi"),
					Phase:    api.ClaimBound,
				},
			},
			newObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume2",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("2Gi"),
					},
					StorageClassName: &goldClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("2Gi"),
					Phase:    api.ClaimBound,
				},
			},
			checkError: expectVolumePluginError,
		},
		{
			name:     "pvc-resize, update, dynamically provisioned error",
			resource: api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			oldObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume3",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("1Gi"),
					Phase:    api.ClaimBound,
				},
			},
			newObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume3",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("2Gi"),
					},
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("2Gi"),
					Phase:    api.ClaimBound,
				},
			},
			checkError: expectDynamicallyProvisionedError,
		},
		{
			name:     "pvc-resize, update, dynamically provisioned error",
			resource: api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			oldObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume4",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
					StorageClassName: &silverClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("1Gi"),
					Phase:    api.ClaimBound,
				},
			},
			newObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume4",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("2Gi"),
					},
					StorageClassName: &silverClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("2Gi"),
					Phase:    api.ClaimBound,
				},
			},
			checkError: expectDynamicallyProvisionedError,
		},
		{
			name:     "PVC update with no change in size",
			resource: api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			oldObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
					StorageClassName: &silverClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("0Gi"),
					Phase:    api.ClaimPending,
				},
			},
			newObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					VolumeName: "volume4",
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
					StorageClassName: &silverClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("1Gi"),
					Phase:    api.ClaimBound,
				},
			},
			checkError: expectNoError,
		},
		{
			name:     "expand pvc in pending state",
			resource: api.SchemeGroupVersion.WithResource("persistentvolumeclaims"),
			oldObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					Resources: api.ResourceRequirements{
						Requests: getResourceList("1Gi"),
					},
					StorageClassName: &silverClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("0Gi"),
					Phase:    api.ClaimPending,
				},
			},
			newObj: &api.PersistentVolumeClaim{
				Spec: api.PersistentVolumeClaimSpec{
					Resources: api.ResourceRequirements{
						Requests: getResourceList("2Gi"),
					},
					StorageClassName: &silverClassName,
				},
				Status: api.PersistentVolumeClaimStatus{
					Capacity: getResourceList("0Gi"),
					Phase:    api.ClaimPending,
				},
			},
			checkError: func(err error) bool {
				return strings.Contains(err.Error(), "Only bound persistent volume claims can be expanded")
			},
		},
	}

	ctrl := newPlugin()
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	ctrl.SetInternalKubeInformerFactory(informerFactory)
	err := ctrl.Validate()
	if err != nil {
		t.Fatalf("neither pv lister nor storageclass lister can be nil")
	}

	pv1 := &api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "volume1"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				Glusterfs: &api.GlusterfsVolumeSource{
					EndpointsName: "http://localhost:8080/",
					Path:          "/heketi",
					ReadOnly:      false,
				},
			},
			StorageClassName: goldClassName,
		},
	}
	pv2 := &api.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: "volume2"},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{},
			},
			StorageClassName: goldClassName,
		},
	}

	pvs := []*api.PersistentVolume{}
	pvs = append(pvs, pv1, pv2)

	for _, pv := range pvs {
		err := informerFactory.Core().InternalVersion().PersistentVolumes().Informer().GetStore().Add(pv)
		if err != nil {
			fmt.Println("add pv error: ", err)
		}
	}

	scs := []*storage.StorageClass{}
	scs = append(scs, goldClass, silverClass)
	for _, sc := range scs {
		err := informerFactory.Storage().InternalVersion().StorageClasses().Informer().GetStore().Add(sc)
		if err != nil {
			fmt.Println("add storageclass error: ", err)
		}
	}

	for _, tc := range tests {
		operation := admission.Update
		attributes := admission.NewAttributesRecord(tc.newObj, tc.oldObj, schema.GroupVersionKind{}, metav1.NamespaceDefault, "foo", tc.resource, tc.subresource, operation, nil)

		err := ctrl.Admit(attributes)
		fmt.Println(tc.name)
		fmt.Println(err)
		if !tc.checkError(err) {
			t.Errorf("%v: unexpected err: %v", tc.name, err)
		}
	}

}
