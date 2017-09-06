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

package cache

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

func Test_AddValidPVCUpdate(t *testing.T) {
	claim := testVolumeClaim("foo", "ns", v1.PersistentVolumeClaimSpec{
		AccessModes: []v1.PersistentVolumeAccessMode{
			v1.ReadWriteOnce,
			v1.ReadOnlyMany,
		},
		Resources: v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("12G"),
			},
		},
		VolumeName: "foo",
	})

	unboundClaim := claim.DeepCopy()
	unboundClaim.Status.Phase = v1.ClaimPending

	noResizeClaim := claim.DeepCopy()
	noResizeClaim.Status.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): resource.MustParse("12G"),
	}

	boundPV := getPersistentVolume("foo", resource.MustParse("10G"), claim)
	unboundPV := getPersistentVolume("foo", resource.MustParse("10G"), nil)
	misboundPV := getPersistentVolume("foo", resource.MustParse("10G"), nil)
	misboundPV.Spec.ClaimRef = &v1.ObjectReference{
		Namespace: "someOtherNamespace",
		Name:      "someOtherName",
	}

	tests := []struct {
		name         string
		pvc          *v1.PersistentVolumeClaim
		pv           *v1.PersistentVolume
		expectedPVCs int
	}{
		{
			"validPVCUpdate",
			claim,
			boundPV,
			1,
		},
		{
			"noResizeRequired",
			noResizeClaim,
			boundPV,
			0,
		},
		{
			"unboundPVC",
			unboundClaim,
			boundPV,
			0,
		},
		{
			"unboundPV",
			claim,
			unboundPV,
			0,
		},
		{
			"misboundPV",
			claim,
			misboundPV,
			0,
		},
	}
	for _, test := range tests {
		resizeMap := createTestVolumeResizeMap()
		pvc := test.pvc.DeepCopy()
		pv := test.pv.DeepCopy()
		resizeMap.AddPVCUpdate(pvc, pv)
		pvcr := resizeMap.GetPVCsWithResizeRequest()
		if len(pvcr) != test.expectedPVCs {
			t.Errorf("Test %q expected %d pvc resize request got %d", test.name, test.expectedPVCs, len(pvcr))
		}
		if test.expectedPVCs > 0 {
			assert.Equal(t, resource.MustParse("12G"), pvcr[0].ExpectedSize, test.name)
		}
		assert.Equal(t, 0, len(resizeMap.pvcrs), test.name)
	}
}

func createTestVolumeResizeMap() *volumeResizeMap {
	fakeClient := &fake.Clientset{}
	resizeMap := &volumeResizeMap{}
	resizeMap.pvcrs = make(map[types.UniquePVCName]*PVCWithResizeRequest)
	resizeMap.kubeClient = fakeClient
	return resizeMap
}

func testVolumeClaim(name string, namespace string, spec v1.PersistentVolumeClaimSpec) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
		Status: v1.PersistentVolumeClaimStatus{
			Phase: v1.ClaimBound,
		},
	}
}

func getPersistentVolume(volumeName string, capacity resource.Quantity, pvc *v1.PersistentVolumeClaim) *v1.PersistentVolume {
	volume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): capacity,
			},
		},
	}
	if pvc != nil {
		volume.Spec.ClaimRef = &v1.ObjectReference{
			Namespace: pvc.Namespace,
			Name:      pvc.Name,
		}
	}
	return volume
}
