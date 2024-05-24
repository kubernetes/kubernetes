/*
Copyright 2019 The Kubernetes Authors.

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

package expand

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	coretesting "k8s.io/client-go/testing"
	csitrans "k8s.io/csi-translation-lib"
	csitranslationplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/controller"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csimigration"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name               string
		pvcKey             string
		pv                 *v1.PersistentVolume
		pvc                *v1.PersistentVolumeClaim
		expansionCalled    bool
		hasError           bool
		expectedAnnotation map[string]string
	}{
		{
			name:     "when pvc has no PV binding",
			pvc:      getFakePersistentVolumeClaim("no-pv-pvc", "", "1Gi", "1Gi", ""),
			pvcKey:   "default/no-pv-pvc",
			hasError: true,
		},
		{
			name:               "when pvc and pv has everything for in-tree plugin",
			pv:                 getFakePersistentVolume("vol-3", csitranslationplugins.AWSEBSInTreePluginName, "1Gi", "good-pvc-vol-3"),
			pvc:                getFakePersistentVolumeClaim("good-pvc", "vol-3", "1Gi", "2Gi", "good-pvc-vol-3"),
			pvcKey:             "default/good-pvc",
			expansionCalled:    false,
			expectedAnnotation: map[string]string{volumetypes.VolumeResizerKey: csitranslationplugins.AWSEBSDriverName},
		},
		{
			name: "if pv has pre-resize capacity annotation, generate expand operation should not be called",
			pv: func() *v1.PersistentVolume {
				pv := getFakePersistentVolume("vol-4", csitranslationplugins.AWSEBSInTreePluginName, "2Gi", "good-pvc-vol-4")
				pv.ObjectMeta.Annotations = make(map[string]string)
				pv.ObjectMeta.Annotations[util.AnnPreResizeCapacity] = "1Gi"
				return pv
			}(),
			pvc:             getFakePersistentVolumeClaim("good-pvc", "vol-4", "2Gi", "2Gi", "good-pvc-vol-4"),
			pvcKey:          "default/good-pvc",
			expansionCalled: false,
		},
		{
			name:            "for csi plugin without migration path",
			pv:              getFakePersistentVolume("vol-5", "com.csi.ceph", "1Gi", "ceph-csi-pvc-vol-6"),
			pvc:             getFakePersistentVolumeClaim("ceph-csi-pvc", "vol-5", "1Gi", "2Gi", "ceph-csi-pvc-vol-6"),
			pvcKey:          "default/ceph-csi-pvc",
			expansionCalled: false,
			hasError:        false,
		},
	}

	for _, tc := range tests {
		tCtx := ktesting.Init(t)
		test := tc
		fakeKubeClient := controllervolumetesting.CreateTestClient()
		informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
		pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()

		pvc := test.pvc
		if tc.pv != nil {
			informerFactory.Core().V1().PersistentVolumes().Informer().GetIndexer().Add(tc.pv)
		}

		if tc.pvc != nil {
			informerFactory.Core().V1().PersistentVolumeClaims().Informer().GetIndexer().Add(pvc)
		}
		allPlugins := []volume.VolumePlugin{}
		translator := csitrans.New()
		expc, err := NewExpandController(tCtx, fakeKubeClient, pvcInformer, nil, allPlugins, translator, csimigration.NewPluginManager(translator, utilfeature.DefaultFeatureGate))
		if err != nil {
			t.Fatalf("error creating expand controller : %v", err)
		}

		var expController *expandController
		expController, _ = expc.(*expandController)
		var expansionCalled bool
		expController.operationGenerator = operationexecutor.NewFakeOGCounter(func() volumetypes.OperationContext {
			expansionCalled = true
			return volumetypes.NewOperationContext(nil, nil, false)
		})

		if test.pv != nil {
			fakeKubeClient.AddReactor("get", "persistentvolumes", func(action coretesting.Action) (bool, runtime.Object, error) {
				return true, test.pv, nil
			})
		}
		fakeKubeClient.AddReactor("patch", "persistentvolumeclaims", func(action coretesting.Action) (bool, runtime.Object, error) {
			if action.GetSubresource() == "status" {
				patchActionaction, _ := action.(coretesting.PatchAction)
				pvc, err = applyPVCPatch(pvc, patchActionaction.GetPatch())
				if err != nil {
					return false, nil, err
				}
				return true, pvc, nil
			}
			return true, pvc, nil
		})

		err = expController.syncHandler(context.TODO(), test.pvcKey)
		if err != nil && !test.hasError {
			t.Fatalf("for: %s; unexpected error while running handler : %v", test.name, err)
		}

		if err == nil && test.hasError {
			t.Fatalf("for: %s; unexpected success", test.name)
		}
		if expansionCalled != test.expansionCalled {
			t.Fatalf("for: %s; expected expansionCalled to be %v but was %v", test.name, test.expansionCalled, expansionCalled)
		}

		if len(test.expectedAnnotation) != 0 && !reflect.DeepEqual(test.expectedAnnotation, pvc.Annotations) {
			t.Fatalf("for: %s; expected %v annotations, got %v", test.name, test.expectedAnnotation, pvc.Annotations)
		}
	}
}

func applyPVCPatch(originalPVC *v1.PersistentVolumeClaim, patch []byte) (*v1.PersistentVolumeClaim, error) {
	pvcData, err := json.Marshal(originalPVC)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal pvc with %v", err)
	}
	updated, err := strategicpatch.StrategicMergePatch(pvcData, patch, v1.PersistentVolumeClaim{})
	if err != nil {
		return nil, fmt.Errorf("failed to apply patch on pvc %v", err)
	}
	updatedPVC := &v1.PersistentVolumeClaim{}
	if err := json.Unmarshal(updated, updatedPVC); err != nil {
		return nil, fmt.Errorf("failed to unmarshal updated pvc : %v", err)
	}
	return updatedPVC, nil
}

func getFakePersistentVolume(volumeName, pluginName string, size string, pvcUID types.UID) *v1.PersistentVolume {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{},
			ClaimRef: &v1.ObjectReference{
				Namespace: "default",
			},
			Capacity: map[v1.ResourceName]resource.Quantity{
				v1.ResourceStorage: resource.MustParse(size),
			},
		},
	}
	if pvcUID != "" {
		pv.Spec.ClaimRef.UID = pvcUID
	}

	if matched, _ := regexp.MatchString(`csi`, pluginName); matched {
		pv.Spec.PersistentVolumeSource.CSI = &v1.CSIPersistentVolumeSource{
			Driver:       pluginName,
			VolumeHandle: volumeName,
		}
	} else {
		pv.Spec.PersistentVolumeSource.AWSElasticBlockStore = &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID: volumeName,
			FSType:   "ext4",
		}
	}
	return pv
}

func getFakePersistentVolumeClaim(pvcName, volumeName, statusSize, requestSize string, uid types.UID) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: pvcName, Namespace: "default", UID: uid},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.VolumeResourceRequirements{
				Requests: map[v1.ResourceName]resource.Quantity{
					v1.ResourceStorage: resource.MustParse(requestSize),
				},
			},
		},
		Status: v1.PersistentVolumeClaimStatus{
			Capacity: map[v1.ResourceName]resource.Quantity{
				v1.ResourceStorage: resource.MustParse(statusSize),
			},
		},
	}
	if volumeName != "" {
		pvc.Spec.VolumeName = volumeName
	}

	return pvc
}
