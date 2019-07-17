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
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"testing"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	coretesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	csitranslationplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/controller"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name                string
		csiMigrationEnabled bool
		storageClass        *storagev1.StorageClass
		pvcKey              string
		pv                  *v1.PersistentVolume
		pvc                 *v1.PersistentVolumeClaim
		expansionCalled     bool
		hasError            bool
		expectedAnnotation  map[string]string
	}{
		{
			name:     "when pvc has no PV binding",
			pvc:      getFakePersistentVolumeClaim("no-pv-pvc", "", "", ""),
			pvcKey:   "default/no-pv-pvc",
			hasError: true,
		},
		{
			name:   "when pvc has no storageclass",
			pv:     getFakePersistentVolume("vol-1", csitranslationplugins.AWSEBSInTreePluginName, "no-sc-pvc-vol-1"),
			pvc:    getFakePersistentVolumeClaim("no-sc-pvc", "vol-1", "", "no-sc-pvc-vol-1"),
			pvcKey: "default/no-sc-pvc",
		},
		{
			name:   "when pvc storageclass is missing",
			pv:     getFakePersistentVolume("vol-2", csitranslationplugins.AWSEBSInTreePluginName, "missing-sc-pvc-vol-2"),
			pvc:    getFakePersistentVolumeClaim("missing-sc-pvc", "vol-2", "resizable", "missing-sc-pvc-vol-2"),
			pvcKey: "default/missing-sc-pvc",
		},
		{
			name:               "when pvc and pv has everything for in-tree plugin",
			pv:                 getFakePersistentVolume("vol-3", csitranslationplugins.AWSEBSInTreePluginName, "good-pvc-vol-3"),
			pvc:                getFakePersistentVolumeClaim("good-pvc", "vol-3", "resizable2", "good-pvc-vol-3"),
			storageClass:       getFakeStorageClass("resizable2", csitranslationplugins.AWSEBSInTreePluginName),
			pvcKey:             "default/good-pvc",
			expansionCalled:    true,
			expectedAnnotation: map[string]string{volumetypes.VolumeResizerKey: csitranslationplugins.AWSEBSInTreePluginName},
		},
		{
			name:                "when csi migration is enabled for a in-tree plugin",
			csiMigrationEnabled: true,
			pv:                  getFakePersistentVolume("vol-4", csitranslationplugins.AWSEBSInTreePluginName, "csi-pvc-vol-4"),
			pvc:                 getFakePersistentVolumeClaim("csi-pvc", "vol-4", "resizable3", "csi-pvc-vol-4"),
			storageClass:        getFakeStorageClass("resizable3", csitranslationplugins.AWSEBSInTreePluginName),
			pvcKey:              "default/csi-pvc",
			expectedAnnotation:  map[string]string{volumetypes.VolumeResizerKey: csitranslationplugins.AWSEBSDriverName},
		},
		{
			name:            "for csi plugin without migration path",
			pv:              getFakePersistentVolume("vol-5", "com.csi.ceph", "ceph-csi-pvc-vol-5"),
			pvc:             getFakePersistentVolumeClaim("ceph-csi-pvc", "vol-5", "resizable4", "ceph-csi-pvc-vol-5"),
			storageClass:    getFakeStorageClass("resizable4", "com.csi.ceph"),
			pvcKey:          "default/ceph-csi-pvc",
			expansionCalled: false,
			hasError:        false,
		},
	}

	for _, tc := range tests {
		test := tc
		fakeKubeClient := controllervolumetesting.CreateTestClient()
		informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
		pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
		pvInformer := informerFactory.Core().V1().PersistentVolumes()
		storageClassInformer := informerFactory.Storage().V1().StorageClasses()

		pvc := test.pvc
		if tc.pv != nil {
			informerFactory.Core().V1().PersistentVolumes().Informer().GetIndexer().Add(tc.pv)
		}

		if tc.pvc != nil {
			informerFactory.Core().V1().PersistentVolumeClaims().Informer().GetIndexer().Add(pvc)
		}
		allPlugins := []volume.VolumePlugin{}
		allPlugins = append(allPlugins, awsebs.ProbeVolumePlugins()...)
		if tc.storageClass != nil {
			informerFactory.Storage().V1().StorageClasses().Informer().GetIndexer().Add(tc.storageClass)
		}
		expc, err := NewExpandController(fakeKubeClient, pvcInformer, pvInformer, storageClassInformer, nil, allPlugins)
		if err != nil {
			t.Fatalf("error creating expand controller : %v", err)
		}

		if test.csiMigrationEnabled {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigrationAWS, true)()
		} else {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, false)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigrationAWS, false)()
		}

		var expController *expandController
		expController, _ = expc.(*expandController)
		var expansionCalled bool
		expController.operationGenerator = operationexecutor.NewFakeOGCounter(func() (error, error) {
			expansionCalled = true
			return nil, nil
		})

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

		err = expController.syncHandler(test.pvcKey)
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

func getFakePersistentVolume(volumeName, pluginName string, pvcUID types.UID) *v1.PersistentVolume {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{},
			ClaimRef: &v1.ObjectReference{
				Namespace: "default",
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

func getFakePersistentVolumeClaim(pvcName, volumeName, scName string, uid types.UID) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: pvcName, Namespace: "default", UID: uid},
		Spec:       v1.PersistentVolumeClaimSpec{},
	}
	if volumeName != "" {
		pvc.Spec.VolumeName = volumeName
	}

	if scName != "" {
		pvc.Spec.StorageClassName = &scName
	}
	return pvc
}

func getFakeStorageClass(scName, pluginName string) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		ObjectMeta:  metav1.ObjectMeta{Name: scName},
		Provisioner: pluginName,
	}
}
