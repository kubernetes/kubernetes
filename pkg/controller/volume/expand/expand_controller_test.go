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
	"regexp"
	"testing"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	csitranslationplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/controller"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/attachdetach/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/awsebs"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

func TestSyncHandler(t *testing.T) {
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	pvInformer := informerFactory.Core().V1().PersistentVolumes()
	storageClassInformer := informerFactory.Storage().V1().StorageClasses()

	tests := []struct {
		name                string
		csiMigrationEnabled bool
		expansionCalled     bool
		storageClass        *storagev1.StorageClass
		pvcKey              string
		pv                  *v1.PersistentVolume
		pvc                 *v1.PersistentVolumeClaim
		hasError            bool
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
			name:            "when pvc and pv has everything for in-tree plugin",
			pv:              getFakePersistentVolume("vol-3", csitranslationplugins.AWSEBSInTreePluginName, "good-pvc-vol-3"),
			pvc:             getFakePersistentVolumeClaim("good-pvc", "vol-3", "resizable2", "good-pvc-vol-3"),
			storageClass:    getFakeStorageClass("resizable2", csitranslationplugins.AWSEBSInTreePluginName),
			pvcKey:          "default/good-pvc",
			expansionCalled: true,
		},
		{
			name:                "when csi migration is enabled for a in-tree plugin",
			csiMigrationEnabled: true,
			pv:                  getFakePersistentVolume("vol-4", csitranslationplugins.AWSEBSInTreePluginName, "csi-pvc-vol-4"),
			pvc:                 getFakePersistentVolumeClaim("csi-pvc", "vol-4", "resizable3", "csi-pvc-vol-4"),
			storageClass:        getFakeStorageClass("resizable3", csitranslationplugins.AWSEBSInTreePluginName),
			pvcKey:              "default/csi-pvc",
		},
	}

	for _, tc := range tests {
		test := tc
		if tc.pv != nil {
			informerFactory.Core().V1().PersistentVolumes().Informer().GetIndexer().Add(tc.pv)
		}

		if tc.pvc != nil {
			informerFactory.Core().V1().PersistentVolumeClaims().Informer().GetIndexer().Add(tc.pvc)
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
		}

		var expController *expandController
		expController, _ = expc.(*expandController)
		var expansionCalled bool
		expController.operationGenerator = operationexecutor.NewFakeOgCounter(func() (error, error) {
			expansionCalled = true
			return nil, nil
		})

		err = expController.syncHandler(test.pvcKey)
		if err != nil && !test.hasError {
			t.Fatalf("for: %s; unexpected error while running handler : %v", test.name, err)
		}
		if expansionCalled != test.expansionCalled {
			t.Fatalf("for: %s; expected expansionCalled to be %v but was %v", test.name, test.expansionCalled, expansionCalled)
		}
	}
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
