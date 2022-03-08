/*
Copyright 2022 The Kubernetes Authors.

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

package operationexecutor

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"testing"
)

func TestNodeExpander(t *testing.T) {
	var tests = []struct {
		name               string
		pvc                *v1.PersistentVolumeClaim
		pv                 *v1.PersistentVolume
		recoverFeatureGate bool

		// expectations of test
		expectedResizeStatus     v1.PersistentVolumeClaimResizeStatus
		expectedStatusSize       resource.Quantity
		expectResizeCall         bool
		assumeResizeOpAsFinished bool
		expectError              bool
	}{
		{
			name:               "pv.spec.cap > pvc.status.cap, resizeStatus=node_expansion_failed",
			pvc:                getTestPVC("test-vol0", "2G", "1G", "", v1.PersistentVolumeClaimNodeExpansionFailed),
			pv:                 getTestPV("test-vol0", "2G"),
			recoverFeatureGate: true,

			expectedResizeStatus:     v1.PersistentVolumeClaimNodeExpansionFailed,
			expectResizeCall:         false,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("1G"),
		},
		{
			name:                     "pv.spec.cap > pvc.status.cap, resizeStatus=node_expansion_pending",
			pvc:                      getTestPVC("test-vol0", "2G", "1G", "2G", v1.PersistentVolumeClaimNodeExpansionPending),
			pv:                       getTestPV("test-vol0", "2G"),
			recoverFeatureGate:       true,
			expectedResizeStatus:     v1.PersistentVolumeClaimNoExpansionInProgress,
			expectResizeCall:         true,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("2G"),
		},
		{
			name:                     "pv.spec.cap > pvc.status.cap, resizeStatus=node_expansion_pending, reize_op=failing",
			pvc:                      getTestPVC(volumetesting.AlwaysFailNodeExpansion, "2G", "1G", "2G", v1.PersistentVolumeClaimNodeExpansionPending),
			pv:                       getTestPV(volumetesting.AlwaysFailNodeExpansion, "2G"),
			recoverFeatureGate:       true,
			expectError:              true,
			expectedResizeStatus:     v1.PersistentVolumeClaimNodeExpansionFailed,
			expectResizeCall:         true,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("1G"),
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, test.recoverFeatureGate)()
			volumePluginMgr, fakePlugin := volumetesting.GetTestKubeletVolumePluginMgr(t)

			pvc := test.pvc
			pv := test.pv
			pod := getTestPod("test-pod", pvc.Name)
			og := getTestOperationGenerator(volumePluginMgr, pvc, pv)

			vmt := VolumeToMount{
				Pod:        pod,
				VolumeName: v1.UniqueVolumeName(pv.Name),
				VolumeSpec: volume.NewSpecFromPersistentVolume(pv, false),
			}
			resizeOp := nodeResizeOperationOpts{
				pvc:                pvc,
				pv:                 pv,
				volumePlugin:       fakePlugin,
				vmt:                vmt,
				actualStateOfWorld: nil,
				pluginResizeOpts: volume.NodeResizeOptions{
					VolumeSpec: vmt.VolumeSpec,
					NewSize:    *pv.Spec.Capacity.Storage(),
					OldSize:    *pvc.Status.Capacity.Storage(),
				},
			}
			ogInstance, _ := og.(*operationGenerator)
			nodeExpander := newNodeExpander(resizeOp, ogInstance.kubeClient, ogInstance.recorder)

			_, err, expansionResponse := nodeExpander.expandOnPlugin()

			pvc = nodeExpander.pvc
			pvcStatusCap := pvc.Status.Capacity[v1.ResourceStorage]

			if !test.expectError && err != nil {
				t.Errorf("For test %s, expected no error got: %v", test.name, err)
			}
			if test.expectError && err == nil {
				t.Errorf("For test %s, expected error but got none", test.name)
			}

			if test.expectResizeCall != expansionResponse.resizeCalledOnPlugin {
				t.Errorf("For test %s, expected resize called %t, got %t", test.name, test.expectResizeCall, expansionResponse.resizeCalledOnPlugin)
			}
			if test.assumeResizeOpAsFinished != expansionResponse.assumeResizeFinished {
				t.Errorf("For test %s, expected assumeResizeOpAsFinished %t, got %t", test.name, test.assumeResizeOpAsFinished, expansionResponse.assumeResizeFinished)
			}
			if test.expectedResizeStatus != *pvc.Status.ResizeStatus {
				t.Errorf("For test %s, expected resizeStatus %v, got %v", test.name, test.expectedResizeStatus, *pvc.Status.ResizeStatus)
			}
			if pvcStatusCap.Cmp(test.expectedStatusSize) != 0 {
				t.Errorf("For test %s, expected status size %s, got %s", test.name, test.expectedStatusSize.String(), pvcStatusCap.String())
			}
		})
	}
}
