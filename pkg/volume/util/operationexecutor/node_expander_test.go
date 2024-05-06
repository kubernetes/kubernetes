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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
)

func TestNodeExpander(t *testing.T) {
	nodeResizeFailed := v1.PersistentVolumeClaimNodeResizeFailed

	nodeResizePending := v1.PersistentVolumeClaimNodeResizePending
	var tests = []struct {
		name string
		pvc  *v1.PersistentVolumeClaim
		pv   *v1.PersistentVolume

		// desired size, defaults to pv.Spec.Capacity
		desiredSize *resource.Quantity
		// actualSize, defaults to pvc.Status.Capacity
		actualSize *resource.Quantity

		// expectations of test
		expectedResizeStatus     v1.ClaimResourceStatus
		expectedStatusSize       resource.Quantity
		expectResizeCall         bool
		assumeResizeOpAsFinished bool
		expectError              bool
	}{
		{
			name: "pv.spec.cap > pvc.status.cap, resizeStatus=node_expansion_failed",
			pvc:  getTestPVC("test-vol0", "2G", "1G", "", &nodeResizeFailed),
			pv:   getTestPV("test-vol0", "2G"),

			expectedResizeStatus:     nodeResizeFailed,
			expectResizeCall:         false,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("1G"),
		},
		{
			name:                     "pv.spec.cap > pvc.status.cap, resizeStatus=node_expansion_pending",
			pvc:                      getTestPVC("test-vol0", "2G", "1G", "2G", &nodeResizePending),
			pv:                       getTestPV("test-vol0", "2G"),
			expectedResizeStatus:     "",
			expectResizeCall:         true,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("2G"),
		},
		{
			name:                     "pv.spec.cap > pvc.status.cap, resizeStatus=node_expansion_pending, reize_op=failing",
			pvc:                      getTestPVC(volumetesting.AlwaysFailNodeExpansion, "2G", "1G", "2G", &nodeResizePending),
			pv:                       getTestPV(volumetesting.AlwaysFailNodeExpansion, "2G"),
			expectError:              true,
			expectedResizeStatus:     nodeResizeFailed,
			expectResizeCall:         true,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("1G"),
		},
		{
			name: "pv.spec.cap = pvc.status.cap, resizeStatus='', desiredSize > actualSize",
			pvc:  getTestPVC("test-vol0", "2G", "2G", "2G", nil),
			pv:   getTestPV("test-vol0", "2G"),

			expectedResizeStatus:     "",
			expectResizeCall:         true,
			assumeResizeOpAsFinished: true,
			expectedStatusSize:       resource.MustParse("2G"),
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RecoverVolumeExpansionFailure, true)
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
			desiredSize := test.desiredSize
			if desiredSize == nil {
				desiredSize = pv.Spec.Capacity.Storage()
			}
			actualSize := test.actualSize
			if actualSize == nil {
				actualSize = pvc.Status.Capacity.Storage()
			}
			resizeOp := nodeResizeOperationOpts{
				pvc:                pvc,
				pv:                 pv,
				volumePlugin:       fakePlugin,
				vmt:                vmt,
				actualStateOfWorld: nil,
				pluginResizeOpts: volume.NodeResizeOptions{
					VolumeSpec: vmt.VolumeSpec,
					NewSize:    *desiredSize,
					OldSize:    *actualSize,
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
			allocatedResourceStatus := pvc.Status.AllocatedResourceStatuses
			resizeStatus := allocatedResourceStatus[v1.ResourceStorage]

			if test.expectedResizeStatus != resizeStatus {
				t.Errorf("For test %s, expected resizeStatus %v, got %v", test.name, test.expectedResizeStatus, resizeStatus)
			}
			if pvcStatusCap.Cmp(test.expectedStatusSize) != 0 {
				t.Errorf("For test %s, expected status size %s, got %s", test.name, test.expectedStatusSize.String(), pvcStatusCap.String())
			}
		})
	}
}
