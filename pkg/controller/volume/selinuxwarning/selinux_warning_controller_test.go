/*
Copyright 2024 The Kubernetes Authors.

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

package selinuxwarning

import (
	"context"
	"reflect"
	"sort"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	volumecache "k8s.io/kubernetes/pkg/controller/volume/selinuxwarning/cache"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/ptr"
)

const (
	namespace = "ns1"
	pvcUID    = "uid1"
)

func TestSELinuxWarningController_Sync(t *testing.T) {
	tests := []struct {
		name               string
		existingPVCs       []*v1.PersistentVolumeClaim
		existingPVs        []*v1.PersistentVolume
		existingCSIDrivers []*storagev1.CSIDriver
		existingPods       []*v1.Pod

		pod                     cache.ObjectName
		csiDriverSELinuxEnabled bool
		conflicts               []volumecache.Conflict
		expectError             bool
		expectedAddedVolumes    []addedVolume
		expectedEvents          []string
		expectedDeletedPods     []cache.ObjectName
	}{
		{
			name: "existing pod with no volumes",
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes:    nil,
		},
		{
			name: "existing pod with unbound PVC",
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("non-existing-pvc", "vol1").build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectError:             true, // PVC is missing, add back to queue with exp. backoff
			expectedEvents:          nil,
			expectedAddedVolumes:    nil,
		},
		{
			name: "existing pod with fully bound PVC",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
			},
		},
		{
			name: "existing pod with fully bound PVC, Recursive change policy",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", ptr.To(v1.SELinuxChangePolicyRecursive)).withPVC("pvc1", "vol1").build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        "", // Label is cleared with the Recursive policy
					changePolicy: v1.SELinuxChangePolicyRecursive,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
			},
		},
		{
			name: "existing pod with inline volume",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withInlineVolume().build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/ebs.csi.aws.com-inlinevol1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com", // The inline volume is AWS EBS
				},
			},
		},
		{
			name: "existing pod with inline volume and PVC",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").withInlineVolume().build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
				{
					volumeName:   "fake-plugin/ebs.csi.aws.com-inlinevol1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com", // The inline volume is AWS EBS
				},
			},
		},
		{
			name: "existing pod with PVC generates conflict, the other pod exists",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").build(),
				pod("pod2", "s0:c98,c99", nil).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			conflicts: []volumecache.Conflict{
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: namespace, Name: "pod1"},
					PropertyValue:      ":::s0:c1,c2",
					OtherPod:           cache.ObjectName{Namespace: namespace, Name: "pod2"},
					OtherPropertyValue: ":::s0:c98,c99",
				},
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: namespace, Name: "pod2"},
					PropertyValue:      ":::s0:c98,c99",
					OtherPod:           cache.ObjectName{Namespace: namespace, Name: "pod1"},
					OtherPropertyValue: ":::s0:c1,c2",
				},
			},
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
			},
			expectedEvents: []string{
				`Normal SELinuxLabelConflict SELinuxLabel ":::s0:c1,c2" conflicts with pod pod2 that uses the same volume as this pod with SELinuxLabel ":::s0:c98,c99". If both pods land on the same node, only one of them may access the volume.`,
				`Normal SELinuxLabelConflict SELinuxLabel ":::s0:c98,c99" conflicts with pod pod1 that uses the same volume as this pod with SELinuxLabel ":::s0:c1,c2". If both pods land on the same node, only one of them may access the volume.`,
			},
		},
		{
			name: "existing pod with Recursive policy does not generate conflicts",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", ptr.To(v1.SELinuxChangePolicyRecursive)).withPVC("pvc1", "vol1").build(),
				pod("pod2", "s0:c98,c99", ptr.To(v1.SELinuxChangePolicyRecursive)).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			conflicts:               []volumecache.Conflict{},
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        "", // Label is cleared with the Recursive policy
					changePolicy: v1.SELinuxChangePolicyRecursive,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
			},
		},
		{
			name: "existing pod with Recursive policy does not conflict with pod with MountOption policy label, only with the policy",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", ptr.To(v1.SELinuxChangePolicyRecursive)).withPVC("pvc1", "vol1").build(),
				pod("pod2", "s0:c98,c99", ptr.To(v1.SELinuxChangePolicyMountOption)).withPVC("pvc1", "vol1").build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			conflicts: []volumecache.Conflict{
				{
					PropertyName:       "SELinuxChangePolicy",
					EventReason:        "SELinuxChangePolicyConflict",
					Pod:                cache.ObjectName{Namespace: namespace, Name: "pod1"},
					PropertyValue:      string(v1.SELinuxChangePolicyRecursive),
					OtherPod:           cache.ObjectName{Namespace: namespace, Name: "pod2"},
					OtherPropertyValue: string(v1.SELinuxChangePolicyMountOption),
				},
				{
					PropertyName:       "SELinuxChangePolicy",
					EventReason:        "SELinuxChangePolicyConflict",
					Pod:                cache.ObjectName{Namespace: namespace, Name: "pod2"},
					PropertyValue:      string(v1.SELinuxChangePolicyMountOption),
					OtherPod:           cache.ObjectName{Namespace: namespace, Name: "pod1"},
					OtherPropertyValue: string(v1.SELinuxChangePolicyRecursive),
				},
			},
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        "", // Label is cleared with the Recursive policy
					changePolicy: v1.SELinuxChangePolicyRecursive,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
			},
			expectedEvents: []string{
				`Normal SELinuxChangePolicyConflict SELinuxChangePolicy "Recursive" conflicts with pod pod2 that uses the same volume as this pod with SELinuxChangePolicy "MountOption". If both pods land on the same node, only one of them may access the volume.`,
				`Normal SELinuxChangePolicyConflict SELinuxChangePolicy "MountOption" conflicts with pod pod1 that uses the same volume as this pod with SELinuxChangePolicy "Recursive". If both pods land on the same node, only one of them may access the volume.`,
			},
		},
		{
			name: "existing pod with PVC generates conflict, the other pod doesn't exist",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").build(),
				// "pod2" does not exist
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			conflicts: []volumecache.Conflict{
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: namespace, Name: "pod1"},
					PropertyValue:      ":::s0:c1,c2",
					OtherPod:           cache.ObjectName{Namespace: namespace, Name: "pod2"},
					OtherPropertyValue: ":::s0:c98,c99",
				},
				{
					PropertyName:       "SELinuxLabel",
					EventReason:        "SELinuxLabelConflict",
					Pod:                cache.ObjectName{Namespace: namespace, Name: "pod2"},
					PropertyValue:      ":::s0:c98,c99",
					OtherPod:           cache.ObjectName{Namespace: namespace, Name: "pod1"},
					OtherPropertyValue: ":::s0:c1,c2",
				},
			},
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com", // The PV is a fake EBS volume
				},
			},
			expectedEvents: []string{
				// Event for the missing pod is not sent
				`Normal SELinuxLabelConflict SELinuxLabel ":::s0:c1,c2" conflicts with pod pod2 that uses the same volume as this pod with SELinuxLabel ":::s0:c98,c99". If both pods land on the same node, only one of them may access the volume.`,
			},
		},
		{
			name: "empty label implies Recursive policy",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "", ptr.To(v1.SELinuxChangePolicyMountOption)).withPVC("pvc1", "vol1").build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			conflicts:               []volumecache.Conflict{},
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        "",
					changePolicy: v1.SELinuxChangePolicyRecursive, // Reset to Recursive when the label is empty
					csiDriver:    "ebs.csi.aws.com",
				},
			},
		},
		{
			name: "pending pod is processed",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").withPhase(v1.PodPending).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com",
				},
			},
		},
		{
			name: "unknown pod is processed",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").withPhase(v1.PodUnknown).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        ":::s0:c1,c2",
					changePolicy: v1.SELinuxChangePolicyMountOption,
					csiDriver:    "ebs.csi.aws.com",
				},
			},
		},
		{
			name: "succeeded pod is removed from the cache",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").withPhase(v1.PodSucceeded).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes:    nil,
			expectedDeletedPods:     []cache.ObjectName{{Namespace: namespace, Name: "pod1"}},
		},
		{
			name: "failed pod is removed from the cache",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").withPhase(v1.PodFailed).build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectedEvents:          nil,
			expectedAddedVolumes:    nil,
			expectedDeletedPods:     []cache.ObjectName{{Namespace: namespace, Name: "pod1"}},
		},
		{
			name:         "deleted pod",
			existingPods: []*v1.Pod{
				// "pod1" does not exist in the informer
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: true,
			expectError:             false,
			expectedEvents:          nil,
			expectedAddedVolumes:    nil,
			expectedDeletedPods:     []cache.ObjectName{{Namespace: namespace, Name: "pod1"}},
		},
		{
			name: "existing pod with fully bound PVC and CSIDriver.SELinuxMount disabled",
			existingPVCs: []*v1.PersistentVolumeClaim{
				pvcBoundToPV("pv1", "pvc1"),
			},
			existingPVs: []*v1.PersistentVolume{
				pvBoundToPVC("pv1", "pvc1"),
			},
			existingPods: []*v1.Pod{
				pod("pod1", "s0:c1,c2", nil).withPVC("pvc1", "vol1").build(),
			},
			pod:                     cache.ObjectName{Namespace: namespace, Name: "pod1"},
			csiDriverSELinuxEnabled: false,
			expectedEvents:          nil,
			expectedAddedVolumes: []addedVolume{
				{
					volumeName:   "fake-plugin/pv1",
					podKey:       cache.ObjectName{Namespace: namespace, Name: "pod1"},
					label:        "",                              // Label is cleared when the CSI driver does not support SELinuxMount
					changePolicy: v1.SELinuxChangePolicyRecursive, // Reset to Recursive when the CSI driver does not support SELinuxMount
					csiDriver:    "ebs.csi.aws.com",               // The PV is a fake EBS volume
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxChangePolicy, true)

			var wg sync.WaitGroup
			defer wg.Wait()
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			_, plugin := volumetesting.GetTestKubeletVolumePluginMgr(t)
			plugin.SupportsSELinux = tt.csiDriverSELinuxEnabled

			fakeClient := fake.NewClientset()
			fakeInformerFactory := informers.NewSharedInformerFactory(fakeClient, controller.NoResyncPeriodFunc())
			podInformer := fakeInformerFactory.Core().V1().Pods()
			pvcInformer := fakeInformerFactory.Core().V1().PersistentVolumeClaims()
			pvInformer := fakeInformerFactory.Core().V1().PersistentVolumes()
			csiDriverInformer := fakeInformerFactory.Storage().V1().CSIDrivers()

			c, err := NewController(
				ctx,
				fakeClient,
				podInformer,
				pvcInformer,
				pvInformer,
				csiDriverInformer,
				[]volume.VolumePlugin{plugin},
				nil,
			)
			if err != nil {
				t.Fatalf("failed to create controller: %v", err)
			}
			// Use a fake volume cache
			labelCache := &fakeVolumeCache{
				conflictsToSend: map[cache.ObjectName][]volumecache.Conflict{
					{Namespace: tt.pod.Namespace, Name: tt.pod.Name}: tt.conflicts,
				},
			}
			c.labelCache = labelCache
			fakeRecorder := record.NewFakeRecorder(10)
			c.eventRecorder = fakeRecorder

			// Start the informers
			fakeInformerFactory.Start(ctx.Done())
			fakeInformerFactory.WaitForCacheSync(ctx.Done())
			// Start the controller
			wg.Go(func() {
				c.Run(ctx, 1)
			})

			// Inject fake existing objects
			for _, pvc := range tt.existingPVCs {
				_ = pvcInformer.Informer().GetStore().Add(pvc)
			}
			for _, pv := range tt.existingPVs {
				_ = pvInformer.Informer().GetStore().Add(pv)
			}
			for _, pod := range tt.existingPods {
				_ = podInformer.Informer().GetStore().Add(pod)
			}

			// Act: call sync() on the pod that *is* in the informer cache
			err = c.sync(ctx, tt.pod)

			// Assert:
			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return // do not check the rest on error
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			sortAddedVolumes(tt.expectedAddedVolumes)
			sortAddedVolumes(labelCache.addedVolumes)
			if !reflect.DeepEqual(tt.expectedAddedVolumes, labelCache.addedVolumes) {
				t.Errorf("unexpected added volumes, expected \n%+v\ngot\n%+v", tt.expectedAddedVolumes, labelCache.addedVolumes)
			}

			events := collectEvents(fakeRecorder.Events)
			receivedSet := sets.New(events...)
			expectedSet := sets.New(tt.expectedEvents...)
			if !receivedSet.Equal(expectedSet) {
				t.Errorf("got unexpected events: %+v", receivedSet.Difference(expectedSet))
				t.Errorf("missing events: %+v", expectedSet.Difference(receivedSet))
			}

			if !reflect.DeepEqual(tt.expectedDeletedPods, labelCache.deletedPods) {
				t.Errorf("unexpected deleted pods, expected \n%+v\ngot\n%+v", tt.expectedDeletedPods, labelCache.deletedPods)
			}
		})
	}
}

func pv(name string) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					VolumeID: name,
				},
			},
		},
	}
}

func pvc(name string) *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       pvcUID,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}
}

func pvBoundToPVC(pvName, pvcName string) *v1.PersistentVolume {
	pv := pv(pvName)
	pv.Spec.ClaimRef = &v1.ObjectReference{
		Kind:       "PersistentVolumeClaim",
		Namespace:  namespace,
		Name:       pvcName,
		UID:        pvcUID,
		APIVersion: "v1",
	}
	pv.Status.Phase = v1.VolumeBound
	return pv
}

func pvcBoundToPV(pvName, pvcName string) *v1.PersistentVolumeClaim {
	pvc := pvc(pvcName)
	pvc.Spec.VolumeName = pvName
	pvc.Status.Phase = v1.ClaimBound

	return pvc
}

type podBuilder struct {
	pod *v1.Pod
}

func pod(podName, level string, changePolicy *v1.PodSELinuxChangePolicy) *podBuilder {
	var opts *v1.SELinuxOptions
	if level != "" {
		opts = &v1.SELinuxOptions{
			Level: level,
		}
	}
	return &podBuilder{
		pod: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "ns1",
				Name:      podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container1",
						Image: "image1",
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol1",
								MountPath: "/mnt",
							},
						},
					},
				},
				SecurityContext: &v1.PodSecurityContext{
					SELinuxChangePolicy: changePolicy,
					SELinuxOptions:      opts,
				},
				Volumes: []v1.Volume{
					{
						Name: "emptyDir1",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
			Status: v1.PodStatus{
				Phase: v1.PodRunning,
			},
		},
	}
}

func (b *podBuilder) withPhase(phase v1.PodPhase) *podBuilder {
	b.pod.Status.Phase = phase
	return b
}

func (b *podBuilder) withInlineVolume() *podBuilder {
	b.pod.Spec.Volumes = append(b.pod.Spec.Volumes, v1.Volume{
		Name: "inlineVolume",
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
				VolumeID: "inlinevol1",
			},
		},
	})
	b.pod.Spec.Containers[0].VolumeMounts = append(b.pod.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
		Name:      "inlineVolume",
		MountPath: "/mnt",
	})
	return b
}

func (b *podBuilder) withPVC(pvcName, volumeName string) *podBuilder {
	b.pod.Spec.Volumes = append(b.pod.Spec.Volumes, v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
			},
		},
	})
	b.pod.Spec.Containers[0].VolumeMounts = append(b.pod.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
		Name:      volumeName,
		MountPath: "/mnt",
	})
	return b
}

func (b *podBuilder) build() *v1.Pod {
	return b.pod
}

type addedVolume struct {
	volumeName   v1.UniqueVolumeName
	podKey       cache.ObjectName
	label        string
	changePolicy v1.PodSELinuxChangePolicy
	csiDriver    string
}

func sortAddedVolumes(a []addedVolume) {
	sort.Slice(a, func(i, j int) bool {
		ikey := string(a[i].volumeName) + "/" + a[i].podKey.String()
		jkey := string(a[j].volumeName) + "/" + a[j].podKey.String()
		return ikey < jkey
	})
}

type fakeVolumeCache struct {
	addedVolumes []addedVolume
	// Conflicts to send when AddPod with given pod name is called.
	conflictsToSend map[cache.ObjectName][]volumecache.Conflict
	deletedPods     []cache.ObjectName
}

var _ volumecache.VolumeCache = &fakeVolumeCache{}

func (f *fakeVolumeCache) AddVolume(logger klog.Logger, volumeName v1.UniqueVolumeName, podKey cache.ObjectName, label string, changePolicy v1.PodSELinuxChangePolicy, csiDriver string) []volumecache.Conflict {
	f.addedVolumes = append(f.addedVolumes, addedVolume{
		volumeName:   volumeName,
		podKey:       podKey,
		label:        label,
		changePolicy: changePolicy,
		csiDriver:    csiDriver,
	})
	conflicts := f.conflictsToSend[podKey]
	return conflicts
}

func (f *fakeVolumeCache) DeletePod(logger klog.Logger, podKey cache.ObjectName) {
	f.deletedPods = append(f.deletedPods, podKey)
}

func (f *fakeVolumeCache) GetPodsForCSIDriver(driverName string) []cache.ObjectName {
	pods := []cache.ObjectName{}
	for _, v := range f.addedVolumes {
		if v.csiDriver == driverName {
			pods = append(pods, v.podKey)
		}
	}
	return pods
}

func (f *fakeVolumeCache) SendConflicts(logger klog.Logger, ch chan<- volumecache.Conflict) {
	for _, conflicts := range f.conflictsToSend {
		for _, conflict := range conflicts {
			ch <- conflict
		}
	}
}

func collectEvents(source <-chan string) []string {
	done := false
	events := make([]string, 0)
	for !done {
		select {
		case event := <-source:
			events = append(events, event)
		default:
			done = true
		}
	}
	return events
}
