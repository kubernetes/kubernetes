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

package nodevolumelimits

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	fakeframework "k8s.io/kubernetes/pkg/scheduler/framework/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/pointer"
)

var (
	oneVolPod = st.MakePod().Volume(v1.Volume{
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp"},
		},
	}).Obj()
	twoVolPod = st.MakePod().Volume(v1.Volume{
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp1"},
		},
	}).Volume(v1.Volume{
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp2"},
		},
	}).Obj()
	splitVolsPod = st.MakePod().Volume(v1.Volume{
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{},
		},
	}).Volume(v1.Volume{
		VolumeSource: v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "svp"},
		},
	}).Obj()
	nonApplicablePod = st.MakePod().Volume(v1.Volume{
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{},
		},
	}).Obj()

	deletedPVCPod    = st.MakePod().PVC("deletedPVC").Obj()
	twoDeletedPVCPod = st.MakePod().PVC("deletedPVC").PVC("anotherDeletedPVC").Obj()
	deletedPVPod     = st.MakePod().PVC("pvcWithDeletedPV").Obj()
	// deletedPVPod2 is a different pod than deletedPVPod but using the same PVC
	deletedPVPod2       = st.MakePod().PVC("pvcWithDeletedPV").Obj()
	anotherDeletedPVPod = st.MakePod().PVC("anotherPVCWithDeletedPV").Obj()
	emptyPod            = st.MakePod().Obj()
	unboundPVCPod       = st.MakePod().PVC("unboundPVC").Obj()
	// Different pod than unboundPVCPod, but using the same unbound PVC
	unboundPVCPod2 = st.MakePod().PVC("unboundPVC").Obj()
	// pod with unbound PVC that's different to unboundPVC
	anotherUnboundPVCPod = st.MakePod().PVC("anotherUnboundPVC").Obj()
)

func TestEphemeralLimits(t *testing.T) {
	// We have to specify a valid filter and arbitrarily pick Cinder here.
	// It doesn't matter for the test cases.
	filterName := gcePDVolumeFilterType
	driverName := csilibplugins.GCEPDInTreePluginName

	ephemeralVolumePod := st.MakePod().Name("abc").Namespace("test").UID("12345").Volume(v1.Volume{
		Name: "xyz",
		VolumeSource: v1.VolumeSource{
			Ephemeral: &v1.EphemeralVolumeSource{},
		},
	}).Obj()

	controller := true
	ephemeralClaim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ephemeralVolumePod.Namespace,
			Name:      ephemeralVolumePod.Name + "-" + ephemeralVolumePod.Spec.Volumes[0].Name,
			OwnerReferences: []metav1.OwnerReference{
				{
					Kind:       "Pod",
					Name:       ephemeralVolumePod.Name,
					UID:        ephemeralVolumePod.UID,
					Controller: &controller,
				},
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			VolumeName:       "missing",
			StorageClassName: &filterName,
		},
	}
	conflictingClaim := ephemeralClaim.DeepCopy()
	conflictingClaim.OwnerReferences = nil

	tests := []struct {
		newPod           *v1.Pod
		existingPods     []*v1.Pod
		extraClaims      []v1.PersistentVolumeClaim
		ephemeralEnabled bool
		maxVols          int
		test             string
		wantStatus       *framework.Status
	}{
		{
			newPod:           ephemeralVolumePod,
			ephemeralEnabled: true,
			test:             "volume missing",
			wantStatus:       framework.NewStatus(framework.Error, `looking up PVC test/abc-xyz: persistentvolumeclaim "abc-xyz" not found`),
		},
		{
			newPod:           ephemeralVolumePod,
			ephemeralEnabled: true,
			extraClaims:      []v1.PersistentVolumeClaim{*conflictingClaim},
			test:             "volume not owned",
			wantStatus:       framework.NewStatus(framework.Error, "PVC test/abc-xyz was not created for pod test/abc (pod is not owner)"),
		},
		{
			newPod:           ephemeralVolumePod,
			ephemeralEnabled: true,
			extraClaims:      []v1.PersistentVolumeClaim{*ephemeralClaim},
			maxVols:          1,
			test:             "volume unbound, allowed",
		},
		{
			newPod:           ephemeralVolumePod,
			ephemeralEnabled: true,
			extraClaims:      []v1.PersistentVolumeClaim{*ephemeralClaim},
			maxVols:          0,
			test:             "volume unbound, exceeds limit",
			wantStatus:       framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			fts := feature.Features{}
			node, csiNode := getNodeWithPodAndVolumeLimits("node", test.existingPods, int64(test.maxVols), filterName)
			p := newNonCSILimits(filterName, getFakeCSINodeLister(csiNode), getFakeCSIStorageClassLister(filterName, driverName), getFakePVLister(filterName), append(getFakePVCLister(filterName), test.extraClaims...), fts).(framework.FilterPlugin)
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestAzureDiskLimits(t *testing.T) {
	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		filterName   string
		driverName   string
		maxVols      int
		test         string
		wantStatus   *framework.Status
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      4,
			test:         "fits when node capacity >= new pod's AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "fit when node capacity < new pod's AzureDisk volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "new pod's count ignores non-AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "existing pods' counts ignore non-AzureDisk volumes",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "new pod's count considers PVCs backed by AzureDisk volumes",
		},
		{
			newPod:       splitPVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "new pod's count ignores PVCs not backed by AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(azureDiskVolumeFilterType)},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "existing pods' counts considers PVCs backed by AzureDisk volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(azureDiskVolumeFilterType)},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      4,
			test:         "already-mounted AzureDisk volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(azureDiskVolumeFilterType)},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "the same AzureDisk volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "pod with missing two PVCs is counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(azureDiskVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      3,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   azureDiskVolumeFilterType,
			maxVols:      2,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits("node", test.existingPods, int64(test.maxVols), test.filterName)
			p := newNonCSILimits(test.filterName, getFakeCSINodeLister(csiNode), getFakeCSIStorageClassLister(test.filterName, test.driverName), getFakePVLister(test.filterName), getFakePVCLister(test.filterName), feature.Features{}).(framework.FilterPlugin)
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestEBSLimits(t *testing.T) {
	unboundPVCWithInvalidSCPod := st.MakePod().PVC("unboundPVCWithInvalidSCPod").Obj()
	unboundPVCWithDefaultSCPod := st.MakePod().PVC("unboundPVCWithDefaultSCPod").Obj()

	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		filterName   string
		driverName   string
		maxVols      int
		test         string
		wantStatus   *framework.Status
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      4,
			test:         "fits when node capacity >= new pod's EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "doesn't fit when node capacity < new pod's EBS volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "new pod's count ignores non-EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "existing pods' counts ignore non-EBS volumes",
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "new pod's count considers PVCs backed by EBS volumes",
		},
		{
			newPod:       splitPVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "new pod's count ignores PVCs not backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(ebsVolumeFilterType)},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "existing pods' counts considers PVCs backed by EBS volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(ebsVolumeFilterType)},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      4,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(ebsVolumeFilterType)},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "the same EBS volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      1,
			test:         "missing PVC is not counted towards the PV limit",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "missing PVC is not counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two missing PVCs are not counted towards the PV limit twice",
		},
		{
			newPod:       unboundPVCWithInvalidSCPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      1,
			test:         "unbound PVC with invalid SC is not counted towards the PV limit",
		},
		{
			newPod:       unboundPVCWithDefaultSCPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      1,
			test:         "unbound PVC from different provisioner is not counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "pod with missing PV is counted towards the PV limit",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "pod with unbound PVC is counted towards the PV limit",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
		{
			newPod:       onePVCPod(ebsVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      3,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   ebsVolumeFilterType,
			driverName:   csilibplugins.AWSEBSInTreePluginName,
			maxVols:      2,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
			wantStatus:   framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded),
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits("node", test.existingPods, int64(test.maxVols), test.filterName)
			p := newNonCSILimits(test.filterName, getFakeCSINodeLister(csiNode), getFakeCSIStorageClassLister(test.filterName, test.driverName), getFakePVLister(test.filterName), getFakePVCLister(test.filterName), feature.Features{}).(framework.FilterPlugin)
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestGCEPDLimits(t *testing.T) {
	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		filterName   string
		driverName   string
		maxVols      int
		test         string
		wantStatus   *framework.Status
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      4,
			test:         "fits when node capacity >= new pod's GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "fit when node capacity < new pod's GCE volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "new pod's count ignores non-GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "existing pods' counts ignore non-GCE volumes",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "new pod's count considers PVCs backed by GCE volumes",
		},
		{
			newPod:       splitPVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "new pod's count ignores PVCs not backed by GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, onePVCPod(gcePDVolumeFilterType)},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "existing pods' counts considers PVCs backed by GCE volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, onePVCPod(gcePDVolumeFilterType)},
			filterName:   gcePDVolumeFilterType,
			maxVols:      4,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, onePVCPod(gcePDVolumeFilterType)},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "the same GCE volumes are not counted multiple times",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "pod with missing two PVCs is counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       onePVCPod(gcePDVolumeFilterType),
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      3,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			filterName:   gcePDVolumeFilterType,
			maxVols:      2,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
		},
	}

	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits("node", test.existingPods, int64(test.maxVols), test.filterName)
			p := newNonCSILimits(test.filterName, getFakeCSINodeLister(csiNode), getFakeCSIStorageClassLister(test.filterName, test.driverName), getFakePVLister(test.filterName), getFakePVCLister(test.filterName), feature.Features{}).(framework.FilterPlugin)
			gotStatus := p.Filter(context.Background(), nil, test.newPod, node)
			if !reflect.DeepEqual(gotStatus, test.wantStatus) {
				t.Errorf("status does not match: %v, want: %v", gotStatus, test.wantStatus)
			}
		})
	}
}

func TestGetMaxVols(t *testing.T) {
	previousValue := os.Getenv(KubeMaxPDVols)

	tests := []struct {
		rawMaxVols string
		expected   int
		name       string
	}{
		{
			rawMaxVols: "invalid",
			expected:   -1,
			name:       "Unable to parse maximum PD volumes value, using default value",
		},
		{
			rawMaxVols: "-2",
			expected:   -1,
			name:       "Maximum PD volumes must be a positive value, using default value",
		},
		{
			rawMaxVols: "40",
			expected:   40,
			name:       "Parse maximum PD volumes value from env",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			os.Setenv(KubeMaxPDVols, test.rawMaxVols)
			result := getMaxVolLimitFromEnv()
			if result != test.expected {
				t.Errorf("expected %v got %v", test.expected, result)
			}
		})
	}

	os.Unsetenv(KubeMaxPDVols)
	if previousValue != "" {
		os.Setenv(KubeMaxPDVols, previousValue)
	}
}

func getFakePVCLister(filterName string) fakeframework.PersistentVolumeClaimLister {
	return fakeframework.PersistentVolumeClaimLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "some" + filterName + "Vol"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "some" + filterName + "Vol",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNon" + filterName + "Vol"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "someNon" + filterName + "Vol",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pvcWithDeletedPV"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "pvcWithDeletedPV",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherPVCWithDeletedPV"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "anotherPVCWithDeletedPV",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVC"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherUnboundPVC"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: &filterName,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVCWithDefaultSCPod"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: pointer.String("standard-sc"),
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVCWithInvalidSCPod"},
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeName:       "",
				StorageClassName: pointer.String("invalid-sc"),
			},
		},
	}
}

func getFakePVLister(filterName string) fakeframework.PersistentVolumeLister {
	return fakeframework.PersistentVolumeLister{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "some" + filterName + "Vol"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: strings.ToLower(filterName) + "Vol"},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNon" + filterName + "Vol"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{},
			},
		},
	}
}

func onePVCPod(filterName string) *v1.Pod {
	return st.MakePod().PVC(fmt.Sprintf("some%sVol", filterName)).Obj()
}

func splitPVCPod(filterName string) *v1.Pod {
	return st.MakePod().PVC(fmt.Sprintf("someNon%sVol", filterName)).PVC(fmt.Sprintf("some%sVol", filterName)).Obj()
}
