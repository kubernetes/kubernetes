/*
Copyright 2018 The Kubernetes Authors.

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

package predicates

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/kubernetes/pkg/features"
)

const (
	ebsCSIDriverName = csilibplugins.AWSEBSDriverName
	gceCSIDriverName = csilibplugins.GCEPDDriverName

	hostpathInTreePluginName = "kubernetes.io/hostpath"
)

func TestCSIVolumeCountPredicate(t *testing.T) {
	runningPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-3",
						},
					},
				},
			},
		},
	}

	pendingVolumePod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-4",
						},
					},
				},
			},
		},
	}

	// Different pod than pendingVolumePod, but using the same unbound PVC
	unboundPVCPod2 := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-4",
						},
					},
				},
			},
		},
	}

	missingPVPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-6",
						},
					},
				},
			},
		},
	}

	noSCPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-5",
						},
					},
				},
			},
		},
	}
	gceTwoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-pd.csi.storage.gke.io-1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-pd.csi.storage.gke.io-2",
						},
					},
				},
			},
		},
	}
	// In-tree volumes
	inTreeOneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-kubernetes.io/aws-ebs-0",
						},
					},
				},
			},
		},
	}
	inTreeTwoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-kubernetes.io/aws-ebs-1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-kubernetes.io/aws-ebs-2",
						},
					},
				},
			},
		},
	}
	// pods with matching csi driver names
	csiEBSOneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-0",
						},
					},
				},
			},
		},
	}
	csiEBSTwoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-1",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-ebs.csi.aws.com-2",
						},
					},
				},
			},
		},
	}
	inTreeNonMigratableOneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "csi-kubernetes.io/hostpath-0",
						},
					},
				},
			},
		},
	}

	tests := []struct {
		newPod                *v1.Pod
		existingPods          []*v1.Pod
		filterName            string
		maxVols               int
		driverNames           []string
		fits                  bool
		test                  string
		migrationEnabled      bool
		limitSource           string
		expectedFailureReason *PredicateFailureError
	}{
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{runningPod, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      4,
			driverNames:  []string{ebsCSIDriverName},
			fits:         true,
			test:         "fits when node volume limit >= new pods CSI volume",
			limitSource:  "node",
		},
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{runningPod, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			fits:         false,
			test:         "doesn't when node volume limit <= pods CSI volume",
			limitSource:  "node",
		},
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{runningPod, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			fits:         true,
			test:         "should when driver does not support volume limits",
			limitSource:  "csinode-with-no-limit",
		},
		// should count pending PVCs
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{pendingVolumePod, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			fits:         false,
			test:         "count pending PVCs towards volume limit <= pods CSI volume",
			limitSource:  "node",
		},
		// two same pending PVCs should be counted as 1
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{pendingVolumePod, unboundPVCPod2, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      4,
			driverNames:  []string{ebsCSIDriverName},
			fits:         true,
			test:         "count multiple pending pvcs towards volume limit >= pods CSI volume",
			limitSource:  "node",
		},
		// should count PVCs with invalid PV name but valid SC
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{missingPVPod, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			fits:         false,
			test:         "should count PVCs with invalid PV name but valid SC",
			limitSource:  "node",
		},
		// don't count a volume which has storageclass missing
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{runningPod, noSCPVCPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName},
			fits:         true,
			test:         "don't count pvcs with missing SC towards volume limit",
			limitSource:  "node",
		},
		// don't count multiple volume types
		{
			newPod:       csiEBSOneVolPod,
			existingPods: []*v1.Pod{gceTwoVolPod, csiEBSTwoVolPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName, gceCSIDriverName},
			fits:         false,
			test:         "count pvcs with the same type towards volume limit",
			limitSource:  "node",
		},
		{
			newPod:       gceTwoVolPod,
			existingPods: []*v1.Pod{csiEBSTwoVolPod, runningPod},
			filterName:   "csi",
			maxVols:      2,
			driverNames:  []string{ebsCSIDriverName, gceCSIDriverName},
			fits:         true,
			test:         "don't count pvcs with different type towards volume limit",
			limitSource:  "node",
		},
		// Tests for in-tree volume migration
		{
			newPod:           inTreeOneVolPod,
			existingPods:     []*v1.Pod{inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             false,
			migrationEnabled: true,
			limitSource:      "csinode",
			test:             "should count in-tree volumes if migration is enabled",
		},
		{
			newPod:           pendingVolumePod,
			existingPods:     []*v1.Pod{inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             false,
			migrationEnabled: true,
			limitSource:      "csinode",
			test:             "should count unbound in-tree volumes if migration is enabled",
		},
		{
			newPod:           inTreeOneVolPod,
			existingPods:     []*v1.Pod{inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             true,
			migrationEnabled: false,
			limitSource:      "csinode",
			test:             "should not count in-tree volume if migration is disabled",
		},
		{
			newPod:           inTreeOneVolPod,
			existingPods:     []*v1.Pod{inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             true,
			migrationEnabled: true,
			limitSource:      "csinode-with-no-limit",
			test:             "should not limit pod if volume used does not report limits",
		},
		{
			newPod:           inTreeOneVolPod,
			existingPods:     []*v1.Pod{inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             true,
			migrationEnabled: false,
			limitSource:      "csinode-with-no-limit",
			test:             "should not limit in-tree pod if migration is disabled",
		},
		{
			newPod:           inTreeNonMigratableOneVolPod,
			existingPods:     []*v1.Pod{csiEBSTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{hostpathInTreePluginName, ebsCSIDriverName},
			fits:             true,
			migrationEnabled: true,
			limitSource:      "csinode",
			test:             "should not count non-migratable in-tree volumes",
		},
		// mixed volumes
		{
			newPod:           inTreeOneVolPod,
			existingPods:     []*v1.Pod{csiEBSTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             false,
			migrationEnabled: true,
			limitSource:      "csinode",
			test:             "should count in-tree and csi volumes if migration is enabled (when scheduling in-tree volumes)",
		},
		{
			newPod:           csiEBSOneVolPod,
			existingPods:     []*v1.Pod{inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             false,
			migrationEnabled: true,
			limitSource:      "csinode",
			test:             "should count in-tree and csi volumes if migration is enabled (when scheduling csi volumes)",
		},
		{
			newPod:           csiEBSOneVolPod,
			existingPods:     []*v1.Pod{csiEBSTwoVolPod, inTreeTwoVolPod},
			filterName:       "csi",
			maxVols:          3,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             true,
			migrationEnabled: false,
			limitSource:      "csinode",
			test:             "should not count in-tree and count csi volumes if migration is disabled (when scheduling csi volumes)",
		},
		{
			newPod:           inTreeOneVolPod,
			existingPods:     []*v1.Pod{csiEBSTwoVolPod},
			filterName:       "csi",
			maxVols:          2,
			driverNames:      []string{csilibplugins.AWSEBSInTreePluginName, ebsCSIDriverName},
			fits:             true,
			migrationEnabled: false,
			limitSource:      "csinode",
			test:             "should not count in-tree and count csi volumes if migration is disabled (when scheduling in-tree volumes)",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AttachVolumeLimit, true)()
	// running attachable predicate tests with feature gate and limit present on nodes
	for _, test := range tests {
		t.Run(test.test, func(t *testing.T) {
			node, csiNode := getNodeWithPodAndVolumeLimits(test.limitSource, test.existingPods, int64(test.maxVols), test.driverNames...)
			if test.migrationEnabled {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigrationAWS, true)()
				enableMigrationOnNode(csiNode, csilibplugins.AWSEBSInTreePluginName)
			} else {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, false)()
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigrationAWS, false)()
			}

			expectedFailureReasons := []PredicateFailureReason{ErrMaxVolumeCountExceeded}
			if test.expectedFailureReason != nil {
				expectedFailureReasons = []PredicateFailureReason{test.expectedFailureReason}
			}

			pred := NewCSIMaxVolumeLimitPredicate(getFakeCSINodeInfo(csiNode),
				getFakeCSIPVInfo(test.filterName, test.driverNames...),
				getFakeCSIPVCInfo(test.filterName, "csi-sc", test.driverNames...),
				getFakeCSIStorageClassInfo("csi-sc", test.driverNames[0]))

			fits, reasons, err := pred(test.newPod, GetPredicateMetadata(test.newPod, nil), node)
			if err != nil {
				t.Errorf("Using allocatable [%s]%s: unexpected error: %v", test.filterName, test.test, err)
			}
			if !fits && !reflect.DeepEqual(expectedFailureReasons, reasons) {
				t.Errorf("Using allocatable [%s]%s: unexpected failure reasons: %v, want: %v", test.filterName, test.test, reasons, expectedFailureReasons)
			}
			if fits != test.fits {
				t.Errorf("Using allocatable [%s]%s: expected %v, got %v", test.filterName, test.test, test.fits, fits)
			}
		})
	}
}

func getFakeCSIPVInfo(volumeName string, driverNames ...string) FakePersistentVolumeInfo {
	pvInfos := FakePersistentVolumeInfo{}
	for _, driver := range driverNames {
		for j := 0; j < 4; j++ {
			volumeHandle := fmt.Sprintf("%s-%s-%d", volumeName, driver, j)
			pv := v1.PersistentVolume{
				ObjectMeta: metav1.ObjectMeta{Name: volumeHandle},
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       driver,
							VolumeHandle: volumeHandle,
						},
					},
				},
			}

			switch driver {
			case csilibplugins.AWSEBSInTreePluginName:
				pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
						VolumeID: volumeHandle,
					},
				}
			case hostpathInTreePluginName:
				pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
					HostPath: &v1.HostPathVolumeSource{
						Path: "/tmp",
					},
				}
			default:
				pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
					CSI: &v1.CSIPersistentVolumeSource{
						Driver:       driver,
						VolumeHandle: volumeHandle,
					},
				}
			}
			pvInfos = append(pvInfos, pv)
		}

	}
	return pvInfos
}

func getFakeCSIPVCInfo(volumeName, scName string, driverNames ...string) FakePersistentVolumeClaimInfo {
	pvcInfos := FakePersistentVolumeClaimInfo{}
	for _, driver := range driverNames {
		for j := 0; j < 4; j++ {
			v := fmt.Sprintf("%s-%s-%d", volumeName, driver, j)
			pvc := v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: v},
				Spec:       v1.PersistentVolumeClaimSpec{VolumeName: v},
			}
			pvcInfos = append(pvcInfos, pvc)
		}
	}

	pvcInfos = append(pvcInfos, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-4"},
		Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &scName},
	})
	pvcInfos = append(pvcInfos, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-5"},
		Spec:       v1.PersistentVolumeClaimSpec{},
	})
	// a pvc with missing PV but available storageclass.
	pvcInfos = append(pvcInfos, v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: volumeName + "-6"},
		Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &scName, VolumeName: "missing-in-action"},
	})
	return pvcInfos
}

func enableMigrationOnNode(csiNode *storagev1beta1.CSINode, pluginName string) {
	nodeInfoAnnotations := csiNode.GetAnnotations()
	if nodeInfoAnnotations == nil {
		nodeInfoAnnotations = map[string]string{}
	}

	newAnnotationSet := sets.NewString()
	newAnnotationSet.Insert(pluginName)
	nas := strings.Join(newAnnotationSet.List(), ",")
	nodeInfoAnnotations[v1.MigratedPluginsAnnotationKey] = nas

	csiNode.Annotations = nodeInfoAnnotations
}

func getFakeCSIStorageClassInfo(scName, provisionerName string) FakeStorageClassInfo {
	return FakeStorageClassInfo{
		{
			ObjectMeta:  metav1.ObjectMeta{Name: scName},
			Provisioner: provisionerName,
		},
	}
}

func getFakeCSINodeInfo(csiNode *storagev1beta1.CSINode) FakeCSINodeInfo {
	if csiNode != nil {
		return FakeCSINodeInfo(*csiNode)
	}
	return FakeCSINodeInfo{}
}
