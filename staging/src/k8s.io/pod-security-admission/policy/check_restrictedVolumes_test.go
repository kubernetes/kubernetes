/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestRestrictedVolumes(t *testing.T) {
	tests := []struct {
		name         string
		pod          *corev1.Pod
		expectReason string
		expectDetail string
	}{
		{
			name: "host path volumes",
			pod: &corev1.Pod{Spec: corev1.PodSpec{
				Volumes: []corev1.Volume{
					// allowed types
					{Name: "a1", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
					{Name: "a2", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{}}},
					{Name: "a3", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{}}},
					{Name: "a4", VolumeSource: corev1.VolumeSource{DownwardAPI: &corev1.DownwardAPIVolumeSource{}}},
					{Name: "a5", VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{}}},
					{Name: "a6", VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{}}},
					{Name: "a7", VolumeSource: corev1.VolumeSource{CSI: &corev1.CSIVolumeSource{}}},
					{Name: "a8", VolumeSource: corev1.VolumeSource{Ephemeral: &corev1.EphemeralVolumeSource{}}},
					{Name: "a9", VolumeSource: corev1.VolumeSource{Image: &corev1.ImageVolumeSource{}}},

					// known restricted types
					{Name: "b1", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{}}},
					{Name: "b2", VolumeSource: corev1.VolumeSource{GCEPersistentDisk: &corev1.GCEPersistentDiskVolumeSource{}}},
					{Name: "b3", VolumeSource: corev1.VolumeSource{AWSElasticBlockStore: &corev1.AWSElasticBlockStoreVolumeSource{}}},
					{Name: "b4", VolumeSource: corev1.VolumeSource{GitRepo: &corev1.GitRepoVolumeSource{}}},
					{Name: "b5", VolumeSource: corev1.VolumeSource{NFS: &corev1.NFSVolumeSource{}}},
					{Name: "b6", VolumeSource: corev1.VolumeSource{ISCSI: &corev1.ISCSIVolumeSource{}}},
					{Name: "b7", VolumeSource: corev1.VolumeSource{Glusterfs: &corev1.GlusterfsVolumeSource{}}},
					{Name: "b8", VolumeSource: corev1.VolumeSource{RBD: &corev1.RBDVolumeSource{}}},
					{Name: "b9", VolumeSource: corev1.VolumeSource{FlexVolume: &corev1.FlexVolumeSource{}}},
					{Name: "b10", VolumeSource: corev1.VolumeSource{Cinder: &corev1.CinderVolumeSource{}}},
					{Name: "b11", VolumeSource: corev1.VolumeSource{CephFS: &corev1.CephFSVolumeSource{}}},
					{Name: "b12", VolumeSource: corev1.VolumeSource{Flocker: &corev1.FlockerVolumeSource{}}},
					{Name: "b13", VolumeSource: corev1.VolumeSource{FC: &corev1.FCVolumeSource{}}},
					{Name: "b14", VolumeSource: corev1.VolumeSource{AzureFile: &corev1.AzureFileVolumeSource{}}},
					{Name: "b15", VolumeSource: corev1.VolumeSource{VsphereVolume: &corev1.VsphereVirtualDiskVolumeSource{}}},
					{Name: "b16", VolumeSource: corev1.VolumeSource{Quobyte: &corev1.QuobyteVolumeSource{}}},
					{Name: "b17", VolumeSource: corev1.VolumeSource{AzureDisk: &corev1.AzureDiskVolumeSource{}}},
					{Name: "b18", VolumeSource: corev1.VolumeSource{PhotonPersistentDisk: &corev1.PhotonPersistentDiskVolumeSource{}}},
					{Name: "b19", VolumeSource: corev1.VolumeSource{PortworxVolume: &corev1.PortworxVolumeSource{}}},
					{Name: "b20", VolumeSource: corev1.VolumeSource{ScaleIO: &corev1.ScaleIOVolumeSource{}}},
					{Name: "b21", VolumeSource: corev1.VolumeSource{StorageOS: &corev1.StorageOSVolumeSource{}}},

					// unknown type
					{Name: "c1", VolumeSource: corev1.VolumeSource{}},
				},
			}},
			expectReason: `restricted volume types`,
			expectDetail: `volumes ` +
				`"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12", "b13", "b14", "b15", "b16", "b17", "b18", "b19", "b20", "b21", "c1"` +
				` use restricted volume types ` +
				`"awsElasticBlockStore", "azureDisk", "azureFile", "cephfs", "cinder", "fc", "flexVolume", "flocker", "gcePersistentDisk", "gitRepo", "glusterfs", ` +
				`"hostPath", "iscsi", "nfs", "photonPersistentDisk", "portworxVolume", "quobyte", "rbd", "scaleIO", "storageos", "unknown", "vsphereVolume"`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := restrictedVolumes_1_0(&tc.pod.ObjectMeta, &tc.pod.Spec)
			if result.Allowed {
				t.Fatal("expected disallowed")
			}
			if e, a := tc.expectReason, result.ForbiddenReason; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
			if e, a := tc.expectDetail, result.ForbiddenDetail; e != a {
				t.Errorf("expected\n%s\ngot\n%s", e, a)
			}
		})
	}
}
