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
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/pod-security-admission/api"
)

/*
In addition to restricting HostPath volumes, the restricted profile
limits usage of inline pod volume sources to:
* configMap
* downwardAPI
* emptyDir
* projected
* secret
* csi
* persistentVolumeClaim
* ephemeral

**Restricted Fields:**

spec.volumes[*].hostPath
spec.volumes[*].gcePersistentDisk
spec.volumes[*].awsElasticBlockStore
spec.volumes[*].gitRepo
spec.volumes[*].nfs
spec.volumes[*].iscsi
spec.volumes[*].glusterfs
spec.volumes[*].rbd
spec.volumes[*].flexVolume
spec.volumes[*].cinder
spec.volumes[*].cephfs
spec.volumes[*].flocker
spec.volumes[*].fc
spec.volumes[*].azureFile
spec.volumes[*].vsphereVolume
spec.volumes[*].quobyte
spec.volumes[*].azureDisk
spec.volumes[*].portworxVolume
spec.volumes[*].photonPersistentDisk
spec.volumes[*].scaleIO
spec.volumes[*].storageos

**Allowed Values:** undefined/null
*/

func init() {
	addCheck(CheckRestrictedVolumes)
}

// CheckRestrictedVolumes returns a restricted level check
// that limits usage of specific volume types in 1.0+
func CheckRestrictedVolumes() Check {
	return Check{
		ID:    "restrictedVolumes",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion:   api.MajorMinorVersion(1, 0),
				CheckPod:         restrictedVolumes_1_0,
				OverrideCheckIDs: []CheckID{checkHostPathVolumesID, checkBaselineVolumesID},
			},
		},
	}
}

func restrictedVolumes_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badVolumes []string
	badVolumeTypes := sets.NewString()

	for _, volume := range podSpec.Volumes {
		switch {
		case volume.ConfigMap != nil,
			volume.CSI != nil,
			volume.DownwardAPI != nil,
			volume.EmptyDir != nil,
			volume.Ephemeral != nil,
			volume.PersistentVolumeClaim != nil,
			volume.Projected != nil,
			volume.Secret != nil:
			continue

		default:
			badVolumes = append(badVolumes, volume.Name)

			switch {
			case volume.HostPath != nil:
				badVolumeTypes.Insert("hostPath")
			case volume.GCEPersistentDisk != nil:
				badVolumeTypes.Insert("gcePersistentDisk")
			case volume.AWSElasticBlockStore != nil:
				badVolumeTypes.Insert("awsElasticBlockStore")
			case volume.GitRepo != nil:
				badVolumeTypes.Insert("gitRepo")
			case volume.NFS != nil:
				badVolumeTypes.Insert("nfs")
			case volume.ISCSI != nil:
				badVolumeTypes.Insert("iscsi")
			case volume.Glusterfs != nil:
				badVolumeTypes.Insert("glusterfs")
			case volume.RBD != nil:
				badVolumeTypes.Insert("rbd")
			case volume.FlexVolume != nil:
				badVolumeTypes.Insert("flexVolume")
			case volume.Cinder != nil:
				badVolumeTypes.Insert("cinder")
			case volume.CephFS != nil:
				badVolumeTypes.Insert("cephfs")
			case volume.Flocker != nil:
				badVolumeTypes.Insert("flocker")
			case volume.FC != nil:
				badVolumeTypes.Insert("fc")
			case volume.AzureFile != nil:
				badVolumeTypes.Insert("azureFile")
			case volume.VsphereVolume != nil:
				badVolumeTypes.Insert("vsphereVolume")
			case volume.Quobyte != nil:
				badVolumeTypes.Insert("quobyte")
			case volume.AzureDisk != nil:
				badVolumeTypes.Insert("azureDisk")
			case volume.PhotonPersistentDisk != nil:
				badVolumeTypes.Insert("photonPersistentDisk")
			case volume.PortworxVolume != nil:
				badVolumeTypes.Insert("portworxVolume")
			case volume.ScaleIO != nil:
				badVolumeTypes.Insert("scaleIO")
			case volume.StorageOS != nil:
				badVolumeTypes.Insert("storageos")
			default:
				badVolumeTypes.Insert("unknown")
			}
		}
	}

	if len(badVolumes) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "restricted volume types",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s %s %s %s",
				pluralize("volume", "volumes", len(badVolumes)),
				joinQuote(badVolumes),
				pluralize("uses", "use", len(badVolumes)),
				pluralize("restricted volume type", "restricted volume types", len(badVolumeTypes)),
				joinQuote(badVolumeTypes.List()),
			),
		}
	}

	return CheckResult{Allowed: true}
}
