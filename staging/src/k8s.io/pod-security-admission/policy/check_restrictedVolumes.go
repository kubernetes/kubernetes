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
	"k8s.io/apimachinery/pkg/util/validation/field"
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
				CheckPod:         withOptions(restrictedVolumesV1Dot0),
				OverrideCheckIDs: []CheckID{checkHostPathVolumesID},
			},
		},
	}
}

func restrictedVolumesV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	badVolumes := NewViolations(opts.withFieldErrors)
	badVolumeTypes := sets.NewString()

	for i, volume := range podSpec.Volumes {
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
			var volumesIndexPath *field.Path
			if opts.withFieldErrors {
				volumesIndexPath = volumesPath.Index(i)
			} else {
				badVolumes.Add(volume.Name)
			}

			switch {
			case volume.HostPath != nil:
				badVolumeTypes.Insert("hostPath")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("hostPath")))
				}
			case volume.GCEPersistentDisk != nil:
				badVolumeTypes.Insert("gcePersistentDisk")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("gcePersistentDisk")))
				}
			case volume.AWSElasticBlockStore != nil:
				badVolumeTypes.Insert("awsElasticBlockStore")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("awsElasticBlockStore")))
				}
			case volume.GitRepo != nil:
				badVolumeTypes.Insert("gitRepo")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("gitRepo")))
				}
			case volume.NFS != nil:
				badVolumeTypes.Insert("nfs")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("nfs")))
				}
			case volume.ISCSI != nil:
				badVolumeTypes.Insert("iscsi")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("iscsi")))
				}
			case volume.Glusterfs != nil:
				badVolumeTypes.Insert("glusterfs")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("glusterfs")))
				}
			case volume.RBD != nil:
				badVolumeTypes.Insert("rbd")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("rbd")))
				}
			case volume.FlexVolume != nil:
				badVolumeTypes.Insert("flexVolume")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("flexVolume")))
				}
			case volume.Cinder != nil:
				badVolumeTypes.Insert("cinder")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("cinder")))
				}
			case volume.CephFS != nil:
				badVolumeTypes.Insert("cephfs")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("cephfs")))
				}
			case volume.Flocker != nil:
				badVolumeTypes.Insert("flocker")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("flocker")))
				}
			case volume.FC != nil:
				badVolumeTypes.Insert("fc")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("fc")))
				}
			case volume.AzureFile != nil:
				badVolumeTypes.Insert("azureFile")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("azureFile")))
				}
			case volume.VsphereVolume != nil:
				badVolumeTypes.Insert("vsphereVolume")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("vsphereVolume")))
				}
			case volume.Quobyte != nil:
				badVolumeTypes.Insert("quobyte")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("quobyte")))
				}
			case volume.AzureDisk != nil:
				badVolumeTypes.Insert("azureDisk")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("azureDisk")))
				}
			case volume.PhotonPersistentDisk != nil:
				badVolumeTypes.Insert("photonPersistentDisk")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("photonPersistentDisk")))
				}
			case volume.PortworxVolume != nil:
				badVolumeTypes.Insert("portworxVolume")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("portworxVolume")))
				}
			case volume.ScaleIO != nil:
				badVolumeTypes.Insert("scaleIO")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("scaleIO")))
				}
			case volume.StorageOS != nil:
				badVolumeTypes.Insert("storageos")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("storageos")))
				}
			default:
				badVolumeTypes.Insert("unknown")
				if opts.withFieldErrors {
					badVolumes.Add(volume.Name, forbidden(volumesIndexPath.Child("unknown")))
				}
			}
		}
	}

	if !badVolumes.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "restricted volume types",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s %s %s %s",
				pluralize("volume", "volumes", badVolumes.Len()),
				joinQuote(badVolumes.Data()),
				pluralize("uses", "use", badVolumes.Len()),
				pluralize("restricted volume type", "restricted volume types", len(badVolumeTypes)),
				joinQuote(badVolumeTypes.List()),
			),
			ErrList: badVolumes.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}
