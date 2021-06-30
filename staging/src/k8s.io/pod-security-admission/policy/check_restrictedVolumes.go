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
limits usage of non-core volume types to those defined through PersistentVolumes.

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
spec.volumes[*].cephFS
spec.volumes[*].flocker
spec.volumes[*].fc
spec.volumes[*].azureFile
spec.volumes[*].vsphereVolume
spec.volumes[*].quobyte
spec.volumes[*].azureDisk
spec.volumes[*].portworxVolume
spec.volumes[*].scaleIO
spec.volumes[*].storageos
spec.volumes[*].csi

**Allowed Values:** undefined/nil
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
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       restrictedVolumes_1_0,
			},
		},
	}
}

func restrictedVolumes_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	restrictedVolumeNames := sets.NewString()

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
			restrictedVolumeNames.Insert(volume.Name)
		}
	}

	if len(restrictedVolumeNames) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "restricted volume types",
			ForbiddenDetail: fmt.Sprintf("volumes %q have restricted types", restrictedVolumeNames.List()),
		}
	}

	return CheckResult{Allowed: true}
}
