/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

const (
	ValidatedPSPAnnotation = "kubernetes.io/psp"
)

func GetAllFSTypesExcept(exceptions ...string) sets.String {
	fstypes := GetAllFSTypesAsSet()
	for _, e := range exceptions {
		fstypes.Delete(e)
	}
	return fstypes
}

func GetAllFSTypesAsSet() sets.String {
	fstypes := sets.NewString()
	fstypes.Insert(
		string(extensions.HostPath),
		string(extensions.AzureFile),
		string(extensions.Flocker),
		string(extensions.FlexVolume),
		string(extensions.EmptyDir),
		string(extensions.GCEPersistentDisk),
		string(extensions.AWSElasticBlockStore),
		string(extensions.GitRepo),
		string(extensions.Secret),
		string(extensions.NFS),
		string(extensions.ISCSI),
		string(extensions.Glusterfs),
		string(extensions.PersistentVolumeClaim),
		string(extensions.RBD),
		string(extensions.Cinder),
		string(extensions.CephFS),
		string(extensions.DownwardAPI),
		string(extensions.FC),
		string(extensions.ConfigMap),
		string(extensions.VsphereVolume),
		string(extensions.Quobyte),
		string(extensions.AzureDisk),
		string(extensions.PhotonPersistentDisk),
		string(extensions.StorageOS),
		string(extensions.Projected),
		string(extensions.PortworxVolume),
		string(extensions.ScaleIO),
	)
	return fstypes
}

// getVolumeFSType gets the FSType for a volume.
func GetVolumeFSType(v api.Volume) (extensions.FSType, error) {
	switch {
	case v.HostPath != nil:
		return extensions.HostPath, nil
	case v.EmptyDir != nil:
		return extensions.EmptyDir, nil
	case v.GCEPersistentDisk != nil:
		return extensions.GCEPersistentDisk, nil
	case v.AWSElasticBlockStore != nil:
		return extensions.AWSElasticBlockStore, nil
	case v.GitRepo != nil:
		return extensions.GitRepo, nil
	case v.Secret != nil:
		return extensions.Secret, nil
	case v.NFS != nil:
		return extensions.NFS, nil
	case v.ISCSI != nil:
		return extensions.ISCSI, nil
	case v.Glusterfs != nil:
		return extensions.Glusterfs, nil
	case v.PersistentVolumeClaim != nil:
		return extensions.PersistentVolumeClaim, nil
	case v.RBD != nil:
		return extensions.RBD, nil
	case v.FlexVolume != nil:
		return extensions.FlexVolume, nil
	case v.Cinder != nil:
		return extensions.Cinder, nil
	case v.CephFS != nil:
		return extensions.CephFS, nil
	case v.Flocker != nil:
		return extensions.Flocker, nil
	case v.DownwardAPI != nil:
		return extensions.DownwardAPI, nil
	case v.FC != nil:
		return extensions.FC, nil
	case v.AzureFile != nil:
		return extensions.AzureFile, nil
	case v.ConfigMap != nil:
		return extensions.ConfigMap, nil
	case v.VsphereVolume != nil:
		return extensions.VsphereVolume, nil
	case v.Quobyte != nil:
		return extensions.Quobyte, nil
	case v.AzureDisk != nil:
		return extensions.AzureDisk, nil
	case v.PhotonPersistentDisk != nil:
		return extensions.PhotonPersistentDisk, nil
	case v.StorageOS != nil:
		return extensions.StorageOS, nil
	case v.Projected != nil:
		return extensions.Projected, nil
	case v.PortworxVolume != nil:
		return extensions.PortworxVolume, nil
	case v.ScaleIO != nil:
		return extensions.ScaleIO, nil
	}

	return "", fmt.Errorf("unknown volume type for volume: %#v", v)
}

// FSTypeToStringSet converts an FSType slice to a string set.
func FSTypeToStringSet(fsTypes []extensions.FSType) sets.String {
	set := sets.NewString()
	for _, v := range fsTypes {
		set.Insert(string(v))
	}
	return set
}

// PSPAllowsAllVolumes checks for FSTypeAll in the psp's allowed volumes.
func PSPAllowsAllVolumes(psp *extensions.PodSecurityPolicy) bool {
	return PSPAllowsFSType(psp, extensions.All)
}

// PSPAllowsFSType is a utility for checking if a PSP allows a particular FSType.
// If all volumes are allowed then this will return true for any FSType passed.
func PSPAllowsFSType(psp *extensions.PodSecurityPolicy, fsType extensions.FSType) bool {
	if psp == nil {
		return false
	}

	for _, v := range psp.Spec.Volumes {
		if v == fsType || v == extensions.All {
			return true
		}
	}
	return false
}

// UserFallsInRange is a utility to determine it the id falls in the valid range.
func UserFallsInRange(id int64, rng extensions.UserIDRange) bool {
	return id >= rng.Min && id <= rng.Max
}

// GroupFallsInRange is a utility to determine it the id falls in the valid range.
func GroupFallsInRange(id int64, rng extensions.GroupIDRange) bool {
	return id >= rng.Min && id <= rng.Max
}
