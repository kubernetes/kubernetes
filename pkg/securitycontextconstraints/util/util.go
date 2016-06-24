/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/sets"
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
		string(api.FSTypeHostPath),
		string(api.FSTypeAzureFile),
		string(api.FSTypeFlocker),
		string(api.FSTypeFlexVolume),
		string(api.FSTypeEmptyDir),
		string(api.FSTypeGCEPersistentDisk),
		string(api.FSTypeAWSElasticBlockStore),
		string(api.FSTypeGitRepo),
		string(api.FSTypeSecret),
		string(api.FSTypeNFS),
		string(api.FSTypeISCSI),
		string(api.FSTypeGlusterfs),
		string(api.FSTypePersistentVolumeClaim),
		string(api.FSTypeRBD),
		string(api.FSTypeCinder),
		string(api.FSTypeCephFS),
		string(api.FSTypeDownwardAPI),
		string(api.FSTypeFC),
		string(api.FSTypeConfigMap))
	return fstypes
}

// getVolumeFSType gets the FSType for a volume.
func GetVolumeFSType(v api.Volume) (api.FSType, error) {
	switch {
	case v.HostPath != nil:
		return api.FSTypeHostPath, nil
	case v.EmptyDir != nil:
		return api.FSTypeEmptyDir, nil
	case v.GCEPersistentDisk != nil:
		return api.FSTypeGCEPersistentDisk, nil
	case v.AWSElasticBlockStore != nil:
		return api.FSTypeAWSElasticBlockStore, nil
	case v.GitRepo != nil:
		return api.FSTypeGitRepo, nil
	case v.Secret != nil:
		return api.FSTypeSecret, nil
	case v.NFS != nil:
		return api.FSTypeNFS, nil
	case v.ISCSI != nil:
		return api.FSTypeISCSI, nil
	case v.Glusterfs != nil:
		return api.FSTypeGlusterfs, nil
	case v.PersistentVolumeClaim != nil:
		return api.FSTypePersistentVolumeClaim, nil
	case v.RBD != nil:
		return api.FSTypeRBD, nil
	case v.FlexVolume != nil:
		return api.FSTypeFlexVolume, nil
	case v.Cinder != nil:
		return api.FSTypeCinder, nil
	case v.CephFS != nil:
		return api.FSTypeCephFS, nil
	case v.Flocker != nil:
		return api.FSTypeFlocker, nil
	case v.DownwardAPI != nil:
		return api.FSTypeDownwardAPI, nil
	case v.FC != nil:
		return api.FSTypeFC, nil
	case v.AzureFile != nil:
		return api.FSTypeAzureFile, nil
	case v.ConfigMap != nil:
		return api.FSTypeConfigMap, nil
	}

	return "", fmt.Errorf("unknown volume type for volume: %#v", v)
}

// fsTypeToStringSet converts an FSType slice to a string set.
func FSTypeToStringSet(fsTypes []api.FSType) sets.String {
	set := sets.NewString()
	for _, v := range fsTypes {
		set.Insert(string(v))
	}
	return set
}

// SCCAllowsAllVolumes checks for FSTypeAll in the scc's allowed volumes.
func SCCAllowsAllVolumes(scc *api.SecurityContextConstraints) bool {
	return SCCAllowsFSType(scc, api.FSTypeAll)
}

// SCCAllowsFSType is a utility for checking if an SCC allows a particular FSType.
// If all volumes are allowed then this will return true for any FSType passed.
func SCCAllowsFSType(scc *api.SecurityContextConstraints, fsType api.FSType) bool {
	if scc == nil {
		return false
	}

	for _, v := range scc.Volumes {
		if v == fsType || v == api.FSTypeAll {
			return true
		}
	}
	return false
}
