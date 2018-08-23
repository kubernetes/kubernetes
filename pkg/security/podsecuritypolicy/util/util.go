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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	//"k8s.io/kubernetes/pkg/apis/policy"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
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
		string(policyv1beta1.HostPath),
		string(policyv1beta1.AzureFile),
		string(policyv1beta1.Flocker),
		string(policyv1beta1.FlexVolume),
		string(policyv1beta1.EmptyDir),
		string(policyv1beta1.GCEPersistentDisk),
		string(policyv1beta1.AWSElasticBlockStore),
		string(policyv1beta1.GitRepo),
		string(policyv1beta1.Secret),
		string(policyv1beta1.NFS),
		string(policyv1beta1.ISCSI),
		string(policyv1beta1.Glusterfs),
		string(policyv1beta1.PersistentVolumeClaim),
		string(policyv1beta1.RBD),
		string(policyv1beta1.Cinder),
		string(policyv1beta1.CephFS),
		string(policyv1beta1.DownwardAPI),
		string(policyv1beta1.FC),
		string(policyv1beta1.ConfigMap),
		//string(policyv1beta1.VsphereVolume),
		string(policyv1beta1.Quobyte),
		string(policyv1beta1.AzureDisk),
		//string(policyv1beta1.PhotonPersistentDisk),
		//string(policyv1beta1.StorageOS),
		//string(policyv1beta1.Projected),
		//string(policyv1beta1.PortworxVolume),
		//string(policyv1beta1.ScaleIO),
		//string(policyv1beta1.CSI),
	)
	return fstypes
}

// getVolumeFSType gets the FSType for a volume.
func GetVolumeFSType(v api.Volume) (policyv1beta1.FSType, error) {
	switch {
	case v.HostPath != nil:
		return policyv1beta1.HostPath, nil
	case v.EmptyDir != nil:
		return policyv1beta1.EmptyDir, nil
	case v.GCEPersistentDisk != nil:
		return policyv1beta1.GCEPersistentDisk, nil
	case v.AWSElasticBlockStore != nil:
		return policyv1beta1.AWSElasticBlockStore, nil
	case v.GitRepo != nil:
		return policyv1beta1.GitRepo, nil
	case v.Secret != nil:
		return policyv1beta1.Secret, nil
	case v.NFS != nil:
		return policyv1beta1.NFS, nil
	case v.ISCSI != nil:
		return policyv1beta1.ISCSI, nil
	case v.Glusterfs != nil:
		return policyv1beta1.Glusterfs, nil
	case v.PersistentVolumeClaim != nil:
		return policyv1beta1.PersistentVolumeClaim, nil
	case v.RBD != nil:
		return policyv1beta1.RBD, nil
	case v.FlexVolume != nil:
		return policyv1beta1.FlexVolume, nil
	case v.Cinder != nil:
		return policyv1beta1.Cinder, nil
	case v.CephFS != nil:
		return policyv1beta1.CephFS, nil
	case v.Flocker != nil:
		return policyv1beta1.Flocker, nil
	case v.DownwardAPI != nil:
		return policyv1beta1.DownwardAPI, nil
	case v.FC != nil:
		return policyv1beta1.FC, nil
	case v.AzureFile != nil:
		return policyv1beta1.AzureFile, nil
	case v.ConfigMap != nil:
		return policyv1beta1.ConfigMap, nil
	//case v.VsphereVolume != nil:
	//	return policyv1beta1.VsphereVolume, nil
	case v.Quobyte != nil:
		return policyv1beta1.Quobyte, nil
	case v.AzureDisk != nil:
		return policyv1beta1.AzureDisk, nil
		//case v.PhotonPersistentDisk != nil:
		//	return policyv1beta1.PhotonPersistentDisk, nil
		//case v.StorageOS != nil:
		//	return policyv1beta1.StorageOS, nil
		//case v.Projected != nil:
		//	return policyv1beta1.Projected, nil
		//case v.PortworxVolume != nil:
		//	return policyv1beta1.PortworxVolume, nil
		//case v.ScaleIO != nil:
		//	return policyv1beta1.ScaleIO, nil
	}

	return "", fmt.Errorf("unknown volume type for volume: %#v", v)
}

// FSTypeToStringSet converts an FSType slice to a string set.
func FSTypeToStringSet(fsTypes []policyv1beta1.FSType) sets.String {
	set := sets.NewString()
	for _, v := range fsTypes {
		set.Insert(string(v))
	}
	return set
}

// PSPAllowsAllVolumes checks for FSTypeAll in the psp's allowed volumes.
func PSPAllowsAllVolumes(psp *policyv1beta1.PodSecurityPolicy) bool {
	return PSPAllowsFSType(psp, policyv1beta1.All)
}

// PSPAllowsFSType is a utility for checking if a PSP allows a particular FSType.
// If all volumes are allowed then this will return true for any FSType passed.
func PSPAllowsFSType(psp *policyv1beta1.PodSecurityPolicy, fsType policyv1beta1.FSType) bool {
	if psp == nil {
		return false
	}

	for _, v := range psp.Spec.Volumes {
		if v == fsType || v == policyv1beta1.All {
			return true
		}
	}
	return false
}

// UserFallsInRange is a utility to determine it the id falls in the valid range.
func UserFallsInRange(id int64, rng policyv1beta1.IDRange) bool {
	return id >= rng.Min && id <= rng.Max
}

// GroupFallsInRange is a utility to determine it the id falls in the valid range.
func GroupFallsInRange(id int64, rng policyv1beta1.IDRange) bool {
	return id >= rng.Min && id <= rng.Max
}

// AllowsHostVolumePath is a utility for checking if a PSP allows the host volume path.
// This only checks the path. You should still check to make sure the host volume fs type is allowed.
func AllowsHostVolumePath(psp *policyv1beta1.PodSecurityPolicy, hostPath string) (pathIsAllowed, mustBeReadOnly bool) {
	if psp == nil {
		return false, false
	}

	// If no allowed paths are specified then allow any path
	if len(psp.Spec.AllowedHostPaths) == 0 {
		return true, false
	}

	for _, allowedPath := range psp.Spec.AllowedHostPaths {
		if hasPathPrefix(hostPath, allowedPath.PathPrefix) {
			if !allowedPath.ReadOnly {
				return true, allowedPath.ReadOnly
			}
			pathIsAllowed = true
			mustBeReadOnly = true
		}
	}

	return pathIsAllowed, mustBeReadOnly
}

// hasPathPrefix returns true if the string matches pathPrefix exactly, or if is prefixed with pathPrefix at a path segment boundary
// the string and pathPrefix are both normalized to remove trailing slashes prior to checking.
func hasPathPrefix(s, pathPrefix string) bool {

	s = strings.TrimSuffix(s, "/")
	pathPrefix = strings.TrimSuffix(pathPrefix, "/")

	// Short circuit if s doesn't contain the prefix at all
	if !strings.HasPrefix(s, pathPrefix) {
		return false
	}

	pathPrefixLength := len(pathPrefix)

	if len(s) == pathPrefixLength {
		// Exact match
		return true
	}

	if s[pathPrefixLength:pathPrefixLength+1] == "/" {
		// The next character in s is a path segment boundary
		// Check this instead of normalizing pathPrefix to avoid allocating on every call
		// Example where this check applies: s=/foo/bar and pathPrefix=/foo
		return true
	}

	return false
}

// EqualStringSlices compares string slices for equality. Slices are equal when
// their sizes and elements on similar positions are equal.
func EqualStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
