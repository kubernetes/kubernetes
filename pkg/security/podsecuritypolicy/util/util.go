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

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
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
		string(policy.HostPath),
		string(policy.AzureFile),
		string(policy.Flocker),
		string(policy.FlexVolume),
		string(policy.EmptyDir),
		string(policy.GCEPersistentDisk),
		string(policy.AWSElasticBlockStore),
		string(policy.GitRepo),
		string(policy.Secret),
		string(policy.NFS),
		string(policy.ISCSI),
		string(policy.Glusterfs),
		string(policy.PersistentVolumeClaim),
		string(policy.RBD),
		string(policy.Cinder),
		string(policy.CephFS),
		string(policy.DownwardAPI),
		string(policy.FC),
		string(policy.ConfigMap),
		string(policy.VsphereVolume),
		string(policy.Quobyte),
		string(policy.AzureDisk),
		string(policy.PhotonPersistentDisk),
		string(policy.StorageOS),
		string(policy.Projected),
		string(policy.PortworxVolume),
		string(policy.ScaleIO),
		string(policy.CSI),
	)
	return fstypes
}

// getVolumeFSType gets the FSType for a volume.
func GetVolumeFSType(v api.Volume) (policy.FSType, error) {
	switch {
	case v.HostPath != nil:
		return policy.HostPath, nil
	case v.EmptyDir != nil:
		return policy.EmptyDir, nil
	case v.GCEPersistentDisk != nil:
		return policy.GCEPersistentDisk, nil
	case v.AWSElasticBlockStore != nil:
		return policy.AWSElasticBlockStore, nil
	case v.GitRepo != nil:
		return policy.GitRepo, nil
	case v.Secret != nil:
		return policy.Secret, nil
	case v.NFS != nil:
		return policy.NFS, nil
	case v.ISCSI != nil:
		return policy.ISCSI, nil
	case v.Glusterfs != nil:
		return policy.Glusterfs, nil
	case v.PersistentVolumeClaim != nil:
		return policy.PersistentVolumeClaim, nil
	case v.RBD != nil:
		return policy.RBD, nil
	case v.FlexVolume != nil:
		return policy.FlexVolume, nil
	case v.Cinder != nil:
		return policy.Cinder, nil
	case v.CephFS != nil:
		return policy.CephFS, nil
	case v.Flocker != nil:
		return policy.Flocker, nil
	case v.DownwardAPI != nil:
		return policy.DownwardAPI, nil
	case v.FC != nil:
		return policy.FC, nil
	case v.AzureFile != nil:
		return policy.AzureFile, nil
	case v.ConfigMap != nil:
		return policy.ConfigMap, nil
	case v.VsphereVolume != nil:
		return policy.VsphereVolume, nil
	case v.Quobyte != nil:
		return policy.Quobyte, nil
	case v.AzureDisk != nil:
		return policy.AzureDisk, nil
	case v.PhotonPersistentDisk != nil:
		return policy.PhotonPersistentDisk, nil
	case v.StorageOS != nil:
		return policy.StorageOS, nil
	case v.Projected != nil:
		return policy.Projected, nil
	case v.PortworxVolume != nil:
		return policy.PortworxVolume, nil
	case v.ScaleIO != nil:
		return policy.ScaleIO, nil
	case v.CSI != nil:
		return policy.CSI, nil
	}

	return "", fmt.Errorf("unknown volume type for volume: %#v", v)
}

// FSTypeToStringSet converts an FSType slice to a string set.
func FSTypeToStringSet(fsTypes []policy.FSType) sets.String {
	set := sets.NewString()
	for _, v := range fsTypes {
		set.Insert(string(v))
	}
	return set
}

// PSPAllowsAllVolumes checks for FSTypeAll in the psp's allowed volumes.
func PSPAllowsAllVolumes(psp *policy.PodSecurityPolicy) bool {
	return PSPAllowsFSType(psp, policy.All)
}

// PSPAllowsFSType is a utility for checking if a PSP allows a particular FSType.
// If all volumes are allowed then this will return true for any FSType passed.
func PSPAllowsFSType(psp *policy.PodSecurityPolicy, fsType policy.FSType) bool {
	if psp == nil {
		return false
	}

	for _, v := range psp.Spec.Volumes {
		if v == fsType || v == policy.All {
			return true
		}
	}
	return false
}

// UserFallsInRange is a utility to determine it the id falls in the valid range.
func UserFallsInRange(id int64, rng policy.IDRange) bool {
	return id >= rng.Min && id <= rng.Max
}

// GroupFallsInRange is a utility to determine it the id falls in the valid range.
func GroupFallsInRange(id int64, rng policy.IDRange) bool {
	return id >= rng.Min && id <= rng.Max
}

// AllowsHostVolumePath is a utility for checking if a PSP allows the host volume path.
// This only checks the path. You should still check to make sure the host volume fs type is allowed.
func AllowsHostVolumePath(psp *policy.PodSecurityPolicy, hostPath string) (pathIsAllowed, mustBeReadOnly bool) {
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

func IsOnlyServiceAccountTokenSources(v *api.ProjectedVolumeSource) bool {
	for _, s := range v.Sources {
		// reject any projected source that does not match any of our expected source types
		if s.ServiceAccountToken == nil && s.ConfigMap == nil && s.DownwardAPI == nil {
			return false
		}
		if t := s.ServiceAccountToken; t != nil && (t.Path != "token" || t.Audience != "") {
			return false
		}

		if s.ConfigMap != nil && s.ConfigMap.LocalObjectReference.Name != "kube-root-ca.crt" {
			return false
		}

		if s.DownwardAPI != nil {
			for _, d := range s.DownwardAPI.Items {
				if d.Path != "namespace" || d.FieldRef == nil || d.FieldRef.APIVersion != "v1" || d.FieldRef.FieldPath != "metadata.namespace" {
					return false
				}
			}
		}
	}
	return true
}
