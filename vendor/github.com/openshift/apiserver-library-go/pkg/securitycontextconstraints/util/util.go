package util

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"

	securityv1 "github.com/openshift/api/security/v1"
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
		string(securityv1.FSTypeHostPath),
		string(securityv1.FSTypeAzureFile),
		string(securityv1.FSTypeFlocker),
		string(securityv1.FSTypeFlexVolume),
		string(securityv1.FSTypeEmptyDir),
		string(securityv1.FSTypeGCEPersistentDisk),
		string(securityv1.FSTypeAWSElasticBlockStore),
		string(securityv1.FSTypeGitRepo),
		string(securityv1.FSTypeSecret),
		string(securityv1.FSTypeNFS),
		string(securityv1.FSTypeISCSI),
		string(securityv1.FSTypeGlusterfs),
		string(securityv1.FSTypePersistentVolumeClaim),
		string(securityv1.FSTypeRBD),
		string(securityv1.FSTypeCinder),
		string(securityv1.FSTypeCephFS),
		string(securityv1.FSTypeDownwardAPI),
		string(securityv1.FSTypeFC),
		string(securityv1.FSTypeConfigMap),
		string(securityv1.FSTypeVsphereVolume),
		string(securityv1.FSTypeQuobyte),
		string(securityv1.FSTypeAzureDisk),
		string(securityv1.FSTypePhotonPersistentDisk),
		string(securityv1.FSProjected),
		string(securityv1.FSPortworxVolume),
		string(securityv1.FSScaleIO),
		string(securityv1.FSStorageOS),
		string(securityv1.FSTypeCSI),
		string(securityv1.FSTypeEphemeral),
	)
	return fstypes
}

// getVolumeFSType gets the FSType for a volume.
func GetVolumeFSType(v api.Volume) (securityv1.FSType, error) {
	switch {
	case v.HostPath != nil:
		return securityv1.FSTypeHostPath, nil
	case v.EmptyDir != nil:
		return securityv1.FSTypeEmptyDir, nil
	case v.GCEPersistentDisk != nil:
		return securityv1.FSTypeGCEPersistentDisk, nil
	case v.AWSElasticBlockStore != nil:
		return securityv1.FSTypeAWSElasticBlockStore, nil
	case v.GitRepo != nil:
		return securityv1.FSTypeGitRepo, nil
	case v.Secret != nil:
		return securityv1.FSTypeSecret, nil
	case v.NFS != nil:
		return securityv1.FSTypeNFS, nil
	case v.ISCSI != nil:
		return securityv1.FSTypeISCSI, nil
	case v.Glusterfs != nil:
		return securityv1.FSTypeGlusterfs, nil
	case v.PersistentVolumeClaim != nil:
		return securityv1.FSTypePersistentVolumeClaim, nil
	case v.RBD != nil:
		return securityv1.FSTypeRBD, nil
	case v.FlexVolume != nil:
		return securityv1.FSTypeFlexVolume, nil
	case v.Cinder != nil:
		return securityv1.FSTypeCinder, nil
	case v.CephFS != nil:
		return securityv1.FSTypeCephFS, nil
	case v.Flocker != nil:
		return securityv1.FSTypeFlocker, nil
	case v.DownwardAPI != nil:
		return securityv1.FSTypeDownwardAPI, nil
	case v.FC != nil:
		return securityv1.FSTypeFC, nil
	case v.AzureFile != nil:
		return securityv1.FSTypeAzureFile, nil
	case v.ConfigMap != nil:
		return securityv1.FSTypeConfigMap, nil
	case v.VsphereVolume != nil:
		return securityv1.FSTypeVsphereVolume, nil
	case v.Quobyte != nil:
		return securityv1.FSTypeQuobyte, nil
	case v.AzureDisk != nil:
		return securityv1.FSTypeAzureDisk, nil
	case v.PhotonPersistentDisk != nil:
		return securityv1.FSTypePhotonPersistentDisk, nil
	case v.Projected != nil:
		return securityv1.FSProjected, nil
	case v.PortworxVolume != nil:
		return securityv1.FSPortworxVolume, nil
	case v.ScaleIO != nil:
		return securityv1.FSScaleIO, nil
	case v.StorageOS != nil:
		return securityv1.FSStorageOS, nil
	case v.CSI != nil:
		return securityv1.FSTypeCSI, nil
	case v.Ephemeral != nil:
		return securityv1.FSTypeEphemeral, nil
	}

	return "", fmt.Errorf("unknown volume type for volume: %#v", v)
}

// fsTypeToStringSet converts an FSType slice to a string set.
func FSTypeToStringSetInternal(fsTypes []securityv1.FSType) sets.String {
	set := sets.NewString()
	for _, v := range fsTypes {
		set.Insert(string(v))
	}
	return set
}

// SCCAllowsAllVolumes checks for FSTypeAll in the scc's allowed volumes.
func SCCAllowsAllVolumes(scc *securityv1.SecurityContextConstraints) bool {
	return SCCAllowsFSTypeInternal(scc, securityv1.FSTypeAll)
}

// SCCAllowsFSTypeInternal is a utility for checking if an SCC allows a particular FSType.
// If all volumes are allowed then this will return true for any FSType passed.
func SCCAllowsFSTypeInternal(scc *securityv1.SecurityContextConstraints, fsType securityv1.FSType) bool {
	if scc == nil {
		return false
	}

	for _, v := range scc.Volumes {
		if v == fsType || v == securityv1.FSTypeAll {
			return true
		}
	}
	return false
}

// SCCAllowsFSType is a utility for checking if an SCC allows a particular FSType.
// If all volumes are allowed then this will return true for any FSType passed.
func SCCAllowsFSType(scc *securityv1.SecurityContextConstraints, fsType securityv1.FSType) bool {
	if scc == nil {
		return false
	}

	for _, v := range scc.Volumes {
		if v == fsType || v == securityv1.FSTypeAll {
			return true
		}
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

// IsOnlyServiceAccountTokenSources returns true if the sources of the projected volume
// source match to what would be injected by the ServiceAccount volume projection controller
//
// This function is derived from pkg/security/podsecuritypolicy/util/util.go with the
// addition of OpenShift-specific "openshift-service-ca.crt" ConfigMap source.
//
// This is what a sample injected volume looks like:
//   - projected:
//     defaultMode: 420
//     sources:
//   - serviceAccountToken:
//     expirationSeconds: 3607
//     path: token
//   - configMap:
//     name: kube-root-ca.crt
//     items:
//   - key: ca.crt
//     path: ca.crt
//   - downwardAPI:
//     items:
//   - path: namespace
//     fieldRef:
//     apiVersion: v1
//     fieldPath: metadata.namespace
//   - configMap:
//     name: openshift-service-ca.crt
//     items:
//   - key: service-ca.crt
//     path: service-ca.crt
func IsOnlyServiceAccountTokenSources(v *api.ProjectedVolumeSource) bool {
	for _, s := range v.Sources {
		// reject any projected source that does not match any of our expected source types
		if s.ServiceAccountToken == nil && s.ConfigMap == nil && s.DownwardAPI == nil {
			return false
		}
		if t := s.ServiceAccountToken; t != nil && (t.Path != "token" || t.Audience != "") {
			return false
		}

		if s.ConfigMap != nil {
			switch cmRef := s.ConfigMap.LocalObjectReference.Name; cmRef {
			case "kube-root-ca.crt":
				if len(s.ConfigMap.Items) != 1 || s.ConfigMap.Items[0].Key != "ca.crt" || s.ConfigMap.Items[0].Path != "ca.crt" {
					return false
				}
			case "openshift-service-ca.crt":
				if len(s.ConfigMap.Items) != 1 || s.ConfigMap.Items[0].Key != "service-ca.crt" || s.ConfigMap.Items[0].Path != "service-ca.crt" {
					return false
				}
			default:
				return false
			}
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
