/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"net"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

const csiDriverNameRexpErrMsg string = "must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character"
const csiDriverNameRexpFmt string = `^[a-zA-Z0-9][-a-zA-Z0-9_.]{0,61}[a-zA-Z-0-9]$`

var pdPartitionErrorMsg string = validation.InclusiveRangeError(1, 255)
var fileModeErrorMsg string = "must be a number between 0 and 0777 (octal), both inclusive"

var iscsiInitiatorIqnRegex = regexp.MustCompile(`iqn\.\d{4}-\d{2}\.([[:alnum:]-.]+)(:[^,;*&$|\s]+)$`)
var iscsiInitiatorEuiRegex = regexp.MustCompile(`^eui.[[:alnum:]]{16}$`)
var iscsiInitiatorNaaRegex = regexp.MustCompile(`^naa.[[:alnum:]]{32}$`)

var csiDriverNameRexp = regexp.MustCompile(csiDriverNameRexpFmt)

func ValidateVolumes(volumes []core.Volume, fldPath *field.Path) (map[string]core.VolumeSource, field.ErrorList) {
	allErrs := field.ErrorList{}

	allNames := sets.String{}
	vols := make(map[string]core.VolumeSource)
	for i, vol := range volumes {
		idxPath := fldPath.Index(i)
		namePath := idxPath.Child("name")
		el := validateVolumeSource(&vol.VolumeSource, idxPath, vol.Name)
		if len(vol.Name) == 0 {
			el = append(el, field.Required(namePath, ""))
		} else {
			el = append(el, ValidateDNS1123Label(vol.Name, namePath)...)
		}
		if allNames.Has(vol.Name) {
			el = append(el, field.Duplicate(namePath, vol.Name))
		}
		if len(el) == 0 {
			allNames.Insert(vol.Name)
			vols[vol.Name] = vol.VolumeSource
		} else {
			allErrs = append(allErrs, el...)
		}

	}
	return vols, allErrs
}

func IsMatchedVolume(name string, volumes map[string]core.VolumeSource) bool {
	if _, ok := volumes[name]; ok {
		return true
	} else {
		return false
	}
}

// ValidatePersistentVolumeName checks that a name is appropriate for a
// PersistentVolumeName object.
var ValidatePersistentVolumeName = NameIsDNSSubdomain

var supportedAccessModes = sets.NewString(string(core.ReadWriteOnce), string(core.ReadOnlyMany), string(core.ReadWriteMany))

var supportedReclaimPolicy = sets.NewString(string(core.PersistentVolumeReclaimDelete), string(core.PersistentVolumeReclaimRecycle), string(core.PersistentVolumeReclaimRetain))

var supportedVolumeModes = sets.NewString(string(core.PersistentVolumeBlock), string(core.PersistentVolumeFilesystem))

func ValidatePersistentVolume(pv *core.PersistentVolume) field.ErrorList {
	metaPath := field.NewPath("metadata")
	allErrs := ValidateObjectMeta(&pv.ObjectMeta, false, ValidatePersistentVolumeName, metaPath)

	specPath := field.NewPath("spec")
	if len(pv.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("accessModes"), ""))
	}
	for _, mode := range pv.Spec.AccessModes {
		if !supportedAccessModes.Has(string(mode)) {
			allErrs = append(allErrs, field.NotSupported(specPath.Child("accessModes"), mode, supportedAccessModes.List()))
		}
	}

	if len(pv.Spec.Capacity) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("capacity"), ""))
	}

	if _, ok := pv.Spec.Capacity[core.ResourceStorage]; !ok || len(pv.Spec.Capacity) > 1 {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("capacity"), pv.Spec.Capacity, []string{string(core.ResourceStorage)}))
	}
	capPath := specPath.Child("capacity")
	for r, qty := range pv.Spec.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty, capPath.Key(string(r)))...)
		allErrs = append(allErrs, ValidatePositiveQuantityValue(qty, capPath.Key(string(r)))...)
	}
	if len(string(pv.Spec.PersistentVolumeReclaimPolicy)) > 0 {
		if !supportedReclaimPolicy.Has(string(pv.Spec.PersistentVolumeReclaimPolicy)) {
			allErrs = append(allErrs, field.NotSupported(specPath.Child("persistentVolumeReclaimPolicy"), pv.Spec.PersistentVolumeReclaimPolicy, supportedReclaimPolicy.List()))
		}
	}

	nodeAffinitySpecified, errs := validateVolumeNodeAffinity(pv.Spec.NodeAffinity, specPath.Child("nodeAffinity"))
	allErrs = append(allErrs, errs...)

	numVolumes := 0
	if pv.Spec.HostPath != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("hostPath"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateHostPathVolumeSource(pv.Spec.HostPath, specPath.Child("hostPath"))...)
		}
	}
	if pv.Spec.GCEPersistentDisk != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("gcePersistentDisk"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateGCEPersistentDiskVolumeSource(pv.Spec.GCEPersistentDisk, specPath.Child("persistentDisk"))...)
		}
	}
	if pv.Spec.AWSElasticBlockStore != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("awsElasticBlockStore"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateAWSElasticBlockStoreVolumeSource(pv.Spec.AWSElasticBlockStore, specPath.Child("awsElasticBlockStore"))...)
		}
	}
	if pv.Spec.Glusterfs != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("glusterfs"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateGlusterfsVolumeSource(pv.Spec.Glusterfs, specPath.Child("glusterfs"))...)
		}
	}
	if pv.Spec.Flocker != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("flocker"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateFlockerVolumeSource(pv.Spec.Flocker, specPath.Child("flocker"))...)
		}
	}
	if pv.Spec.NFS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("nfs"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateNFSVolumeSource(pv.Spec.NFS, specPath.Child("nfs"))...)
		}
	}
	if pv.Spec.RBD != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("rbd"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateRBDPersistentVolumeSource(pv.Spec.RBD, specPath.Child("rbd"))...)
		}
	}
	if pv.Spec.Quobyte != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("quobyte"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateQuobyteVolumeSource(pv.Spec.Quobyte, specPath.Child("quobyte"))...)
		}
	}
	if pv.Spec.CephFS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("cephFS"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateCephFSPersistentVolumeSource(pv.Spec.CephFS, specPath.Child("cephfs"))...)
		}
	}
	if pv.Spec.ISCSI != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("iscsi"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateISCSIPersistentVolumeSource(pv.Spec.ISCSI, specPath.Child("iscsi"))...)
		}
		if pv.Spec.ISCSI.InitiatorName != nil && len(pv.ObjectMeta.Name+":"+pv.Spec.ISCSI.TargetPortal) > 64 {
			tooLongErr := "Total length of <volume name>:<iscsi.targetPortal> must be under 64 characters if iscsi.initiatorName is specified."
			allErrs = append(allErrs, field.Invalid(metaPath.Child("name"), pv.ObjectMeta.Name, tooLongErr))
		}
	}
	if pv.Spec.Cinder != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("cinder"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateCinderVolumeSource(pv.Spec.Cinder, specPath.Child("cinder"))...)
		}
	}
	if pv.Spec.FC != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("fc"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateFCVolumeSource(pv.Spec.FC, specPath.Child("fc"))...)
		}
	}
	if pv.Spec.FlexVolume != nil {
		numVolumes++
		allErrs = append(allErrs, validateFlexPersistentVolumeSource(pv.Spec.FlexVolume, specPath.Child("flexVolume"))...)
	}
	if pv.Spec.AzureFile != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("azureFile"), "may not specify more than 1 volume type"))

		} else {
			numVolumes++
			allErrs = append(allErrs, validateAzureFilePV(pv.Spec.AzureFile, specPath.Child("azureFile"))...)
		}
	}

	if pv.Spec.VsphereVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("vsphereVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateVsphereVolumeSource(pv.Spec.VsphereVolume, specPath.Child("vsphereVolume"))...)
		}
	}
	if pv.Spec.PhotonPersistentDisk != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("photonPersistentDisk"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validatePhotonPersistentDiskVolumeSource(pv.Spec.PhotonPersistentDisk, specPath.Child("photonPersistentDisk"))...)
		}
	}
	if pv.Spec.PortworxVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("portworxVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validatePortworxVolumeSource(pv.Spec.PortworxVolume, specPath.Child("portworxVolume"))...)
		}
	}
	if pv.Spec.AzureDisk != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("azureDisk"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateAzureDisk(pv.Spec.AzureDisk, specPath.Child("azureDisk"))...)
		}
	}
	if pv.Spec.ScaleIO != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("scaleIO"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateScaleIOPersistentVolumeSource(pv.Spec.ScaleIO, specPath.Child("scaleIO"))...)
		}
	}
	if pv.Spec.Local != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("local"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			if !utilfeature.DefaultFeatureGate.Enabled(features.PersistentLocalVolumes) {
				allErrs = append(allErrs, field.Forbidden(specPath.Child("local"), "Local volumes are disabled by feature-gate"))
			}
			allErrs = append(allErrs, validateLocalVolumeSource(pv.Spec.Local, specPath.Child("local"))...)

			// NodeAffinity is required
			if !nodeAffinitySpecified {
				allErrs = append(allErrs, field.Required(metaPath.Child("annotations"), "Local volume requires node affinity"))
			}
		}
	}
	if pv.Spec.StorageOS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("storageos"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateStorageOSPersistentVolumeSource(pv.Spec.StorageOS, specPath.Child("storageos"))...)
		}
	}

	if pv.Spec.CSI != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("csi"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateCSIPersistentVolumeSource(pv.Spec.CSI, specPath.Child("csi"))...)
		}
	}

	if numVolumes == 0 {
		allErrs = append(allErrs, field.Required(specPath, "must specify a volume type"))
	}

	// do not allow hostPath mounts of '/' to have a 'recycle' reclaim policy
	if pv.Spec.HostPath != nil && path.Clean(pv.Spec.HostPath.Path) == "/" && pv.Spec.PersistentVolumeReclaimPolicy == core.PersistentVolumeReclaimRecycle {
		allErrs = append(allErrs, field.Forbidden(specPath.Child("persistentVolumeReclaimPolicy"), "may not be 'recycle' for a hostPath mount of '/'"))
	}

	if len(pv.Spec.StorageClassName) > 0 {
		for _, msg := range ValidateClassName(pv.Spec.StorageClassName, false) {
			allErrs = append(allErrs, field.Invalid(specPath.Child("storageClassName"), pv.Spec.StorageClassName, msg))
		}
	}
	if pv.Spec.VolumeMode != nil && !utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		allErrs = append(allErrs, field.Forbidden(specPath.Child("volumeMode"), "PersistentVolume volumeMode is disabled by feature-gate"))
	} else if pv.Spec.VolumeMode != nil && !supportedVolumeModes.Has(string(*pv.Spec.VolumeMode)) {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("volumeMode"), *pv.Spec.VolumeMode, supportedVolumeModes.List()))
	}
	return allErrs
}

// ValidatePersistentVolumeUpdate tests to see if the update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeUpdate(newPv, oldPv *core.PersistentVolume) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = ValidatePersistentVolume(newPv)

	// PersistentVolumeSource should be immutable after creation.
	if !apiequality.Semantic.DeepEqual(newPv.Spec.PersistentVolumeSource, oldPv.Spec.PersistentVolumeSource) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "persistentvolumesource"), "is immutable after creation"))
	}

	newPv.Status = oldPv.Status

	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		allErrs = append(allErrs, ValidateImmutableField(newPv.Spec.VolumeMode, oldPv.Spec.VolumeMode, field.NewPath("volumeMode"))...)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		// Allow setting NodeAffinity if oldPv NodeAffinity was not set
		if oldPv.Spec.NodeAffinity != nil {
			allErrs = append(allErrs, ValidateImmutableField(newPv.Spec.NodeAffinity, oldPv.Spec.NodeAffinity, field.NewPath("nodeAffinity"))...)
		}
	}

	return allErrs
}

// ValidatePersistentVolumeStatusUpdate tests to see if the status update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeStatusUpdate(newPv, oldPv *core.PersistentVolume) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPv.ObjectMeta, &oldPv.ObjectMeta, field.NewPath("metadata"))
	if len(newPv.ResourceVersion) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("resourceVersion"), ""))
	}
	newPv.Spec = oldPv.Spec
	return allErrs
}

// ValidatePersistentVolumeClaim validates a PersistentVolumeClaim
func ValidatePersistentVolumeClaim(pvc *core.PersistentVolumeClaim) field.ErrorList {
	allErrs := ValidateObjectMeta(&pvc.ObjectMeta, true, ValidatePersistentVolumeName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePersistentVolumeClaimSpec(&pvc.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidatePersistentVolumeClaimSpec validates a PersistentVolumeClaimSpec
func ValidatePersistentVolumeClaimSpec(spec *core.PersistentVolumeClaimSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(spec.AccessModes) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("accessModes"), "at least 1 access mode is required"))
	}
	if spec.Selector != nil {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
	}
	for _, mode := range spec.AccessModes {
		if mode != core.ReadWriteOnce && mode != core.ReadOnlyMany && mode != core.ReadWriteMany {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("accessModes"), mode, supportedAccessModes.List()))
		}
	}
	storageValue, ok := spec.Resources.Requests[core.ResourceStorage]
	if !ok {
		allErrs = append(allErrs, field.Required(fldPath.Child("resources").Key(string(core.ResourceStorage)), ""))
	} else {
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(core.ResourceStorage), storageValue, fldPath.Child("resources").Key(string(core.ResourceStorage)))...)
		allErrs = append(allErrs, ValidatePositiveQuantityValue(storageValue, fldPath.Child("resources").Key(string(core.ResourceStorage)))...)
	}

	if spec.StorageClassName != nil && len(*spec.StorageClassName) > 0 {
		for _, msg := range ValidateClassName(*spec.StorageClassName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("storageClassName"), *spec.StorageClassName, msg))
		}
	}
	if spec.VolumeMode != nil && !utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("volumeMode"), "PersistentVolumeClaim volumeMode is disabled by feature-gate"))
	} else if spec.VolumeMode != nil && !supportedVolumeModes.Has(string(*spec.VolumeMode)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("volumeMode"), *spec.VolumeMode, supportedVolumeModes.List()))
	}
	return allErrs
}

// ValidatePersistentVolumeClaimStatusUpdate validates an update to status of a PersistentVolumeClaim
func ValidatePersistentVolumeClaimStatusUpdate(newPvc, oldPvc *core.PersistentVolumeClaim) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPvc.ObjectMeta, &oldPvc.ObjectMeta, field.NewPath("metadata"))
	if len(newPvc.ResourceVersion) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("resourceVersion"), ""))
	}
	if len(newPvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("Spec", "accessModes"), ""))
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) && len(newPvc.Status.Conditions) > 0 {
		conditionPath := field.NewPath("status", "conditions")
		allErrs = append(allErrs, field.Forbidden(conditionPath, "invalid field"))
	}
	capPath := field.NewPath("status", "capacity")
	for r, qty := range newPvc.Status.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty, capPath.Key(string(r)))...)
	}
	newPvc.Spec = oldPvc.Spec
	return allErrs
}

func GetVolumeMountMap(mounts []core.VolumeMount) map[string]string {
	volmounts := make(map[string]string)

	for _, mnt := range mounts {
		volmounts[mnt.Name] = mnt.MountPath
	}

	return volmounts
}

func GetVolumeDeviceMap(devices []core.VolumeDevice) map[string]string {
	voldevices := make(map[string]string)

	for _, dev := range devices {
		voldevices[dev.Name] = dev.DevicePath
	}

	return voldevices
}

func ValidateVolumeMounts(mounts []core.VolumeMount, voldevices map[string]string, volumes map[string]core.VolumeSource, container *core.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	mountpoints := sets.NewString()

	for i, mnt := range mounts {
		idxPath := fldPath.Index(i)
		if len(mnt.Name) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("name"), ""))
		}
		if !IsMatchedVolume(mnt.Name, volumes) {
			allErrs = append(allErrs, field.NotFound(idxPath.Child("name"), mnt.Name))
		}
		if len(mnt.MountPath) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("mountPath"), ""))
		}
		if mountpoints.Has(mnt.MountPath) {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("mountPath"), mnt.MountPath, "must be unique"))
		}
		mountpoints.Insert(mnt.MountPath)

		// check for overlap with VolumeDevice
		if mountNameAlreadyExists(mnt.Name, voldevices) {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), mnt.Name, "must not already exist in volumeDevices"))
		}
		if mountPathAlreadyExists(mnt.MountPath, voldevices) {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("mountPath"), mnt.MountPath, "must not already exist as a path in volumeDevices"))
		}

		if len(mnt.SubPath) > 0 {
			if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeSubpath) {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("subPath"), "subPath is disabled by feature-gate"))
			} else {
				allErrs = append(allErrs, validateLocalDescendingPath(mnt.SubPath, fldPath.Child("subPath"))...)
			}
		}

		if mnt.MountPropagation != nil {
			allErrs = append(allErrs, validateMountPropagation(mnt.MountPropagation, container, fldPath.Child("mountPropagation"))...)
		}
	}
	return allErrs
}

func ValidateVolumeDevices(devices []core.VolumeDevice, volmounts map[string]string, volumes map[string]core.VolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	devicepath := sets.NewString()
	devicename := sets.NewString()

	if devices != nil && !utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("volumeDevices"), "Container volumeDevices is disabled by feature-gate"))
		return allErrs
	}
	if devices != nil {
		for i, dev := range devices {
			idxPath := fldPath.Index(i)
			devName := dev.Name
			devPath := dev.DevicePath
			didMatch, isPVC := isMatchedDevice(devName, volumes)
			if len(devName) == 0 {
				allErrs = append(allErrs, field.Required(idxPath.Child("name"), ""))
			}
			if devicename.Has(devName) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), devName, "must be unique"))
			}
			// Must be PersistentVolumeClaim volume source
			if didMatch && !isPVC {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), devName, "can only use volume source type of PersistentVolumeClaim for block mode"))
			}
			if !didMatch {
				allErrs = append(allErrs, field.NotFound(idxPath.Child("name"), devName))
			}
			if len(devPath) == 0 {
				allErrs = append(allErrs, field.Required(idxPath.Child("devicePath"), ""))
			}
			if devicepath.Has(devPath) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("devicePath"), devPath, "must be unique"))
			}
			if len(devPath) > 0 && len(validatePathNoBacksteps(devPath, fldPath.Child("devicePath"))) > 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("devicePath"), devPath, "can not contain backsteps ('..')"))
			} else {
				devicepath.Insert(devPath)
			}
			// check for overlap with VolumeMount
			if deviceNameAlreadyExists(devName, volmounts) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), devName, "must not already exist in volumeMounts"))
			}
			if devicePathAlreadyExists(devPath, volmounts) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("devicePath"), devPath, "must not already exist as a path in volumeMounts"))
			}
			if len(devName) > 0 {
				devicename.Insert(devName)
			}
		}
	}
	return allErrs
}

func ValidateReadOnlyPersistentDisks(volumes []core.Volume, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i := range volumes {
		vol := &volumes[i]
		idxPath := fldPath.Index(i)
		if vol.GCEPersistentDisk != nil {
			if vol.GCEPersistentDisk.ReadOnly == false {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("gcePersistentDisk", "readOnly"), false, "must be true for replicated pods > 1; GCE PD can only be mounted on multiple machines if it is read-only"))
			}
		}
		// TODO: What to do for AWS?  It doesn't support replicas
	}
	return allErrs
}

// ValidatePersistentVolumeClaimUpdate validates an update to a PersistentVolumeClaim
func ValidatePersistentVolumeClaimUpdate(newPvc, oldPvc *core.PersistentVolumeClaim) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPvc.ObjectMeta, &oldPvc.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePersistentVolumeClaim(newPvc)...)
	newPvcClone := newPvc.DeepCopy()
	oldPvcClone := oldPvc.DeepCopy()

	// PVController needs to update PVC.Spec w/ VolumeName.
	// Claims are immutable in order to enforce quota, range limits, etc. without gaming the system.
	if len(oldPvc.Spec.VolumeName) == 0 {
		// volumeName changes are allowed once.
		oldPvcClone.Spec.VolumeName = newPvcClone.Spec.VolumeName
	}

	if validateStorageClassUpgrade(oldPvcClone.Annotations, newPvcClone.Annotations,
		oldPvcClone.Spec.StorageClassName, newPvcClone.Spec.StorageClassName) {
		newPvcClone.Spec.StorageClassName = nil
		metav1.SetMetaDataAnnotation(&newPvcClone.ObjectMeta, core.BetaStorageClassAnnotation, oldPvcClone.Annotations[core.BetaStorageClassAnnotation])
	} else {
		// storageclass annotation should be immutable after creation
		// TODO: remove Beta when no longer needed
		allErrs = append(allErrs, ValidateImmutableAnnotation(newPvc.ObjectMeta.Annotations[v1.BetaStorageClassAnnotation], oldPvc.ObjectMeta.Annotations[v1.BetaStorageClassAnnotation], v1.BetaStorageClassAnnotation, field.NewPath("metadata"))...)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		// lets make sure storage values are same.
		if newPvc.Status.Phase == core.ClaimBound && newPvcClone.Spec.Resources.Requests != nil {
			newPvcClone.Spec.Resources.Requests["storage"] = oldPvc.Spec.Resources.Requests["storage"]
		}

		oldSize := oldPvc.Spec.Resources.Requests["storage"]
		newSize := newPvc.Spec.Resources.Requests["storage"]

		if !apiequality.Semantic.DeepEqual(newPvcClone.Spec, oldPvcClone.Spec) {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "is immutable after creation except resources.requests for bound claims"))
		}
		if newSize.Cmp(oldSize) < 0 {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "resources", "requests", "storage"), "field can not be less than previous value"))
		}

	} else {
		// changes to Spec are not allowed, but updates to label/and some annotations are OK.
		// no-op updates pass validation.
		if !apiequality.Semantic.DeepEqual(newPvcClone.Spec, oldPvcClone.Spec) {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "field is immutable after creation"))
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.BlockVolume) {
		allErrs = append(allErrs, ValidateImmutableField(newPvc.Spec.VolumeMode, oldPvc.Spec.VolumeMode, field.NewPath("volumeMode"))...)
	}
	return allErrs
}

func isMatchedDevice(name string, volumes map[string]core.VolumeSource) (bool, bool) {
	if source, ok := volumes[name]; ok {
		if source.PersistentVolumeClaim != nil {
			return true, true
		} else {
			return true, false
		}
	} else {
		return false, false
	}
}

func mountNameAlreadyExists(name string, devices map[string]string) bool {
	if _, ok := devices[name]; ok {
		return true
	} else {
		return false
	}
}

func mountPathAlreadyExists(mountPath string, devices map[string]string) bool {
	for _, devPath := range devices {
		if mountPath == devPath {
			return true
		}
	}

	return false
}

func deviceNameAlreadyExists(name string, mounts map[string]string) bool {
	if _, ok := mounts[name]; ok {
		return true
	} else {
		return false
	}
}

func devicePathAlreadyExists(devicePath string, mounts map[string]string) bool {
	for _, mountPath := range mounts {
		if mountPath == devicePath {
			return true
		}
	}

	return false
}

func validateVolumeSource(source *core.VolumeSource, fldPath *field.Path, volName string) field.ErrorList {
	numVolumes := 0
	allErrs := field.ErrorList{}
	if source.EmptyDir != nil {
		numVolumes++
		if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
			if source.EmptyDir.SizeLimit != nil && source.EmptyDir.SizeLimit.Cmp(resource.Quantity{}) != 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("emptyDir").Child("sizeLimit"), "SizeLimit field disabled by feature-gate for EmptyDir volumes"))
			}
		} else {
			if source.EmptyDir.SizeLimit != nil && source.EmptyDir.SizeLimit.Cmp(resource.Quantity{}) < 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("emptyDir").Child("sizeLimit"), "SizeLimit field must be a valid resource quantity"))
			}
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.HugePages) && source.EmptyDir.Medium == core.StorageMediumHugePages {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("emptyDir").Child("medium"), "HugePages medium is disabled by feature-gate for EmptyDir volumes"))
		}
	}
	if source.HostPath != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("hostPath"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateHostPathVolumeSource(source.HostPath, fldPath.Child("hostPath"))...)
		}
	}
	if source.GitRepo != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("gitRepo"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateGitRepoVolumeSource(source.GitRepo, fldPath.Child("gitRepo"))...)
		}
	}
	if source.GCEPersistentDisk != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("gcePersistentDisk"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateGCEPersistentDiskVolumeSource(source.GCEPersistentDisk, fldPath.Child("persistentDisk"))...)
		}
	}
	if source.AWSElasticBlockStore != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("awsElasticBlockStore"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateAWSElasticBlockStoreVolumeSource(source.AWSElasticBlockStore, fldPath.Child("awsElasticBlockStore"))...)
		}
	}
	if source.Secret != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("secret"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateSecretVolumeSource(source.Secret, fldPath.Child("secret"))...)
		}
	}
	if source.NFS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("nfs"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateNFSVolumeSource(source.NFS, fldPath.Child("nfs"))...)
		}
	}
	if source.ISCSI != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("iscsi"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateISCSIVolumeSource(source.ISCSI, fldPath.Child("iscsi"))...)
		}
		if source.ISCSI.InitiatorName != nil && len(volName+":"+source.ISCSI.TargetPortal) > 64 {
			tooLongErr := "Total length of <volume name>:<iscsi.targetPortal> must be under 64 characters if iscsi.initiatorName is specified."
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), volName, tooLongErr))
		}
	}
	if source.Glusterfs != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("glusterfs"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateGlusterfsVolumeSource(source.Glusterfs, fldPath.Child("glusterfs"))...)
		}
	}
	if source.Flocker != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("flocker"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateFlockerVolumeSource(source.Flocker, fldPath.Child("flocker"))...)
		}
	}
	if source.PersistentVolumeClaim != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("persistentVolumeClaim"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validatePersistentClaimVolumeSource(source.PersistentVolumeClaim, fldPath.Child("persistentVolumeClaim"))...)
		}
	}
	if source.RBD != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("rbd"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateRBDVolumeSource(source.RBD, fldPath.Child("rbd"))...)
		}
	}
	if source.Cinder != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("cinder"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateCinderVolumeSource(source.Cinder, fldPath.Child("cinder"))...)
		}
	}
	if source.CephFS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("cephFS"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateCephFSVolumeSource(source.CephFS, fldPath.Child("cephfs"))...)
		}
	}
	if source.Quobyte != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("quobyte"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateQuobyteVolumeSource(source.Quobyte, fldPath.Child("quobyte"))...)
		}
	}
	if source.DownwardAPI != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("downwarAPI"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateDownwardAPIVolumeSource(source.DownwardAPI, fldPath.Child("downwardAPI"))...)
		}
	}
	if source.FC != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("fc"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateFCVolumeSource(source.FC, fldPath.Child("fc"))...)
		}
	}
	if source.FlexVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("flexVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateFlexVolumeSource(source.FlexVolume, fldPath.Child("flexVolume"))...)
		}
	}
	if source.ConfigMap != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("configMap"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateConfigMapVolumeSource(source.ConfigMap, fldPath.Child("configMap"))...)
		}
	}

	if source.AzureFile != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("azureFile"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateAzureFile(source.AzureFile, fldPath.Child("azureFile"))...)
		}
	}

	if source.VsphereVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("vsphereVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateVsphereVolumeSource(source.VsphereVolume, fldPath.Child("vsphereVolume"))...)
		}
	}
	if source.PhotonPersistentDisk != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("photonPersistentDisk"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validatePhotonPersistentDiskVolumeSource(source.PhotonPersistentDisk, fldPath.Child("photonPersistentDisk"))...)
		}
	}
	if source.PortworxVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("portworxVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validatePortworxVolumeSource(source.PortworxVolume, fldPath.Child("portworxVolume"))...)
		}
	}
	if source.AzureDisk != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("azureDisk"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateAzureDisk(source.AzureDisk, fldPath.Child("azureDisk"))...)
		}
	}
	if source.StorageOS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("storageos"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateStorageOSVolumeSource(source.StorageOS, fldPath.Child("storageos"))...)
		}
	}
	if source.Projected != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("projected"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateProjectedVolumeSource(source.Projected, fldPath.Child("projected"))...)
		}
	}
	if source.ScaleIO != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("scaleIO"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateScaleIOVolumeSource(source.ScaleIO, fldPath.Child("scaleIO"))...)
		}
	}

	if numVolumes == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "must specify a volume type"))
	}

	return allErrs
}

func validateHostPathVolumeSource(hostPath *core.HostPathVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(hostPath.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
		return allErrs
	}

	allErrs = append(allErrs, validatePathNoBacksteps(hostPath.Path, fldPath.Child("path"))...)
	allErrs = append(allErrs, validateHostPathType(hostPath.Type, fldPath.Child("type"))...)
	return allErrs
}

func validateGitRepoVolumeSource(gitRepo *core.GitRepoVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(gitRepo.Repository) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("repository"), ""))
	}

	pathErrs := validateLocalDescendingPath(gitRepo.Directory, fldPath.Child("directory"))
	allErrs = append(allErrs, pathErrs...)
	return allErrs
}

func validateISCSIVolumeSource(iscsi *core.ISCSIVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(iscsi.TargetPortal) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetPortal"), ""))
	}
	if len(iscsi.IQN) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("iqn"), ""))
	} else {
		if !strings.HasPrefix(iscsi.IQN, "iqn") && !strings.HasPrefix(iscsi.IQN, "eui") && !strings.HasPrefix(iscsi.IQN, "naa") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format starting with iqn, eui, or naa"))
		} else if strings.HasPrefix(iscsi.IQN, "iqn") && !iscsiInitiatorIqnRegex.MatchString(iscsi.IQN) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		} else if strings.HasPrefix(iscsi.IQN, "eui") && !iscsiInitiatorEuiRegex.MatchString(iscsi.IQN) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		} else if strings.HasPrefix(iscsi.IQN, "naa") && !iscsiInitiatorNaaRegex.MatchString(iscsi.IQN) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		}
	}
	if iscsi.Lun < 0 || iscsi.Lun > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("lun"), iscsi.Lun, validation.InclusiveRangeError(0, 255)))
	}
	if (iscsi.DiscoveryCHAPAuth || iscsi.SessionCHAPAuth) && iscsi.SecretRef == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretRef"), ""))
	}
	if iscsi.InitiatorName != nil {
		initiator := *iscsi.InitiatorName
		if !strings.HasPrefix(initiator, "iqn") && !strings.HasPrefix(initiator, "eui") && !strings.HasPrefix(initiator, "naa") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format starting with iqn, eui, or naa"))
		}
		if strings.HasPrefix(initiator, "iqn") && !iscsiInitiatorIqnRegex.MatchString(initiator) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		} else if strings.HasPrefix(initiator, "eui") && !iscsiInitiatorEuiRegex.MatchString(initiator) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		} else if strings.HasPrefix(initiator, "naa") && !iscsiInitiatorNaaRegex.MatchString(initiator) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		}
	}
	return allErrs
}

func validateISCSIPersistentVolumeSource(iscsi *core.ISCSIPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(iscsi.TargetPortal) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetPortal"), ""))
	}
	if len(iscsi.IQN) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("iqn"), ""))
	} else {
		if !strings.HasPrefix(iscsi.IQN, "iqn") && !strings.HasPrefix(iscsi.IQN, "eui") && !strings.HasPrefix(iscsi.IQN, "naa") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		} else if strings.HasPrefix(iscsi.IQN, "iqn") && !iscsiInitiatorIqnRegex.MatchString(iscsi.IQN) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		} else if strings.HasPrefix(iscsi.IQN, "eui") && !iscsiInitiatorEuiRegex.MatchString(iscsi.IQN) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		} else if strings.HasPrefix(iscsi.IQN, "naa") && !iscsiInitiatorNaaRegex.MatchString(iscsi.IQN) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("iqn"), iscsi.IQN, "must be valid format"))
		}
	}
	if iscsi.Lun < 0 || iscsi.Lun > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("lun"), iscsi.Lun, validation.InclusiveRangeError(0, 255)))
	}
	if (iscsi.DiscoveryCHAPAuth || iscsi.SessionCHAPAuth) && iscsi.SecretRef == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretRef"), ""))
	}
	if iscsi.SecretRef != nil {
		if len(iscsi.SecretRef.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("secretRef", "name"), ""))
		}
	}
	if iscsi.InitiatorName != nil {
		initiator := *iscsi.InitiatorName
		if !strings.HasPrefix(initiator, "iqn") && !strings.HasPrefix(initiator, "eui") && !strings.HasPrefix(initiator, "naa") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		}
		if strings.HasPrefix(initiator, "iqn") && !iscsiInitiatorIqnRegex.MatchString(initiator) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		} else if strings.HasPrefix(initiator, "eui") && !iscsiInitiatorEuiRegex.MatchString(initiator) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		} else if strings.HasPrefix(initiator, "naa") && !iscsiInitiatorNaaRegex.MatchString(initiator) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("initiatorname"), initiator, "must be valid format"))
		}
	}
	return allErrs
}

func validateFCVolumeSource(fc *core.FCVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(fc.TargetWWNs) < 1 && len(fc.WWIDs) < 1 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetWWNs"), "must specify either targetWWNs or wwids, but not both"))
	}

	if len(fc.TargetWWNs) != 0 && len(fc.WWIDs) != 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("targetWWNs"), fc.TargetWWNs, "targetWWNs and wwids can not be specified simultaneously"))
	}

	if len(fc.TargetWWNs) != 0 {
		if fc.Lun == nil {
			allErrs = append(allErrs, field.Required(fldPath.Child("lun"), "lun is required if targetWWNs is specified"))
		} else {
			if *fc.Lun < 0 || *fc.Lun > 255 {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("lun"), fc.Lun, validation.InclusiveRangeError(0, 255)))
			}
		}
	}
	return allErrs
}

func validateGCEPersistentDiskVolumeSource(pd *core.GCEPersistentDiskVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(pd.PDName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("pdName"), ""))
	}
	if pd.Partition < 0 || pd.Partition > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("partition"), pd.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateAWSElasticBlockStoreVolumeSource(PD *core.AWSElasticBlockStoreVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(PD.VolumeID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeID"), ""))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("partition"), PD.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateSecretVolumeSource(secretSource *core.SecretVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(secretSource.SecretName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretName"), ""))
	}

	secretMode := secretSource.DefaultMode
	if secretMode != nil && (*secretMode > 0777 || *secretMode < 0) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("defaultMode"), *secretMode, fileModeErrorMsg))
	}

	itemsPath := fldPath.Child("items")
	for i, kp := range secretSource.Items {
		itemPath := itemsPath.Index(i)
		allErrs = append(allErrs, validateKeyToPath(&kp, itemPath)...)
	}
	return allErrs
}

func validateConfigMapVolumeSource(configMapSource *core.ConfigMapVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(configMapSource.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}

	configMapMode := configMapSource.DefaultMode
	if configMapMode != nil && (*configMapMode > 0777 || *configMapMode < 0) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("defaultMode"), *configMapMode, fileModeErrorMsg))
	}

	itemsPath := fldPath.Child("items")
	for i, kp := range configMapSource.Items {
		itemPath := itemsPath.Index(i)
		allErrs = append(allErrs, validateKeyToPath(&kp, itemPath)...)
	}
	return allErrs
}

func validateKeyToPath(kp *core.KeyToPath, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(kp.Key) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("key"), ""))
	}
	if len(kp.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	allErrs = append(allErrs, validateLocalNonReservedPath(kp.Path, fldPath.Child("path"))...)
	if kp.Mode != nil && (*kp.Mode > 0777 || *kp.Mode < 0) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("mode"), *kp.Mode, fileModeErrorMsg))
	}

	return allErrs
}

func validatePersistentClaimVolumeSource(claim *core.PersistentVolumeClaimVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(claim.ClaimName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("claimName"), ""))
	}
	return allErrs
}

func validateNFSVolumeSource(nfs *core.NFSVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(nfs.Server) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("server"), ""))
	}
	if len(nfs.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	if !path.IsAbs(nfs.Path) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("path"), nfs.Path, "must be an absolute path"))
	}
	return allErrs
}

func validateQuobyteVolumeSource(quobyte *core.QuobyteVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(quobyte.Registry) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("registry"), "must be a host:port pair or multiple pairs separated by commas"))
	} else {
		for _, hostPortPair := range strings.Split(quobyte.Registry, ",") {
			if _, _, err := net.SplitHostPort(hostPortPair); err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("registry"), quobyte.Registry, "must be a host:port pair or multiple pairs separated by commas"))
			}
		}
	}

	if len(quobyte.Volume) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volume"), ""))
	}
	return allErrs
}

func validateGlusterfsVolumeSource(glusterfs *core.GlusterfsVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(glusterfs.EndpointsName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("endpoints"), ""))
	}
	if len(glusterfs.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	return allErrs
}

func validateFlockerVolumeSource(flocker *core.FlockerVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(flocker.DatasetName) == 0 && len(flocker.DatasetUUID) == 0 {
		//TODO: consider adding a RequiredOneOf() error for this and similar cases
		allErrs = append(allErrs, field.Required(fldPath, "one of datasetName and datasetUUID is required"))
	}
	if len(flocker.DatasetName) != 0 && len(flocker.DatasetUUID) != 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "resource", "datasetName and datasetUUID can not be specified simultaneously"))
	}
	if strings.Contains(flocker.DatasetName, "/") {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("datasetName"), flocker.DatasetName, "must not contain '/'"))
	}
	return allErrs
}

var validVolumeDownwardAPIFieldPathExpressions = sets.NewString(
	"metadata.name",
	"metadata.namespace",
	"metadata.labels",
	"metadata.annotations",
	"metadata.uid")

func validateDownwardAPIVolumeFile(file *core.DownwardAPIVolumeFile, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(file.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	allErrs = append(allErrs, validateLocalNonReservedPath(file.Path, fldPath.Child("path"))...)
	if file.FieldRef != nil {
		allErrs = append(allErrs, validateObjectFieldSelector(file.FieldRef, &validVolumeDownwardAPIFieldPathExpressions, fldPath.Child("fieldRef"))...)
		if file.ResourceFieldRef != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, "resource", "fieldRef and resourceFieldRef can not be specified simultaneously"))
		}
	} else if file.ResourceFieldRef != nil {
		allErrs = append(allErrs, validateContainerResourceFieldSelector(file.ResourceFieldRef, &validContainerResourceFieldPathExpressions, fldPath.Child("resourceFieldRef"), true)...)
	} else {
		allErrs = append(allErrs, field.Required(fldPath, "one of fieldRef and resourceFieldRef is required"))
	}
	if file.Mode != nil && (*file.Mode > 0777 || *file.Mode < 0) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("mode"), *file.Mode, fileModeErrorMsg))
	}

	return allErrs
}

func validateDownwardAPIVolumeSource(downwardAPIVolume *core.DownwardAPIVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	downwardAPIMode := downwardAPIVolume.DefaultMode
	if downwardAPIMode != nil && (*downwardAPIMode > 0777 || *downwardAPIMode < 0) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("defaultMode"), *downwardAPIMode, fileModeErrorMsg))
	}

	for _, file := range downwardAPIVolume.Items {
		allErrs = append(allErrs, validateDownwardAPIVolumeFile(&file, fldPath)...)
	}
	return allErrs
}

func validateProjectionSources(projection *core.ProjectedVolumeSource, projectionMode *int32, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allPaths := sets.String{}

	for _, source := range projection.Sources {
		numSources := 0
		if source.Secret != nil {
			if numSources > 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("secret"), "may not specify more than 1 volume type"))
			} else {
				numSources++
				if len(source.Secret.Name) == 0 {
					allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
				}
				itemsPath := fldPath.Child("items")
				for i, kp := range source.Secret.Items {
					itemPath := itemsPath.Index(i)
					allErrs = append(allErrs, validateKeyToPath(&kp, itemPath)...)
					if len(kp.Path) > 0 {
						curPath := kp.Path
						if !allPaths.Has(curPath) {
							allPaths.Insert(curPath)
						} else {
							allErrs = append(allErrs, field.Invalid(fldPath, source.Secret.Name, "conflicting duplicate paths"))
						}
					}
				}
			}
		}
		if source.ConfigMap != nil {
			if numSources > 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("configMap"), "may not specify more than 1 volume type"))
			} else {
				numSources++
				if len(source.ConfigMap.Name) == 0 {
					allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
				}
				itemsPath := fldPath.Child("items")
				for i, kp := range source.ConfigMap.Items {
					itemPath := itemsPath.Index(i)
					allErrs = append(allErrs, validateKeyToPath(&kp, itemPath)...)
					if len(kp.Path) > 0 {
						curPath := kp.Path
						if !allPaths.Has(curPath) {
							allPaths.Insert(curPath)
						} else {
							allErrs = append(allErrs, field.Invalid(fldPath, source.ConfigMap.Name, "conflicting duplicate paths"))
						}

					}
				}
			}
		}
		if source.DownwardAPI != nil {
			if numSources > 0 {
				allErrs = append(allErrs, field.Forbidden(fldPath.Child("downwardAPI"), "may not specify more than 1 volume type"))
			} else {
				numSources++
				for _, file := range source.DownwardAPI.Items {
					allErrs = append(allErrs, validateDownwardAPIVolumeFile(&file, fldPath.Child("downwardAPI"))...)
					if len(file.Path) > 0 {
						curPath := file.Path
						if !allPaths.Has(curPath) {
							allPaths.Insert(curPath)
						} else {
							allErrs = append(allErrs, field.Invalid(fldPath, curPath, "conflicting duplicate paths"))
						}

					}
				}
			}
		}
	}
	return allErrs
}

func validateProjectedVolumeSource(projection *core.ProjectedVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	projectionMode := projection.DefaultMode
	if projectionMode != nil && (*projectionMode > 0777 || *projectionMode < 0) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("defaultMode"), *projectionMode, fileModeErrorMsg))
	}

	allErrs = append(allErrs, validateProjectionSources(projection, projectionMode, fldPath)...)
	return allErrs
}

var supportedHostPathTypes = sets.NewString(
	string(core.HostPathUnset),
	string(core.HostPathDirectoryOrCreate),
	string(core.HostPathDirectory),
	string(core.HostPathFileOrCreate),
	string(core.HostPathFile),
	string(core.HostPathSocket),
	string(core.HostPathCharDev),
	string(core.HostPathBlockDev))

func validateHostPathType(hostPathType *core.HostPathType, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if hostPathType != nil && !supportedHostPathTypes.Has(string(*hostPathType)) {
		allErrs = append(allErrs, field.NotSupported(fldPath, hostPathType, supportedHostPathTypes.List()))
	}

	return allErrs
}

// This validate will make sure targetPath:
// 1. is not abs path
// 2. does not have any element which is ".."
func validateLocalDescendingPath(targetPath string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if path.IsAbs(targetPath) {
		allErrs = append(allErrs, field.Invalid(fldPath, targetPath, "must be a relative path"))
	}

	allErrs = append(allErrs, validatePathNoBacksteps(targetPath, fldPath)...)

	return allErrs
}

// validatePathNoBacksteps makes sure the targetPath does not have any `..` path elements when split
//
// This assumes the OS of the apiserver and the nodes are the same. The same check should be done
// on the node to ensure there are no backsteps.
func validatePathNoBacksteps(targetPath string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	parts := strings.Split(filepath.ToSlash(targetPath), "/")
	for _, item := range parts {
		if item == ".." {
			allErrs = append(allErrs, field.Invalid(fldPath, targetPath, "must not contain '..'"))
			break // even for `../../..`, one error is sufficient to make the point
		}
	}
	return allErrs
}

// validateMountPropagation verifies that MountPropagation field is valid and
// allowed for given container.
func validateMountPropagation(mountPropagation *core.MountPropagationMode, container *core.Container, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if mountPropagation == nil {
		return allErrs
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.MountPropagation) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "mount propagation is disabled by feature-gate"))
		return allErrs
	}

	supportedMountPropagations := sets.NewString(string(core.MountPropagationBidirectional), string(core.MountPropagationHostToContainer))
	if !supportedMountPropagations.Has(string(*mountPropagation)) {
		allErrs = append(allErrs, field.NotSupported(fldPath, *mountPropagation, supportedMountPropagations.List()))
	}

	if container == nil {
		// The container is not available yet, e.g. during validation of
		// PodPreset. Stop validation now, Pod validation will refuse final
		// Pods with Bidirectional propagation in non-privileged containers.
		return allErrs
	}

	privileged := container.SecurityContext != nil && container.SecurityContext.Privileged != nil && *container.SecurityContext.Privileged
	if *mountPropagation == core.MountPropagationBidirectional && !privileged {
		allErrs = append(allErrs, field.Forbidden(fldPath, "Bidirectional mount propagation is available only to privileged containers"))
	}
	return allErrs
}

// This validate will make sure targetPath:
// 1. is not abs path
// 2. does not contain any '..' elements
// 3. does not start with '..'
func validateLocalNonReservedPath(targetPath string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validateLocalDescendingPath(targetPath, fldPath)...)
	// Don't report this error if the check for .. elements already caught it.
	if strings.HasPrefix(targetPath, "..") && !strings.HasPrefix(targetPath, "../") {
		allErrs = append(allErrs, field.Invalid(fldPath, targetPath, "must not start with '..'"))
	}
	return allErrs
}

func validateRBDVolumeSource(rbd *core.RBDVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(rbd.CephMonitors) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("monitors"), ""))
	}
	if len(rbd.RBDImage) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("image"), ""))
	}
	return allErrs
}

func validateRBDPersistentVolumeSource(rbd *core.RBDPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(rbd.CephMonitors) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("monitors"), ""))
	}
	if len(rbd.RBDImage) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("image"), ""))
	}
	return allErrs
}

func validateCinderVolumeSource(cd *core.CinderVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cd.VolumeID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeID"), ""))
	}
	return allErrs
}

func validateCephFSVolumeSource(cephfs *core.CephFSVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cephfs.Monitors) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("monitors"), ""))
	}
	return allErrs
}

func validateCephFSPersistentVolumeSource(cephfs *core.CephFSPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cephfs.Monitors) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("monitors"), ""))
	}
	return allErrs
}

func validateFlexVolumeSource(fv *core.FlexVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(fv.Driver) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), ""))
	}

	// Make sure user-specified options don't use kubernetes namespaces
	for k := range fv.Options {
		namespace := k
		if parts := strings.SplitN(k, "/", 2); len(parts) == 2 {
			namespace = parts[0]
		}
		normalized := "." + strings.ToLower(namespace)
		if strings.HasSuffix(normalized, ".kubernetes.io") || strings.HasSuffix(normalized, ".k8s.io") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("options").Key(k), k, "kubernetes.io and k8s.io namespaces are reserved"))
		}
	}

	return allErrs
}

func validateFlexPersistentVolumeSource(fv *core.FlexPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(fv.Driver) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), ""))
	}

	// Make sure user-specified options don't use kubernetes namespaces
	for k := range fv.Options {
		namespace := k
		if parts := strings.SplitN(k, "/", 2); len(parts) == 2 {
			namespace = parts[0]
		}
		normalized := "." + strings.ToLower(namespace)
		if strings.HasSuffix(normalized, ".kubernetes.io") || strings.HasSuffix(normalized, ".k8s.io") {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("options").Key(k), k, "kubernetes.io and k8s.io namespaces are reserved"))
		}
	}

	return allErrs
}

func validateAzureFile(azure *core.AzureFileVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if azure.SecretName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretName"), ""))
	}
	if azure.ShareName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("shareName"), ""))
	}
	return allErrs
}

func validateAzureFilePV(azure *core.AzureFilePersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if azure.SecretName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretName"), ""))
	}
	if azure.ShareName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("shareName"), ""))
	}
	if azure.SecretNamespace != nil {
		if len(*azure.SecretNamespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("secretNamespace"), ""))
		}
	}
	return allErrs
}

func validateAzureDisk(azure *core.AzureDiskVolumeSource, fldPath *field.Path) field.ErrorList {
	var supportedCachingModes = sets.NewString(string(core.AzureDataDiskCachingNone), string(core.AzureDataDiskCachingReadOnly), string(core.AzureDataDiskCachingReadWrite))
	var supportedDiskKinds = sets.NewString(string(core.AzureSharedBlobDisk), string(core.AzureDedicatedBlobDisk), string(core.AzureManagedDisk))

	diskUriSupportedManaged := []string{"/subscriptions/{sub-id}/resourcegroups/{group-name}/providers/microsoft.compute/disks/{disk-id}"}
	diskUriSupportedblob := []string{"https://{account-name}.blob.core.windows.net/{container-name}/{disk-name}.vhd"}

	allErrs := field.ErrorList{}
	if azure.DiskName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("diskName"), ""))
	}

	if azure.DataDiskURI == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("diskURI"), ""))
	}

	if azure.CachingMode != nil && !supportedCachingModes.Has(string(*azure.CachingMode)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("cachingMode"), *azure.CachingMode, supportedCachingModes.List()))
	}

	if azure.Kind != nil && !supportedDiskKinds.Has(string(*azure.Kind)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("kind"), *azure.Kind, supportedDiskKinds.List()))
	}

	// validate that DiskUri is the correct format
	if azure.Kind != nil && *azure.Kind == core.AzureManagedDisk && strings.Index(azure.DataDiskURI, "/subscriptions/") != 0 {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("diskURI"), azure.DataDiskURI, diskUriSupportedManaged))
	}

	if azure.Kind != nil && *azure.Kind != core.AzureManagedDisk && strings.Index(azure.DataDiskURI, "https://") != 0 {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("diskURI"), azure.DataDiskURI, diskUriSupportedblob))
	}

	return allErrs
}

func validateVsphereVolumeSource(cd *core.VsphereVirtualDiskVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cd.VolumePath) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumePath"), ""))
	}
	return allErrs
}

func validatePhotonPersistentDiskVolumeSource(cd *core.PhotonPersistentDiskVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cd.PdID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("pdID"), ""))
	}
	return allErrs
}

func validatePortworxVolumeSource(pwx *core.PortworxVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(pwx.VolumeID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeID"), ""))
	}
	return allErrs
}

func validateScaleIOVolumeSource(sio *core.ScaleIOVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if sio.Gateway == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("gateway"), ""))
	}
	if sio.System == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("system"), ""))
	}
	if sio.VolumeName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeName"), ""))
	}
	return allErrs
}

func validateScaleIOPersistentVolumeSource(sio *core.ScaleIOPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if sio.Gateway == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("gateway"), ""))
	}
	if sio.System == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("system"), ""))
	}
	if sio.VolumeName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeName"), ""))
	}
	return allErrs
}

func validateLocalVolumeSource(ls *core.LocalVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if ls.Path == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
		return allErrs
	}

	if !path.IsAbs(ls.Path) {
		allErrs = append(allErrs, field.Invalid(fldPath, ls.Path, "must be an absolute path"))
	}
	allErrs = append(allErrs, validatePathNoBacksteps(ls.Path, fldPath.Child("path"))...)
	return allErrs
}

func validateStorageOSVolumeSource(storageos *core.StorageOSVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(storageos.VolumeName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeName"), ""))
	} else {
		allErrs = append(allErrs, ValidateDNS1123Label(storageos.VolumeName, fldPath.Child("volumeName"))...)
	}
	if len(storageos.VolumeNamespace) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(storageos.VolumeNamespace, fldPath.Child("volumeNamespace"))...)
	}
	if storageos.SecretRef != nil {
		if len(storageos.SecretRef.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("secretRef", "name"), ""))
		}
	}
	return allErrs
}

func validateStorageOSPersistentVolumeSource(storageos *core.StorageOSPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(storageos.VolumeName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeName"), ""))
	} else {
		allErrs = append(allErrs, ValidateDNS1123Label(storageos.VolumeName, fldPath.Child("volumeName"))...)
	}
	if len(storageos.VolumeNamespace) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(storageos.VolumeNamespace, fldPath.Child("volumeNamespace"))...)
	}
	if storageos.SecretRef != nil {
		if len(storageos.SecretRef.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("secretRef", "name"), ""))
		}
		if len(storageos.SecretRef.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("secretRef", "namespace"), ""))
		}
	}
	return allErrs
}

func validateCSIPersistentVolumeSource(csi *core.CSIPersistentVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIPersistentVolume) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "CSIPersistentVolume disabled by feature-gate"))
	}

	if len(csi.Driver) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), ""))
	}

	if len(csi.Driver) > 63 {
		allErrs = append(allErrs, field.TooLong(fldPath.Child("driver"), csi.Driver, 63))
	}

	if !csiDriverNameRexp.MatchString(csi.Driver) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("driver"), csi.Driver, validation.RegexError(csiDriverNameRexpErrMsg, csiDriverNameRexpFmt, "csi-hostpath")))
	}

	if len(csi.VolumeHandle) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeHandle"), ""))
	}

	if csi.ControllerPublishSecretRef != nil {
		if len(csi.ControllerPublishSecretRef.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("controllerPublishSecretRef", "name"), ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(csi.ControllerPublishSecretRef.Name, fldPath.Child("name"))...)
		}
		if len(csi.ControllerPublishSecretRef.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("controllerPublishSecretRef", "namespace"), ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(csi.ControllerPublishSecretRef.Namespace, fldPath.Child("namespace"))...)
		}
	}

	if csi.NodePublishSecretRef != nil {
		if len(csi.NodePublishSecretRef.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("nodePublishSecretRef ", "name"), ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(csi.NodePublishSecretRef.Name, fldPath.Child("name"))...)
		}
		if len(csi.NodePublishSecretRef.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("nodePublishSecretRef ", "namespace"), ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(csi.NodePublishSecretRef.Namespace, fldPath.Child("namespace"))...)
		}
	}

	if csi.NodeStageSecretRef != nil {
		if len(csi.NodeStageSecretRef.Name) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("nodeStageSecretRef", "name"), ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(csi.NodeStageSecretRef.Name, fldPath.Child("name"))...)
		}
		if len(csi.NodeStageSecretRef.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("nodeStageSecretRef", "namespace"), ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(csi.NodeStageSecretRef.Namespace, fldPath.Child("namespace"))...)
		}
	}

	return allErrs
}

// Provide an upgrade path from PVC with storage class specified in beta
// annotation to storage class specified in attribute. We allow update of
// StorageClassName only if following four conditions are met at the same time:
// 1. The old pvc's StorageClassAnnotation is set
// 2. The old pvc's StorageClassName is not set
// 3. The new pvc's StorageClassName is set and equal to the old value in annotation
// 4. If the new pvc's StorageClassAnnotation is set,it must be equal to the old pv/pvc's StorageClassAnnotation
func validateStorageClassUpgrade(oldAnnotations, newAnnotations map[string]string, oldScName, newScName *string) bool {
	oldSc, oldAnnotationExist := oldAnnotations[core.BetaStorageClassAnnotation]
	newScInAnnotation, newAnnotationExist := newAnnotations[core.BetaStorageClassAnnotation]
	return oldAnnotationExist /* condition 1 */ &&
		oldScName == nil /* condition 2*/ &&
		(newScName != nil && *newScName == oldSc) /* condition 3 */ &&
		(!newAnnotationExist || newScInAnnotation == oldSc) /* condition 4 */
}

func validateBasicResource(quantity resource.Quantity, fldPath *field.Path) field.ErrorList {
	if quantity.Value() < 0 {
		return field.ErrorList{field.Invalid(fldPath, quantity.Value(), "must be a valid resource quantity")}
	}
	return field.ErrorList{}
}

// validateVolumeNodeAffinity tests that the PersistentVolume.NodeAffinity has valid data
// returns:
// - true if volumeNodeAffinity is set
// - errorList if there are validation errors
func validateVolumeNodeAffinity(nodeAffinity *core.VolumeNodeAffinity, fldPath *field.Path) (bool, field.ErrorList) {
	allErrs := field.ErrorList{}

	if nodeAffinity == nil {
		return false, allErrs
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		allErrs = append(allErrs, field.Forbidden(fldPath, "Volume node affinity is disabled by feature-gate"))
	}

	if nodeAffinity.Required != nil {
		allErrs = append(allErrs, ValidateNodeSelector(nodeAffinity.Required, fldPath.Child("required"))...)
	} else {
		allErrs = append(allErrs, field.Required(fldPath.Child("required"), "must specify required node constraints"))
	}

	return true, allErrs
}
