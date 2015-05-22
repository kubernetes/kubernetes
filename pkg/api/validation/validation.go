/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"net"
	"path"
	"reflect"
	"regexp"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	errs "github.com/GoogleCloudPlatform/kubernetes/pkg/util/fielderrors"

	"github.com/golang/glog"
)

const cIdentifierErrorMsg string = `must be a C identifier (matching regex ` + util.CIdentifierFmt + `): e.g. "my_name" or "MyName"`
const isNegativeErrorMsg string = `must be non-negative`

func intervalErrorMsg(lo, hi int) string {
	return fmt.Sprintf(`must be greater than %d and less than %d`, lo, hi)
}

var labelValueErrorMsg string = fmt.Sprintf(`must have at most %d characters, matching regex %s: e.g. "MyValue" or ""`, util.LabelValueMaxLength, util.LabelValueFmt)
var qualifiedNameErrorMsg string = fmt.Sprintf(`must be a qualified name (at most %d characters, matching regex %s), with an optional DNS subdomain prefix (at most %d characters, matching regex %s) and slash (/): e.g. "MyName" or "example.com/MyName"`, util.QualifiedNameMaxLength, util.QualifiedNameFmt, util.DNS1123SubdomainMaxLength, util.DNS1123SubdomainFmt)
var dnsSubdomainErrorMsg string = fmt.Sprintf(`must be a DNS subdomain (at most %d characters, matching regex %s): e.g. "example.com"`, util.DNS1123SubdomainMaxLength, util.DNS1123SubdomainFmt)
var dns1123LabelErrorMsg string = fmt.Sprintf(`must be a DNS label (at most %d characters, matching regex %s): e.g. "my-name"`, util.DNS1123LabelMaxLength, util.DNS1123LabelFmt)
var dns952LabelErrorMsg string = fmt.Sprintf(`must be a DNS 952 label (at most %d characters, matching regex %s): e.g. "my-name"`, util.DNS952LabelMaxLength, util.DNS952LabelFmt)
var pdPartitionErrorMsg string = intervalErrorMsg(0, 255)
var portRangeErrorMsg string = intervalErrorMsg(0, 65536)

const totalAnnotationSizeLimitB int = 64 * (1 << 10) // 64 kB

// ValidateLabels validates that a set of labels are correctly defined.
func ValidateLabels(labels map[string]string, field string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for k, v := range labels {
		if !util.IsQualifiedName(k) {
			allErrs = append(allErrs, errs.NewFieldInvalid(field, k, qualifiedNameErrorMsg))
		}
		if !util.IsValidLabelValue(v) {
			allErrs = append(allErrs, errs.NewFieldInvalid(field, v, labelValueErrorMsg))
		}
	}
	return allErrs
}

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, field string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	var totalSize int64
	for k, v := range annotations {
		if !util.IsQualifiedName(strings.ToLower(k)) {
			allErrs = append(allErrs, errs.NewFieldInvalid(field, k, qualifiedNameErrorMsg))
		}
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}
	if totalSize > (int64)(totalAnnotationSizeLimitB) {
		allErrs = append(allErrs, errs.NewFieldTooLong(field, "", totalAnnotationSizeLimitB))
	}
	return allErrs
}

// ValidateNameFunc validates that the provided name is valid for a given resource type.
// Not all resources have the same validation rules for names. Prefix is true if the
// name will have a value appended to it.
type ValidateNameFunc func(name string, prefix bool) (bool, string)

// maskTrailingDash replaces the final character of a string with a subdomain safe
// value if is a dash.
func maskTrailingDash(name string) string {
	if strings.HasSuffix(name, "-") {
		return name[:len(name)-2] + "a"
	}
	return name
}

// ValidatePodName can be used to check whether the given pod name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidatePodName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateReplicationControllerName can be used to check whether the given replication
// controller name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateReplicationControllerName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateServiceName can be used to check whether the given service name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateServiceName(name string, prefix bool) (bool, string) {
	return nameIsDNS952Label(name, prefix)
}

// ValidateNodeName can be used to check whether the given node name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateNodeName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateNamespaceName can be used to check whether the given namespace name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateNamespaceName(name string, prefix bool) (bool, string) {
	return nameIsDNSLabel(name, prefix)
}

// ValidateLimitRangeName can be used to check whether the given limit range name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateLimitRangeName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateResourceQuotaName can be used to check whether the given
// resource quota name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateResourceQuotaName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateSecretName can be used to check whether the given secret name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateSecretName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateServiceAccountName can be used to check whether the given service account name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateServiceAccountName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// ValidateEndpointsName can be used to check whether the given endpoints name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateEndpointsName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

// nameIsDNSSubdomain is a ValidateNameFunc for names that must be a DNS subdomain.
func nameIsDNSSubdomain(name string, prefix bool) (bool, string) {
	if prefix {
		name = maskTrailingDash(name)
	}
	if util.IsDNS1123Subdomain(name) {
		return true, ""
	}
	return false, dnsSubdomainErrorMsg
}

// nameIsDNSLabel is a ValidateNameFunc for names that must be a DNS 1123 label.
func nameIsDNSLabel(name string, prefix bool) (bool, string) {
	if prefix {
		name = maskTrailingDash(name)
	}
	if util.IsDNS1123Label(name) {
		return true, ""
	}
	return false, dns1123LabelErrorMsg
}

// nameIsDNS952Label is a ValidateNameFunc for names that must be a DNS 952 label.
func nameIsDNS952Label(name string, prefix bool) (bool, string) {
	if prefix {
		name = maskTrailingDash(name)
	}
	if util.IsDNS952Label(name) {
		return true, ""
	}
	return false, dns952LabelErrorMsg
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
func ValidateObjectMeta(meta *api.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if len(meta.GenerateName) != 0 {
		if ok, qualifier := nameFn(meta.GenerateName, true); !ok {
			allErrs = append(allErrs, errs.NewFieldInvalid("generateName", meta.GenerateName, qualifier))
		}
	}
	// if the generated name validates, but the calculated value does not, it's a problem with generation, and we
	// report it here. This may confuse users, but indicates a programming bug and still must be validated.
	if len(meta.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name"))
	} else {
		if ok, qualifier := nameFn(meta.Name, false); !ok {
			allErrs = append(allErrs, errs.NewFieldInvalid("name", meta.Name, qualifier))
		}
	}

	if requiresNamespace {
		if len(meta.Namespace) == 0 {
			allErrs = append(allErrs, errs.NewFieldRequired("namespace"))
		} else if ok, _ := ValidateNamespaceName(meta.Namespace, false); !ok {
			allErrs = append(allErrs, errs.NewFieldInvalid("namespace", meta.Namespace, dns1123LabelErrorMsg))
		}
	} else {
		if len(meta.Namespace) != 0 {
			allErrs = append(allErrs, errs.NewFieldInvalid("namespace", meta.Namespace, "namespace is not allowed on this type"))
		}
	}
	allErrs = append(allErrs, ValidateLabels(meta.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(meta.Annotations, "annotations")...)

	return allErrs
}

// ValidateObjectMetaUpdate validates an object's metadata when updated
func ValidateObjectMetaUpdate(old, meta *api.ObjectMeta) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	// in the event it is left empty, set it, to allow clients more flexibility
	if len(meta.UID) == 0 {
		meta.UID = old.UID
	}
	// ignore changes to timestamp
	if old.CreationTimestamp.IsZero() {
		old.CreationTimestamp = meta.CreationTimestamp
	} else {
		meta.CreationTimestamp = old.CreationTimestamp
	}

	// Reject updates that don't specify a resource version
	if meta.ResourceVersion == "" {
		allErrs = append(allErrs, errs.NewFieldInvalid("resourceVersion", meta.ResourceVersion, "resourceVersion must be specified for an update"))
	}

	if old.Name != meta.Name {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", meta.Name, "field is immutable"))
	}
	if old.Namespace != meta.Namespace {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", meta.Namespace, "field is immutable"))
	}
	if old.UID != meta.UID {
		allErrs = append(allErrs, errs.NewFieldInvalid("uid", meta.UID, "field is immutable"))
	}
	if old.CreationTimestamp != meta.CreationTimestamp {
		allErrs = append(allErrs, errs.NewFieldInvalid("creationTimestamp", meta.CreationTimestamp, "field is immutable"))
	}

	allErrs = append(allErrs, ValidateLabels(meta.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(meta.Annotations, "annotations")...)

	return allErrs
}

func validateVolumes(volumes []api.Volume) (util.StringSet, errs.ValidationErrorList) {
	allErrs := errs.ValidationErrorList{}

	allNames := util.StringSet{}
	for i, vol := range volumes {
		el := validateSource(&vol.VolumeSource).Prefix("source")
		if len(vol.Name) == 0 {
			el = append(el, errs.NewFieldRequired("name"))
		} else if !util.IsDNS1123Label(vol.Name) {
			el = append(el, errs.NewFieldInvalid("name", vol.Name, dns1123LabelErrorMsg))
		} else if allNames.Has(vol.Name) {
			el = append(el, errs.NewFieldDuplicate("name", vol.Name))
		}
		if len(el) == 0 {
			allNames.Insert(vol.Name)
		} else {
			allErrs = append(allErrs, el.PrefixIndex(i)...)
		}
	}
	return allNames, allErrs
}

func validateSource(source *api.VolumeSource) errs.ValidationErrorList {
	numVolumes := 0
	allErrs := errs.ValidationErrorList{}
	if source.HostPath != nil {
		numVolumes++
		allErrs = append(allErrs, validateHostPathVolumeSource(source.HostPath).Prefix("hostPath")...)
	}
	if source.EmptyDir != nil {
		numVolumes++
		// EmptyDirs have nothing to validate
	}
	if source.GitRepo != nil {
		numVolumes++
		allErrs = append(allErrs, validateGitRepoVolumeSource(source.GitRepo).Prefix("gitRepo")...)
	}
	if source.GCEPersistentDisk != nil {
		numVolumes++
		allErrs = append(allErrs, validateGCEPersistentDiskVolumeSource(source.GCEPersistentDisk).Prefix("persistentDisk")...)
	}
	if source.AWSElasticBlockStore != nil {
		numVolumes++
		allErrs = append(allErrs, validateAWSElasticBlockStoreVolumeSource(source.AWSElasticBlockStore).Prefix("awsElasticBlockStore")...)
	}
	if source.Secret != nil {
		numVolumes++
		allErrs = append(allErrs, validateSecretVolumeSource(source.Secret).Prefix("secret")...)
	}
	if source.NFS != nil {
		numVolumes++
		allErrs = append(allErrs, validateNFS(source.NFS).Prefix("nfs")...)
	}
	if source.ISCSI != nil {
		numVolumes++
		allErrs = append(allErrs, validateISCSIVolumeSource(source.ISCSI).Prefix("iscsi")...)
	}
	if source.Glusterfs != nil {
		numVolumes++
		allErrs = append(allErrs, validateGlusterfs(source.Glusterfs).Prefix("glusterfs")...)
	}
	if source.PersistentVolumeClaimVolumeSource != nil {
		numVolumes++
		allErrs = append(allErrs, validatePersistentClaimVolumeSource(source.PersistentVolumeClaimVolumeSource).Prefix("persistentVolumeClaim")...)
	}
	if source.RBD != nil {
		numVolumes++
		allErrs = append(allErrs, validateRBD(source.RBD).Prefix("rbd")...)
	}
	if numVolumes != 1 {
		allErrs = append(allErrs, errs.NewFieldInvalid("", source, "exactly 1 volume type is required"))
	}

	return allErrs
}

func validateHostPathVolumeSource(hostDir *api.HostPathVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if hostDir.Path == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("path"))
	}
	return allErrs
}

func validateGitRepoVolumeSource(gitRepo *api.GitRepoVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if gitRepo.Repository == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("repository"))
	}
	return allErrs
}

func validateISCSIVolumeSource(iscsi *api.ISCSIVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if iscsi.TargetPortal == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("targetPortal"))
	}
	if iscsi.IQN == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("iqn"))
	}
	if iscsi.FSType == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("fsType"))
	}
	if iscsi.Lun < 0 || iscsi.Lun > 255 {
		allErrs = append(allErrs, errs.NewFieldInvalid("lun", iscsi.Lun, ""))
	}
	return allErrs
}

func validateGCEPersistentDiskVolumeSource(PD *api.GCEPersistentDiskVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if PD.PDName == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("pdName"))
	}
	if PD.FSType == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("fsType"))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, errs.NewFieldInvalid("partition", PD.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateAWSElasticBlockStoreVolumeSource(PD *api.AWSElasticBlockStoreVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if PD.VolumeID == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("volumeID"))
	}
	if PD.FSType == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("fsType"))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, errs.NewFieldInvalid("partition", PD.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateSecretVolumeSource(secretSource *api.SecretVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if secretSource.SecretName == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("secretName"))
	}
	return allErrs
}

func validatePersistentClaimVolumeSource(claim *api.PersistentVolumeClaimVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if claim.ClaimName == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("claimName"))
	}
	return allErrs
}

func validateNFS(nfs *api.NFSVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if nfs.Server == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("server"))
	}
	if nfs.Path == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("path"))
	}
	if !path.IsAbs(nfs.Path) {
		allErrs = append(allErrs, errs.NewFieldInvalid("path", nfs.Path, "must be an absolute path"))
	}
	return allErrs
}

func validateGlusterfs(glusterfs *api.GlusterfsVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if glusterfs.EndpointsName == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("endpoints"))
	}
	if glusterfs.Path == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("path"))
	}
	return allErrs
}

func validateRBD(rbd *api.RBDVolumeSource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(rbd.CephMonitors) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("monitors"))
	}
	if rbd.RBDImage == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("image"))
	}
	if rbd.FSType == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("fsType"))
	}
	return allErrs
}

func ValidatePersistentVolumeName(name string, prefix bool) (bool, string) {
	return nameIsDNSSubdomain(name, prefix)
}

func ValidatePersistentVolume(pv *api.PersistentVolume) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pv.ObjectMeta, false, ValidatePersistentVolumeName).Prefix("metadata")...)

	if len(pv.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("persistentVolume.AccessModes"))
	}

	if len(pv.Spec.Capacity) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("persistentVolume.Capacity"))
	}

	if _, ok := pv.Spec.Capacity[api.ResourceStorage]; !ok || len(pv.Spec.Capacity) > 1 {
		allErrs = append(allErrs, errs.NewFieldInvalid("", pv.Spec.Capacity, fmt.Sprintf("only %s is expected", api.ResourceStorage)))
	}

	for _, qty := range pv.Spec.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty)...)
	}

	numVolumes := 0
	if pv.Spec.HostPath != nil {
		numVolumes++
		allErrs = append(allErrs, validateHostPathVolumeSource(pv.Spec.HostPath).Prefix("hostPath")...)
	}
	if pv.Spec.GCEPersistentDisk != nil {
		numVolumes++
		allErrs = append(allErrs, validateGCEPersistentDiskVolumeSource(pv.Spec.GCEPersistentDisk).Prefix("persistentDisk")...)
	}
	if pv.Spec.AWSElasticBlockStore != nil {
		numVolumes++
		allErrs = append(allErrs, validateAWSElasticBlockStoreVolumeSource(pv.Spec.AWSElasticBlockStore).Prefix("awsElasticBlockStore")...)
	}
	if pv.Spec.Glusterfs != nil {
		numVolumes++
		allErrs = append(allErrs, validateGlusterfs(pv.Spec.Glusterfs).Prefix("glusterfs")...)
	}
	if pv.Spec.NFS != nil {
		numVolumes++
		allErrs = append(allErrs, validateNFS(pv.Spec.NFS).Prefix("nfs")...)
	}
	if pv.Spec.RBD != nil {
		numVolumes++
		allErrs = append(allErrs, validateRBD(pv.Spec.RBD).Prefix("rbd")...)
	}
	if numVolumes != 1 {
		allErrs = append(allErrs, errs.NewFieldInvalid("", pv.Spec.PersistentVolumeSource, "exactly 1 volume type is required"))
	}
	return allErrs
}

// ValidatePersistentVolumeUpdate tests to see if the update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeUpdate(newPv, oldPv *api.PersistentVolume) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = ValidatePersistentVolume(newPv)
	newPv.Status = oldPv.Status
	return allErrs
}

// ValidatePersistentVolumeStatusUpdate tests to see if the status update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeStatusUpdate(newPv, oldPv *api.PersistentVolume) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPv.ObjectMeta, &newPv.ObjectMeta).Prefix("metadata")...)
	if newPv.ResourceVersion == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("resourceVersion"))
	}
	newPv.Spec = oldPv.Spec
	return allErrs
}

func ValidatePersistentVolumeClaim(pvc *api.PersistentVolumeClaim) errs.ValidationErrorList {
	allErrs := ValidateObjectMeta(&pvc.ObjectMeta, true, ValidatePersistentVolumeName)
	if len(pvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("persistentVolumeClaim.Spec.AccessModes", pvc.Spec.AccessModes, "at least 1 PersistentVolumeAccessMode is required"))
	}
	if _, ok := pvc.Spec.Resources.Requests[api.ResourceStorage]; !ok {
		allErrs = append(allErrs, errs.NewFieldInvalid("persistentVolumeClaim.Spec.Resources.Requests", pvc.Spec.Resources.Requests, "No Storage size specified"))
	}
	return allErrs
}

func ValidatePersistentVolumeClaimUpdate(newPvc, oldPvc *api.PersistentVolumeClaim) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = ValidatePersistentVolumeClaim(newPvc)
	newPvc.Status = oldPvc.Status
	return allErrs
}

func ValidatePersistentVolumeClaimStatusUpdate(newPvc, oldPvc *api.PersistentVolumeClaim) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPvc.ObjectMeta, &newPvc.ObjectMeta).Prefix("metadata")...)
	if newPvc.ResourceVersion == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("resourceVersion"))
	}
	if len(newPvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("persistentVolume.AccessModes"))
	}
	for _, qty := range newPvc.Status.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty)...)
	}
	newPvc.Spec = oldPvc.Spec
	return allErrs
}

var supportedPortProtocols = util.NewStringSet(string(api.ProtocolTCP), string(api.ProtocolUDP))

func validatePorts(ports []api.ContainerPort) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allNames := util.StringSet{}
	for i, port := range ports {
		pErrs := errs.ValidationErrorList{}
		if len(port.Name) > 0 {
			if len(port.Name) > util.DNS1123LabelMaxLength || !util.IsDNS1123Label(port.Name) {
				pErrs = append(pErrs, errs.NewFieldInvalid("name", port.Name, dns1123LabelErrorMsg))
			} else if allNames.Has(port.Name) {
				pErrs = append(pErrs, errs.NewFieldDuplicate("name", port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if port.ContainerPort == 0 {
			pErrs = append(pErrs, errs.NewFieldInvalid("containerPort", port.ContainerPort, portRangeErrorMsg))
		} else if !util.IsValidPortNum(port.ContainerPort) {
			pErrs = append(pErrs, errs.NewFieldInvalid("containerPort", port.ContainerPort, portRangeErrorMsg))
		}
		if port.HostPort != 0 && !util.IsValidPortNum(port.HostPort) {
			pErrs = append(pErrs, errs.NewFieldInvalid("hostPort", port.HostPort, portRangeErrorMsg))
		}
		if len(port.Protocol) == 0 {
			pErrs = append(pErrs, errs.NewFieldRequired("protocol"))
		} else if !supportedPortProtocols.Has(strings.ToUpper(string(port.Protocol))) {
			pErrs = append(pErrs, errs.NewFieldNotSupported("protocol", port.Protocol))
		}
		allErrs = append(allErrs, pErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateEnv(vars []api.EnvVar) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for i, ev := range vars {
		vErrs := errs.ValidationErrorList{}
		if len(ev.Name) == 0 {
			vErrs = append(vErrs, errs.NewFieldRequired("name"))
		} else if !util.IsCIdentifier(ev.Name) {
			vErrs = append(vErrs, errs.NewFieldInvalid("name", ev.Name, cIdentifierErrorMsg))
		}
		vErrs = append(vErrs, validateEnvVarValueFrom(ev).Prefix("valueFrom")...)
		allErrs = append(allErrs, vErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateEnvVarValueFrom(ev api.EnvVar) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if ev.ValueFrom == nil {
		return allErrs
	}

	numSources := 0

	switch {
	case ev.ValueFrom.FieldRef != nil:
		numSources++
		allErrs = append(allErrs, validateObjectFieldSelector(ev.ValueFrom.FieldRef).Prefix("fieldRef")...)
	}

	if ev.Value != "" && numSources != 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("", "", "sources cannot be specified when value is not empty"))
	}

	return allErrs
}

var validFieldPathExpressions = util.NewStringSet("metadata.name", "metadata.namespace")

func validateObjectFieldSelector(fs *api.ObjectFieldSelector) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if fs.APIVersion == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("apiVersion"))
	} else if fs.FieldPath == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("fieldPath"))
	} else {
		internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(fs.APIVersion, "Pod", fs.FieldPath, "")
		if err != nil {
			allErrs = append(allErrs, errs.NewFieldInvalid("fieldPath", fs.FieldPath, "error converting fieldPath"))
		} else if !validFieldPathExpressions.Has(internalFieldPath) {
			allErrs = append(allErrs, errs.NewFieldNotSupported("fieldPath", internalFieldPath))
		}
	}

	return allErrs
}

func validateVolumeMounts(mounts []api.VolumeMount, volumes util.StringSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for i, mnt := range mounts {
		mErrs := errs.ValidationErrorList{}
		if len(mnt.Name) == 0 {
			mErrs = append(mErrs, errs.NewFieldRequired("name"))
		} else if !volumes.Has(mnt.Name) {
			mErrs = append(mErrs, errs.NewFieldNotFound("name", mnt.Name))
		}
		if len(mnt.MountPath) == 0 {
			mErrs = append(mErrs, errs.NewFieldRequired("mountPath"))
		}
		allErrs = append(allErrs, mErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateProbe(probe *api.Probe) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if probe == nil {
		return allErrs
	}
	allErrs = append(allErrs, validateHandler(&probe.Handler)...)
	if probe.InitialDelaySeconds < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("initialDelay", probe.InitialDelaySeconds, "may not be less than zero"))
	}
	if probe.TimeoutSeconds < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("timeout", probe.TimeoutSeconds, "may not be less than zero"))
	}
	return allErrs
}

// AccumulateUniqueHostPorts extracts each HostPort of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniqueHostPorts(containers []api.Container, accumulator *util.StringSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for ci, ctr := range containers {
		cErrs := errs.ValidationErrorList{}
		for pi := range ctr.Ports {
			port := ctr.Ports[pi].HostPort
			if port == 0 {
				continue
			}
			str := fmt.Sprintf("%d/%s", port, ctr.Ports[pi].Protocol)
			if accumulator.Has(str) {
				cErrs = append(cErrs, errs.NewFieldDuplicate("port", str))
			} else {
				accumulator.Insert(str)
			}
		}
		allErrs = append(allErrs, cErrs.PrefixIndex(ci)...)
	}
	return allErrs
}

// checkHostPortConflicts checks for colliding Port.HostPort values across
// a slice of containers.
func checkHostPortConflicts(containers []api.Container) errs.ValidationErrorList {
	allPorts := util.StringSet{}
	return AccumulateUniqueHostPorts(containers, &allPorts)
}

func validateExecAction(exec *api.ExecAction) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	if len(exec.Command) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("command"))
	}
	return allErrors
}

func validateHTTPGetAction(http *api.HTTPGetAction) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	if len(http.Path) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("path"))
	}
	if http.Port.Kind == util.IntstrInt && !util.IsValidPortNum(http.Port.IntVal) {
		allErrors = append(allErrors, errs.NewFieldInvalid("port", http.Port, portRangeErrorMsg))
	} else if http.Port.Kind == util.IntstrString && len(http.Port.StrVal) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("port"))
	}
	return allErrors
}

func validateTCPSocketAction(tcp *api.TCPSocketAction) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	if tcp.Port.Kind == util.IntstrInt && !util.IsValidPortNum(tcp.Port.IntVal) {
		allErrors = append(allErrors, errs.NewFieldInvalid("port", tcp.Port, portRangeErrorMsg))
	} else if tcp.Port.Kind == util.IntstrString && len(tcp.Port.StrVal) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("port"))
	}
	return allErrors
}

func validateHandler(handler *api.Handler) errs.ValidationErrorList {
	numHandlers := 0
	allErrors := errs.ValidationErrorList{}
	if handler.Exec != nil {
		numHandlers++
		allErrors = append(allErrors, validateExecAction(handler.Exec).Prefix("exec")...)
	}
	if handler.HTTPGet != nil {
		numHandlers++
		allErrors = append(allErrors, validateHTTPGetAction(handler.HTTPGet).Prefix("httpGet")...)
	}
	if handler.TCPSocket != nil {
		numHandlers++
		allErrors = append(allErrors, validateTCPSocketAction(handler.TCPSocket).Prefix("tcpSocket")...)
	}
	if numHandlers != 1 {
		allErrors = append(allErrors, errs.NewFieldInvalid("", handler, "exactly 1 handler type is required"))
	}
	return allErrors
}

func validateLifecycle(lifecycle *api.Lifecycle) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if lifecycle.PostStart != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PostStart).Prefix("postStart")...)
	}
	if lifecycle.PreStop != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PreStop).Prefix("preStop")...)
	}
	return allErrs
}

func validatePullPolicy(ctr *api.Container) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}

	switch ctr.ImagePullPolicy {
	case api.PullAlways, api.PullIfNotPresent, api.PullNever:
		break
	case "":
		allErrors = append(allErrors, errs.NewFieldRequired(""))
	default:
		allErrors = append(allErrors, errs.NewFieldNotSupported("", ctr.ImagePullPolicy))
	}

	return allErrors
}

func validateContainers(containers []api.Container, volumes util.StringSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if len(containers) == 0 {
		return append(allErrs, errs.NewFieldRequired(""))
	}

	allNames := util.StringSet{}
	for i, ctr := range containers {
		cErrs := errs.ValidationErrorList{}
		if len(ctr.Name) == 0 {
			cErrs = append(cErrs, errs.NewFieldRequired("name"))
		} else if !util.IsDNS1123Label(ctr.Name) {
			cErrs = append(cErrs, errs.NewFieldInvalid("name", ctr.Name, dns1123LabelErrorMsg))
		} else if allNames.Has(ctr.Name) {
			cErrs = append(cErrs, errs.NewFieldDuplicate("name", ctr.Name))
		} else {
			allNames.Insert(ctr.Name)
		}
		if len(ctr.Image) == 0 {
			cErrs = append(cErrs, errs.NewFieldRequired("image"))
		}
		if ctr.Lifecycle != nil {
			cErrs = append(cErrs, validateLifecycle(ctr.Lifecycle).Prefix("lifecycle")...)
		}
		cErrs = append(cErrs, validateProbe(ctr.LivenessProbe).Prefix("livenessProbe")...)
		cErrs = append(cErrs, validateProbe(ctr.ReadinessProbe).Prefix("readinessProbe")...)
		cErrs = append(cErrs, validatePorts(ctr.Ports).Prefix("ports")...)
		cErrs = append(cErrs, validateEnv(ctr.Env).Prefix("env")...)
		cErrs = append(cErrs, validateVolumeMounts(ctr.VolumeMounts, volumes).Prefix("volumeMounts")...)
		cErrs = append(cErrs, validatePullPolicy(&ctr).Prefix("pullPolicy")...)
		cErrs = append(cErrs, ValidateResourceRequirements(&ctr.Resources).Prefix("resources")...)
		cErrs = append(cErrs, ValidateSecurityContext(ctr.SecurityContext).Prefix("securityContext")...)
		allErrs = append(allErrs, cErrs.PrefixIndex(i)...)
	}
	// Check for colliding ports across all containers.
	allErrs = append(allErrs, checkHostPortConflicts(containers)...)

	return allErrs
}

var supportedManifestVersions = util.NewStringSet("v1beta1", "v1beta2")

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
// TODO: replaced by ValidatePodSpec
func ValidateManifest(manifest *api.ContainerManifest) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if len(manifest.Version) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("version"))
	} else if !supportedManifestVersions.Has(strings.ToLower(manifest.Version)) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("version", manifest.Version))
	}
	allVolumes, vErrs := validateVolumes(manifest.Volumes)
	allErrs = append(allErrs, vErrs.Prefix("volumes")...)
	allErrs = append(allErrs, validateContainers(manifest.Containers, allVolumes).Prefix("containers")...)
	allErrs = append(allErrs, validateRestartPolicy(&manifest.RestartPolicy).Prefix("restartPolicy")...)
	allErrs = append(allErrs, validateDNSPolicy(&manifest.DNSPolicy).Prefix("dnsPolicy")...)
	return allErrs
}

func validateRestartPolicy(restartPolicy *api.RestartPolicy) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	switch *restartPolicy {
	case api.RestartPolicyAlways, api.RestartPolicyOnFailure, api.RestartPolicyNever:
		break
	case "":
		allErrors = append(allErrors, errs.NewFieldRequired(""))
	default:
		allErrors = append(allErrors, errs.NewFieldNotSupported("", restartPolicy))
	}

	return allErrors
}

func validateDNSPolicy(dnsPolicy *api.DNSPolicy) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	switch *dnsPolicy {
	case api.DNSClusterFirst, api.DNSDefault:
		break
	case "":
		allErrors = append(allErrors, errs.NewFieldRequired(""))
	default:
		allErrors = append(allErrors, errs.NewFieldNotSupported("", dnsPolicy))
	}
	return allErrors
}

func validateHostNetwork(hostNetwork bool, containers []api.Container) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	if hostNetwork {
		for _, container := range containers {
			for _, port := range container.Ports {
				if port.HostPort != port.ContainerPort {
					allErrors = append(allErrors, errs.NewFieldInvalid("containerPort", port.ContainerPort, "containerPort must match hostPort if hostNetwork is set to true"))
				}
			}
		}
	}
	return allErrors
}

// validateImagePullSecrets checks to make sure the pull secrets are well formed.  Right now, we only expect name to be set (it's the only field).  If this ever changes
// and someone decides to set those fields, we'd like to know.
func validateImagePullSecrets(imagePullSecrets []api.LocalObjectReference) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	for i, currPullSecret := range imagePullSecrets {
		strippedRef := api.LocalObjectReference{Name: currPullSecret.Name}

		if !reflect.DeepEqual(strippedRef, currPullSecret) {
			allErrors = append(allErrors, errs.NewFieldInvalid(fmt.Sprintf("[%d]", i), currPullSecret, "only name may be set"))
		}
	}
	return allErrors
}

// ValidatePod tests if required fields in the pod are set.
func ValidatePod(pod *api.Pod) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodSpec(&pod.Spec).Prefix("spec")...)

	return allErrs
}

// ValidatePodSpec tests that the specified PodSpec has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidatePodSpec(spec *api.PodSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allVolumes, vErrs := validateVolumes(spec.Volumes)
	allErrs = append(allErrs, vErrs.Prefix("volumes")...)
	allErrs = append(allErrs, validateContainers(spec.Containers, allVolumes).Prefix("containers")...)
	allErrs = append(allErrs, validateRestartPolicy(&spec.RestartPolicy).Prefix("restartPolicy")...)
	allErrs = append(allErrs, validateDNSPolicy(&spec.DNSPolicy).Prefix("dnsPolicy")...)
	allErrs = append(allErrs, ValidateLabels(spec.NodeSelector, "nodeSelector")...)
	allErrs = append(allErrs, validateHostNetwork(spec.HostNetwork, spec.Containers).Prefix("hostNetwork")...)
	allErrs = append(allErrs, validateImagePullSecrets(spec.ImagePullSecrets).Prefix("imagePullSecrets")...)

	if spec.ActiveDeadlineSeconds != nil {
		if *spec.ActiveDeadlineSeconds <= 0 {
			allErrs = append(allErrs, errs.NewFieldInvalid("activeDeadlineSeconds", spec.ActiveDeadlineSeconds, "activeDeadlineSeconds must be a positive integer greater than 0"))
		}
	}
	return allErrs
}

// ValidatePodUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodUpdate(newPod, oldPod *api.Pod) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPod.ObjectMeta, &newPod.ObjectMeta).Prefix("metadata")...)

	if len(newPod.Spec.Containers) != len(oldPod.Spec.Containers) {
		allErrs = append(allErrs, errs.NewFieldInvalid("spec.containers", newPod.Spec.Containers, "may not add or remove containers"))
		return allErrs
	}
	pod := *newPod
	// Tricky, we need to copy the container list so that we don't overwrite the update
	var newContainers []api.Container
	for ix, container := range pod.Spec.Containers {
		container.Image = oldPod.Spec.Containers[ix].Image
		newContainers = append(newContainers, container)
	}
	pod.Spec.Containers = newContainers
	if !api.Semantic.DeepEqual(pod.Spec, oldPod.Spec) {
		allErrs = append(allErrs, errs.NewFieldInvalid("spec", newPod.Spec, "may not update fields other than container.image"))
	}

	newPod.Status = oldPod.Status
	return allErrs
}

// ValidatePodStatusUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodStatusUpdate(newPod, oldPod *api.Pod) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPod.ObjectMeta, &newPod.ObjectMeta).Prefix("metadata")...)

	// TODO: allow change when bindings are properly decoupled from pods
	if newPod.Spec.Host != oldPod.Spec.Host {
		allErrs = append(allErrs, errs.NewFieldInvalid("status.host", newPod.Spec.Host, "pod host cannot be changed directly"))
	}

	// For status update we ignore changes to pod spec.
	newPod.Spec = oldPod.Spec

	return allErrs
}

// ValidatePodTemplate tests if required fields in the pod template are set.
func ValidatePodTemplate(pod *api.PodTemplate) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodTemplateSpec(&pod.Template, 0).Prefix("template")...)
	return allErrs
}

// ValidatePodTemplateUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodTemplateUpdate(newPod, oldPod *api.PodTemplate) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPod.ObjectMeta, &newPod.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodTemplateSpec(&newPod.Template, 0).Prefix("template")...)
	return allErrs
}

var supportedSessionAffinityType = util.NewStringSet(string(api.ServiceAffinityClientIP), string(api.ServiceAffinityNone))
var supportedServiceType = util.NewStringSet(string(api.ServiceTypeClusterIP), string(api.ServiceTypeNodePort),
	string(api.ServiceTypeLoadBalancer))

// ValidateService tests if required fields in the service are set.
func ValidateService(service *api.Service) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&service.ObjectMeta, true, ValidateServiceName).Prefix("metadata")...)

	if len(service.Spec.Ports) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("spec.ports"))
	}
	allPortNames := util.StringSet{}
	for i := range service.Spec.Ports {
		allErrs = append(allErrs, validateServicePort(&service.Spec.Ports[i], len(service.Spec.Ports) > 1, &allPortNames).PrefixIndex(i).Prefix("spec.ports")...)
	}

	if service.Spec.Selector != nil {
		allErrs = append(allErrs, ValidateLabels(service.Spec.Selector, "spec.selector")...)
	}

	if service.Spec.SessionAffinity == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("spec.sessionAffinity"))
	} else if !supportedSessionAffinityType.Has(string(service.Spec.SessionAffinity)) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("spec.sessionAffinity", service.Spec.SessionAffinity))
	}

	if api.IsServiceIPSet(service) {
		if ip := net.ParseIP(service.Spec.PortalIP); ip == nil {
			allErrs = append(allErrs, errs.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP, "portalIP should be empty, 'None', or a valid IP address"))
		}
	}

	for _, ip := range service.Spec.DeprecatedPublicIPs {
		if ip == "0.0.0.0" {
			allErrs = append(allErrs, errs.NewFieldInvalid("spec.publicIPs", ip, "is not an IP address"))
		} else if util.IsValidIPv4(ip) && net.ParseIP(ip).IsLoopback() {
			allErrs = append(allErrs, errs.NewFieldInvalid("spec.publicIPs", ip, "publicIP cannot be a loopback"))
		}
	}

	if service.Spec.Type == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("spec.type"))
	} else if !supportedServiceType.Has(string(service.Spec.Type)) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("spec.type", service.Spec.Type))
	}

	if service.Spec.Type == api.ServiceTypeLoadBalancer {
		for i := range service.Spec.Ports {
			if service.Spec.Ports[i].Protocol != api.ProtocolTCP {
				allErrs = append(allErrs, errs.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].protocol", i), service.Spec.Ports[i].Protocol, "cannot create an external load balancer with non-TCP ports"))
			}
		}
	}

	if service.Spec.Type == api.ServiceTypeClusterIP {
		for i := range service.Spec.Ports {
			if service.Spec.Ports[i].NodePort != 0 {
				allErrs = append(allErrs, errs.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].nodePort", i), service.Spec.Ports[i].NodePort, "cannot specify a node port with services of type ClusterIP"))
			}
		}
	}

	// Check for duplicate NodePorts, considering (protocol,port) pairs
	nodePorts := make(map[api.ServicePort]bool)
	for i := range service.Spec.Ports {
		port := &service.Spec.Ports[i]
		if port.NodePort == 0 {
			continue
		}
		var key api.ServicePort
		key.Protocol = port.Protocol
		key.NodePort = port.NodePort
		_, found := nodePorts[key]
		if found {
			allErrs = append(allErrs, errs.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].nodePort", i), port.NodePort, "duplicate nodePort specified"))
		}
		nodePorts[key] = true
	}

	return allErrs
}

func validateServicePort(sp *api.ServicePort, requireName bool, allNames *util.StringSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if requireName && sp.Name == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("name"))
	} else if sp.Name != "" {
		if !util.IsDNS1123Label(sp.Name) {
			allErrs = append(allErrs, errs.NewFieldInvalid("name", sp.Name, dns1123LabelErrorMsg))
		} else if allNames.Has(sp.Name) {
			allErrs = append(allErrs, errs.NewFieldDuplicate("name", sp.Name))
		} else {
			allNames.Insert(sp.Name)
		}
	}

	if !util.IsValidPortNum(sp.Port) {
		allErrs = append(allErrs, errs.NewFieldInvalid("port", sp.Port, portRangeErrorMsg))
	}

	if len(sp.Protocol) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("protocol"))
	} else if !supportedPortProtocols.Has(strings.ToUpper(string(sp.Protocol))) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("protocol", sp.Protocol))
	}

	if sp.TargetPort != util.NewIntOrStringFromInt(0) && sp.TargetPort != util.NewIntOrStringFromString("") {
		if sp.TargetPort.Kind == util.IntstrInt && !util.IsValidPortNum(sp.TargetPort.IntVal) {
			allErrs = append(allErrs, errs.NewFieldInvalid("targetPort", sp.TargetPort, portRangeErrorMsg))
		}
	}

	return allErrs
}

// ValidateServiceUpdate tests if required fields in the service are set during an update
func ValidateServiceUpdate(oldService, service *api.Service) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldService.ObjectMeta, &service.ObjectMeta).Prefix("metadata")...)

	// TODO: PortalIP should be a Status field, since the system can set a value != to the user's value
	// once PortalIP is set, it cannot be unset.
	if api.IsServiceIPSet(oldService) && service.Spec.PortalIP != oldService.Spec.PortalIP {
		allErrs = append(allErrs, errs.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP, "field is immutable"))
	}

	allErrs = append(allErrs, ValidateService(service)...)
	return allErrs
}

// ValidateReplicationController tests if required fields in the replication controller are set.
func ValidateReplicationController(controller *api.ReplicationController) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&controller.ObjectMeta, true, ValidateReplicationControllerName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec).Prefix("spec")...)

	return allErrs
}

// ValidateReplicationControllerUpdate tests if required fields in the replication controller are set.
func ValidateReplicationControllerUpdate(oldController, controller *api.ReplicationController) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldController.ObjectMeta, &controller.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateReplicationControllerSpec tests if required fields in the replication controller spec are set.
func ValidateReplicationControllerSpec(spec *api.ReplicationControllerSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	selector := labels.Set(spec.Selector).AsSelector()
	if selector.Empty() {
		allErrs = append(allErrs, errs.NewFieldRequired("selector"))
	}
	if spec.Replicas < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("replicas", spec.Replicas, isNegativeErrorMsg))
	}

	if spec.Template == nil {
		allErrs = append(allErrs, errs.NewFieldRequired("template"))
	} else {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, errs.NewFieldInvalid("template.labels", spec.Template.Labels, "selector does not match template"))
		}
		allErrs = append(allErrs, ValidatePodTemplateSpec(spec.Template, spec.Replicas).Prefix("template")...)
		// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
		if spec.Template.Spec.RestartPolicy != api.RestartPolicyAlways {
			allErrs = append(allErrs, errs.NewFieldNotSupported("template.restartPolicy", spec.Template.Spec.RestartPolicy))
		}
	}
	return allErrs
}

// ValidatePodTemplateSpec validates the spec of a pod template
func ValidatePodTemplateSpec(spec *api.PodTemplateSpec, replicas int) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateLabels(spec.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(spec.Annotations, "annotations")...)
	allErrs = append(allErrs, ValidatePodSpec(&spec.Spec).Prefix("spec")...)
	if replicas > 1 {
		allErrs = append(allErrs, ValidateReadOnlyPersistentDisks(spec.Spec.Volumes).Prefix("spec.volumes")...)
	}
	return allErrs
}

func ValidateReadOnlyPersistentDisks(volumes []api.Volume) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for _, vol := range volumes {
		if vol.GCEPersistentDisk != nil {
			if vol.GCEPersistentDisk.ReadOnly == false {
				allErrs = append(allErrs, errs.NewFieldInvalid("GCEPersistentDisk.ReadOnly", false, "ReadOnly must be true for replicated pods > 1, as GCE PD can only be mounted on multiple machines if it is read-only."))
			}
		}
		// TODO: What to do for AWS?  It doesn't support replicas
	}
	return allErrs
}

// ValidateNode tests if required fields in the node are set.
func ValidateNode(node *api.Node) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&node.ObjectMeta, false, ValidateNodeName).Prefix("metadata")...)

	// Only validate spec. All status fields are optional and can be updated later.

	// external ID is required.
	if len(node.Spec.ExternalID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("spec.ExternalID"))
	}

	// TODO(rjnagal): Ignore PodCIDR till its completely implemented.
	return allErrs
}

// ValidateNodeUpdate tests to make sure a node update can be applied.  Modifies oldNode.
func ValidateNodeUpdate(oldNode *api.Node, node *api.Node) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldNode.ObjectMeta, &node.ObjectMeta).Prefix("metadata")...)

	// TODO: Enable the code once we have better api object.status update model. Currently,
	// anyone can update node status.
	// if !api.Semantic.DeepEqual(node.Status, api.NodeStatus{}) {
	// 	allErrs = append(allErrs, errs.NewFieldInvalid("status", node.Status, "status must be empty"))
	// }

	// Validte no duplicate addresses in node status.
	addresses := make(map[api.NodeAddress]bool)
	for _, address := range node.Status.Addresses {
		if _, ok := addresses[address]; ok {
			allErrs = append(allErrs, fmt.Errorf("duplicate node addresses found"))
		}
		addresses[address] = true
	}

	// TODO: move reset function to its own location
	// Ignore metadata changes now that they have been tested
	oldNode.ObjectMeta = node.ObjectMeta
	// Allow users to update capacity
	oldNode.Status.Capacity = node.Status.Capacity
	// Allow the controller manager to assign a CIDR to a node.
	oldNode.Spec.PodCIDR = node.Spec.PodCIDR
	// Allow users to unschedule node
	oldNode.Spec.Unschedulable = node.Spec.Unschedulable
	// Clear status
	oldNode.Status = node.Status

	// TODO: Add a 'real' ValidationError type for this error and provide print actual diffs.
	if !api.Semantic.DeepEqual(oldNode, node) {
		glog.V(4).Infof("Update failed validation %#v vs %#v", oldNode, node)
		allErrs = append(allErrs, fmt.Errorf("update contains more than labels or capacity changes"))
	}

	return allErrs
}

// Validate compute resource typename.
// Refer to docs/resources.md for more details.
func validateResourceName(value string, field string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if !util.IsQualifiedName(value) {
		return append(allErrs, errs.NewFieldInvalid(field, value, "resource typename: "+qualifiedNameErrorMsg))
	}

	if len(strings.Split(value, "/")) == 1 {
		if !api.IsStandardResourceName(value) {
			return append(allErrs, errs.NewFieldInvalid(field, value, "is neither a standard resource type nor is fully qualified"))
		}
	}

	return errs.ValidationErrorList{}
}

// ValidateLimitRange tests if required fields in the LimitRange are set.
func ValidateLimitRange(limitRange *api.LimitRange) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&limitRange.ObjectMeta, true, ValidateLimitRangeName).Prefix("metadata")...)

	// ensure resource names are properly qualified per docs/resources.md
	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		for k := range limit.Max {
			allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].max[%s]", i, k))...)
		}
		for k := range limit.Min {
			allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].min[%s]", i, k))...)
		}
	}
	return allErrs
}

// ValidateServiceAccount tests if required fields in the ServiceAccount are set.
func ValidateServiceAccount(serviceAccount *api.ServiceAccount) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&serviceAccount.ObjectMeta, true, ValidateServiceAccountName).Prefix("metadata")...)
	return allErrs
}

// ValidateServiceAccountUpdate tests if required fields in the ServiceAccount are set.
func ValidateServiceAccountUpdate(oldServiceAccount, newServiceAccount *api.ServiceAccount) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldServiceAccount.ObjectMeta, &newServiceAccount.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateServiceAccount(newServiceAccount)...)
	return allErrs
}

const SecretKeyFmt string = "\\.?" + util.DNS1123LabelFmt + "(\\." + util.DNS1123LabelFmt + ")*"

var secretKeyRegexp = regexp.MustCompile("^" + SecretKeyFmt + "$")

// IsSecretKey tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123), except that a leading dot is allowed
func IsSecretKey(value string) bool {
	return len(value) <= util.DNS1123SubdomainMaxLength && secretKeyRegexp.MatchString(value)
}

// ValidateSecret tests if required fields in the Secret are set.
func ValidateSecret(secret *api.Secret) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&secret.ObjectMeta, true, ValidateSecretName).Prefix("metadata")...)

	totalSize := 0
	for key, value := range secret.Data {
		if !IsSecretKey(key) {
			allErrs = append(allErrs, errs.NewFieldInvalid(fmt.Sprintf("data[%s]", key), key, fmt.Sprintf("must have at most %d characters and match regex %s", util.DNS1123SubdomainMaxLength, SecretKeyFmt)))
		}

		totalSize += len(value)
	}

	if totalSize > api.MaxSecretSize {
		allErrs = append(allErrs, errs.NewFieldForbidden("data", "Maximum secret size exceeded"))
	}

	switch secret.Type {
	case api.SecretTypeServiceAccountToken:
		// Only require Annotations[kubernetes.io/service-account.name]
		// Additional fields (like Annotations[kubernetes.io/service-account.uid] and Data[token]) might be contributed later by a controller loop
		if value := secret.Annotations[api.ServiceAccountNameKey]; len(value) == 0 {
			allErrs = append(allErrs, errs.NewFieldRequired(fmt.Sprintf("metadata.annotations[%s]", api.ServiceAccountNameKey)))
		}
	case api.SecretTypeOpaque, "":
	// no-op
	case api.SecretTypeDockercfg:
		dockercfgBytes, exists := secret.Data[api.DockerConfigKey]
		if !exists {
			allErrs = append(allErrs, errs.NewFieldRequired(fmt.Sprintf("data[%s]", api.DockerConfigKey)))
			break
		}

		// make sure that the content is well-formed json.
		if err := json.Unmarshal(dockercfgBytes, &map[string]interface{}{}); err != nil {
			allErrs = append(allErrs, errs.NewFieldInvalid(fmt.Sprintf("data[%s]", api.DockerConfigKey), "<secret contents redacted>", err.Error()))
		}

	default:
		// no-op
	}

	return allErrs
}

// ValidateSecretUpdate tests if required fields in the Secret are set.
func ValidateSecretUpdate(oldSecret, newSecret *api.Secret) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldSecret.ObjectMeta, &newSecret.ObjectMeta).Prefix("metadata")...)

	if len(newSecret.Type) == 0 {
		newSecret.Type = oldSecret.Type
	}
	if newSecret.Type != oldSecret.Type {
		allErrs = append(allErrs, errs.NewFieldInvalid("type", newSecret.Type, "field is immutable"))
	}

	allErrs = append(allErrs, ValidateSecret(newSecret)...)
	return allErrs
}

func validateBasicResource(quantity resource.Quantity) errs.ValidationErrorList {
	if quantity.Value() < 0 {
		return errs.ValidationErrorList{errs.NewFieldInvalid("", quantity.Value(), "must be a valid resource quantity")}
	}
	return errs.ValidationErrorList{}
}

// Validates resource requirement spec.
func ValidateResourceRequirements(requirements *api.ResourceRequirements) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for resourceName, quantity := range requirements.Limits {
		// Validate resource name.
		errs := validateResourceName(resourceName.String(), fmt.Sprintf("resources.limits[%s]", resourceName))
		if api.IsStandardResourceName(resourceName.String()) {
			errs = append(errs, validateBasicResource(quantity).Prefix(fmt.Sprintf("Resource %s: ", resourceName))...)
		}
		allErrs = append(allErrs, errs...)
	}
	for resourceName, quantity := range requirements.Requests {
		// Validate resource name.
		errs := validateResourceName(resourceName.String(), fmt.Sprintf("resources.requests[%s]", resourceName))
		if api.IsStandardResourceName(resourceName.String()) {
			errs = append(errs, validateBasicResource(quantity).Prefix(fmt.Sprintf("Resource %s: ", resourceName))...)
		}
		allErrs = append(allErrs, errs...)
	}
	return allErrs
}

// ValidateResourceQuota tests if required fields in the ResourceQuota are set.
func ValidateResourceQuota(resourceQuota *api.ResourceQuota) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&resourceQuota.ObjectMeta, true, ValidateResourceQuotaName).Prefix("metadata")...)

	for k := range resourceQuota.Spec.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(resourceQuota.TypeMeta.Kind))...)
	}
	for k := range resourceQuota.Status.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(resourceQuota.TypeMeta.Kind))...)
	}
	for k := range resourceQuota.Status.Used {
		allErrs = append(allErrs, validateResourceName(string(k), string(resourceQuota.TypeMeta.Kind))...)
	}
	return allErrs
}

// ValidateResourceQuotaUpdate tests to see if the update is legal for an end user to make.
// newResourceQuota is updated with fields that cannot be changed.
func ValidateResourceQuotaUpdate(newResourceQuota, oldResourceQuota *api.ResourceQuota) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldResourceQuota.ObjectMeta, &newResourceQuota.ObjectMeta).Prefix("metadata")...)
	for k := range newResourceQuota.Spec.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(newResourceQuota.TypeMeta.Kind))...)
	}
	newResourceQuota.Status = oldResourceQuota.Status
	return allErrs
}

// ValidateResourceQuotaStatusUpdate tests to see if the status update is legal for an end user to make.
// newResourceQuota is updated with fields that cannot be changed.
func ValidateResourceQuotaStatusUpdate(newResourceQuota, oldResourceQuota *api.ResourceQuota) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldResourceQuota.ObjectMeta, &newResourceQuota.ObjectMeta).Prefix("metadata")...)
	if newResourceQuota.ResourceVersion == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("resourceVersion"))
	}
	for k := range newResourceQuota.Status.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(newResourceQuota.TypeMeta.Kind))...)
	}
	for k := range newResourceQuota.Status.Used {
		allErrs = append(allErrs, validateResourceName(string(k), string(newResourceQuota.TypeMeta.Kind))...)
	}
	newResourceQuota.Spec = oldResourceQuota.Spec
	return allErrs
}

// ValidateNamespace tests if required fields are set.
func ValidateNamespace(namespace *api.Namespace) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&namespace.ObjectMeta, false, ValidateNamespaceName).Prefix("metadata")...)
	for i := range namespace.Spec.Finalizers {
		allErrs = append(allErrs, validateFinalizerName(string(namespace.Spec.Finalizers[i]))...)
	}
	return allErrs
}

// Validate finalizer names
func validateFinalizerName(stringValue string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if !util.IsQualifiedName(stringValue) {
		return append(allErrs, errs.NewFieldInvalid("spec.finalizers", stringValue, qualifiedNameErrorMsg))
	}

	if len(strings.Split(stringValue, "/")) == 1 {
		if !api.IsStandardFinalizerName(stringValue) {
			return append(allErrs, errs.NewFieldInvalid("spec.finalizers", stringValue, fmt.Sprintf("finalizer name is neither a standard finalizer name nor is it fully qualified")))
		}
	}

	return errs.ValidationErrorList{}
}

// ValidateNamespaceUpdate tests to make sure a namespace update can be applied.
// newNamespace is updated with fields that cannot be changed
// TODO The syntax here is the reverse of the (old, new) pattern in most other validation.  Fix this.
func ValidateNamespaceUpdate(newNamespace *api.Namespace, oldNamespace *api.Namespace) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldNamespace.ObjectMeta, &newNamespace.ObjectMeta).Prefix("metadata")...)
	newNamespace.Spec.Finalizers = oldNamespace.Spec.Finalizers
	newNamespace.Status = oldNamespace.Status
	return allErrs
}

// ValidateNamespaceStatusUpdate tests to see if the update is legal for an end user to make. newNamespace is updated with fields
// that cannot be changed.
func ValidateNamespaceStatusUpdate(newNamespace, oldNamespace *api.Namespace) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldNamespace.ObjectMeta, &newNamespace.ObjectMeta).Prefix("metadata")...)
	newNamespace.Spec = oldNamespace.Spec
	if newNamespace.DeletionTimestamp.IsZero() {
		if newNamespace.Status.Phase != api.NamespaceActive {
			allErrs = append(allErrs, errs.NewFieldInvalid("Status.Phase", newNamespace.Status.Phase, "A namespace may only be in active status if it does not have a deletion timestamp."))
		}
	} else {
		if newNamespace.Status.Phase != api.NamespaceTerminating {
			allErrs = append(allErrs, errs.NewFieldInvalid("Status.Phase", newNamespace.Status.Phase, "A namespace may only be in terminating status if it has a deletion timestamp."))
		}
	}
	return allErrs
}

// ValidateNamespaceFinalizeUpdate tests to see if the update is legal for an end user to make.
// newNamespace is updated with fields that cannot be changed.
func ValidateNamespaceFinalizeUpdate(newNamespace, oldNamespace *api.Namespace) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldNamespace.ObjectMeta, &newNamespace.ObjectMeta).Prefix("metadata")...)
	for i := range newNamespace.Spec.Finalizers {
		allErrs = append(allErrs, validateFinalizerName(string(newNamespace.Spec.Finalizers[i]))...)
	}
	newNamespace.Status = oldNamespace.Status
	return allErrs
}

// ValidateEndpoints tests if required fields are set.
func ValidateEndpoints(endpoints *api.Endpoints) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&endpoints.ObjectMeta, true, ValidateEndpointsName).Prefix("metadata")...)
	allErrs = append(allErrs, validateEndpointSubsets(endpoints.Subsets).Prefix("subsets")...)
	return allErrs
}

func validateEndpointSubsets(subsets []api.EndpointSubset) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for i := range subsets {
		ss := &subsets[i]

		ssErrs := errs.ValidationErrorList{}

		if len(ss.Addresses) == 0 {
			ssErrs = append(ssErrs, errs.NewFieldRequired("addresses"))
		}
		if len(ss.Ports) == 0 {
			ssErrs = append(ssErrs, errs.NewFieldRequired("ports"))
		}
		for addr := range ss.Addresses {
			ssErrs = append(ssErrs, validateEndpointAddress(&ss.Addresses[addr]).PrefixIndex(addr).Prefix("addresses")...)
		}
		for port := range ss.Ports {
			ssErrs = append(ssErrs, validateEndpointPort(&ss.Ports[port], len(ss.Ports) > 1).PrefixIndex(port).Prefix("ports")...)
		}

		allErrs = append(allErrs, ssErrs.PrefixIndex(i)...)
	}

	return allErrs
}

func validateEndpointAddress(address *api.EndpointAddress) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if !util.IsValidIPv4(address.IP) {
		allErrs = append(allErrs, errs.NewFieldInvalid("ip", address.IP, "invalid IPv4 address"))
	}
	return allErrs
}

func validateEndpointPort(port *api.EndpointPort, requireName bool) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if requireName && port.Name == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("name"))
	} else if port.Name != "" {
		if !util.IsDNS1123Label(port.Name) {
			allErrs = append(allErrs, errs.NewFieldInvalid("name", port.Name, dns1123LabelErrorMsg))
		}
	}
	if !util.IsValidPortNum(port.Port) {
		allErrs = append(allErrs, errs.NewFieldInvalid("port", port.Port, portRangeErrorMsg))
	}
	if len(port.Protocol) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("protocol"))
	} else if !supportedPortProtocols.Has(strings.ToUpper(string(port.Protocol))) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("protocol", port.Protocol))
	}
	return allErrs
}

// ValidateEndpointsUpdate tests to make sure an endpoints update can be applied.
func ValidateEndpointsUpdate(oldEndpoints, newEndpoints *api.Endpoints) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldEndpoints.ObjectMeta, &newEndpoints.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, validateEndpointSubsets(newEndpoints.Subsets).Prefix("subsets")...)
	return allErrs
}

// ValidateSecurityContext ensure the security context contains valid settings
func ValidateSecurityContext(sc *api.SecurityContext) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	//this should only be true for testing since SecurityContext is defaulted by the api
	if sc == nil {
		return allErrs
	}

	if sc.Privileged != nil {
		if *sc.Privileged && !capabilities.Get().AllowPrivileged {
			allErrs = append(allErrs, errs.NewFieldForbidden("privileged", sc.Privileged))
		}
	}

	if sc.RunAsUser != nil {
		if *sc.RunAsUser < 0 {
			allErrs = append(allErrs, errs.NewFieldInvalid("runAsUser", *sc.RunAsUser, "runAsUser cannot be negative"))
		}
	}
	return allErrs
}
