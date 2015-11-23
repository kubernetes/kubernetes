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
	"os"
	"path"
	"reflect"
	"regexp"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation"

	"github.com/golang/glog"
)

// TODO: delete this global variable when we enable the validation of common
// fields by default.
var RepairMalformedUpdates bool = true

const isNegativeErrorMsg string = `must be non-negative`
const fieldImmutableErrorMsg string = `field is immutable`
const cIdentifierErrorMsg string = `must be a C identifier (matching regex ` + validation.CIdentifierFmt + `): e.g. "my_name" or "MyName"`
const isNotIntegerErrorMsg string = `must be an integer`

func IntervalErrorMsg(lo, hi int) string {
	return fmt.Sprintf(`must be greater than %d and less than %d`, lo, hi)
}

var labelValueErrorMsg string = fmt.Sprintf(`must have at most %d characters, matching regex %s: e.g. "MyValue" or ""`, validation.LabelValueMaxLength, validation.LabelValueFmt)
var qualifiedNameErrorMsg string = fmt.Sprintf(`must be a qualified name (at most %d characters, matching regex %s), with an optional DNS subdomain prefix (at most %d characters, matching regex %s) and slash (/): e.g. "MyName" or "example.com/MyName"`, validation.QualifiedNameMaxLength, validation.QualifiedNameFmt, validation.DNS1123SubdomainMaxLength, validation.DNS1123SubdomainFmt)
var DNSSubdomainErrorMsg string = fmt.Sprintf(`must be a DNS subdomain (at most %d characters, matching regex %s): e.g. "example.com"`, validation.DNS1123SubdomainMaxLength, validation.DNS1123SubdomainFmt)
var DNS1123LabelErrorMsg string = fmt.Sprintf(`must be a DNS label (at most %d characters, matching regex %s): e.g. "my-name"`, validation.DNS1123LabelMaxLength, validation.DNS1123LabelFmt)
var DNS952LabelErrorMsg string = fmt.Sprintf(`must be a DNS 952 label (at most %d characters, matching regex %s): e.g. "my-name"`, validation.DNS952LabelMaxLength, validation.DNS952LabelFmt)
var pdPartitionErrorMsg string = IntervalErrorMsg(0, 255)
var PortRangeErrorMsg string = IntervalErrorMsg(0, 65536)
var PortNameErrorMsg string = fmt.Sprintf(`must be an IANA_SVC_NAME (at most 15 characters, matching regex %s, it must contain at least one letter [a-z], and hyphens cannot be adjacent to other hyphens): e.g. "http"`, validation.IdentifierNoHyphensBeginEndFmt)

const totalAnnotationSizeLimitB int = 256 * (1 << 10) // 256 kB

func ValidateLabelName(labelName, fieldName string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if !validation.IsQualifiedName(labelName) {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, labelName, qualifiedNameErrorMsg))
	}
	return allErrs
}

// ValidateLabels validates that a set of labels are correctly defined.
func ValidateLabels(labels map[string]string, field string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	for k, v := range labels {
		allErrs = append(allErrs, ValidateLabelName(k, field)...)
		if !validation.IsValidLabelValue(v) {
			allErrs = append(allErrs, validation.NewFieldInvalid(field, v, labelValueErrorMsg))
		}
	}
	return allErrs
}

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, field string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	var totalSize int64
	for k, v := range annotations {
		if !validation.IsQualifiedName(strings.ToLower(k)) {
			allErrs = append(allErrs, validation.NewFieldInvalid(field, k, qualifiedNameErrorMsg))
		}
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}
	if totalSize > (int64)(totalAnnotationSizeLimitB) {
		allErrs = append(allErrs, validation.NewFieldTooLong(field, "", totalAnnotationSizeLimitB))
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
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateReplicationControllerName can be used to check whether the given replication
// controller name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateReplicationControllerName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateServiceName can be used to check whether the given service name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateServiceName(name string, prefix bool) (bool, string) {
	return NameIsDNS952Label(name, prefix)
}

// ValidateNodeName can be used to check whether the given node name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateNodeName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateNamespaceName can be used to check whether the given namespace name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateNamespaceName(name string, prefix bool) (bool, string) {
	return NameIsDNSLabel(name, prefix)
}

// ValidateLimitRangeName can be used to check whether the given limit range name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateLimitRangeName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateResourceQuotaName can be used to check whether the given
// resource quota name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateResourceQuotaName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateSecretName can be used to check whether the given secret name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateSecretName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateServiceAccountName can be used to check whether the given service account name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateServiceAccountName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// ValidateEndpointsName can be used to check whether the given endpoints name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateEndpointsName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

// NameIsDNSSubdomain is a ValidateNameFunc for names that must be a DNS subdomain.
func NameIsDNSSubdomain(name string, prefix bool) (bool, string) {
	if prefix {
		name = maskTrailingDash(name)
	}
	if validation.IsDNS1123Subdomain(name) {
		return true, ""
	}
	return false, DNSSubdomainErrorMsg
}

// NameIsDNSLabel is a ValidateNameFunc for names that must be a DNS 1123 label.
func NameIsDNSLabel(name string, prefix bool) (bool, string) {
	if prefix {
		name = maskTrailingDash(name)
	}
	if validation.IsDNS1123Label(name) {
		return true, ""
	}
	return false, DNS1123LabelErrorMsg
}

// NameIsDNS952Label is a ValidateNameFunc for names that must be a DNS 952 label.
func NameIsDNS952Label(name string, prefix bool) (bool, string) {
	if prefix {
		name = maskTrailingDash(name)
	}
	if validation.IsDNS952Label(name) {
		return true, ""
	}
	return false, DNS952LabelErrorMsg
}

// Validates that given value is not negative.
func ValidatePositiveField(value int64, fieldName string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if value < 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, value, isNegativeErrorMsg))
	}
	return allErrs
}

// Validates that a Quantity is not negative
func ValidatePositiveQuantity(value resource.Quantity, fieldName string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if value.Cmp(resource.Quantity{}) < 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, value.String(), isNegativeErrorMsg))
	}
	return allErrs
}

func ValidateImmutableField(new, old interface{}, fieldName string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if !api.Semantic.DeepEqual(old, new) {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, new, fieldImmutableErrorMsg))
	}
	return allErrs
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
// It doesn't return an error for rootscoped resources with namespace, because namespace should already be cleared before.
// TODO: Remove calls to this method scattered in validations of specific resources, e.g., ValidatePodUpdate.
func ValidateObjectMeta(meta *api.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if len(meta.GenerateName) != 0 {
		if ok, qualifier := nameFn(meta.GenerateName, true); !ok {
			allErrs = append(allErrs, validation.NewFieldInvalid("generateName", meta.GenerateName, qualifier))
		}
	}
	// If the generated name validates, but the calculated value does not, it's a problem with generation, and we
	// report it here. This may confuse users, but indicates a programming bug and still must be validated.
	// If there are multiple fields out of which one is required then add a or as a separator
	if len(meta.Name) == 0 {
		requiredErr := validation.NewFieldRequired("name")
		requiredErr.Detail = "name or generateName is required"
		allErrs = append(allErrs, requiredErr)
	} else {
		if ok, qualifier := nameFn(meta.Name, false); !ok {
			allErrs = append(allErrs, validation.NewFieldInvalid("name", meta.Name, qualifier))
		}
	}
	allErrs = append(allErrs, ValidatePositiveField(meta.Generation, "generation")...)
	if requiresNamespace {
		if len(meta.Namespace) == 0 {
			allErrs = append(allErrs, validation.NewFieldRequired("namespace"))
		} else if ok, _ := ValidateNamespaceName(meta.Namespace, false); !ok {
			allErrs = append(allErrs, validation.NewFieldInvalid("namespace", meta.Namespace, DNS1123LabelErrorMsg))
		}
	} else {
		if len(meta.Namespace) != 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("namespace", meta.Namespace, "namespace is not allowed on this type"))
		}
	}
	allErrs = append(allErrs, ValidateLabels(meta.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(meta.Annotations, "annotations")...)

	return allErrs
}

// ValidateObjectMetaUpdate validates an object's metadata when updated
func ValidateObjectMetaUpdate(new, old *api.ObjectMeta) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if !RepairMalformedUpdates && new.UID != old.UID {
		allErrs = append(allErrs, validation.NewFieldInvalid("uid", new.UID, "field is immutable"))
	}
	// in the event it is left empty, set it, to allow clients more flexibility
	// TODO: remove the following code that repairs the update request when we retire the clients that modify the immutable fields.
	// Please do not copy this pattern elsewhere; validation functions should not be modifying the objects they are passed!
	if RepairMalformedUpdates {
		if len(new.UID) == 0 {
			new.UID = old.UID
		}
		// ignore changes to timestamp
		if old.CreationTimestamp.IsZero() {
			old.CreationTimestamp = new.CreationTimestamp
		} else {
			new.CreationTimestamp = old.CreationTimestamp
		}
		// an object can never remove a deletion timestamp or clear/change grace period seconds
		if !old.DeletionTimestamp.IsZero() {
			new.DeletionTimestamp = old.DeletionTimestamp
		}
		if old.DeletionGracePeriodSeconds != nil && new.DeletionGracePeriodSeconds == nil {
			new.DeletionGracePeriodSeconds = old.DeletionGracePeriodSeconds
		}
	}

	// TODO: needs to check if new==nil && old !=nil after the repair logic is removed.
	if new.DeletionGracePeriodSeconds != nil && old.DeletionGracePeriodSeconds != nil && *new.DeletionGracePeriodSeconds != *old.DeletionGracePeriodSeconds {
		allErrs = append(allErrs, validation.NewFieldInvalid("deletionGracePeriodSeconds", new.DeletionGracePeriodSeconds, "field is immutable; may only be changed via deletion"))
	}

	// Reject updates that don't specify a resource version
	if new.ResourceVersion == "" {
		allErrs = append(allErrs, validation.NewFieldInvalid("resourceVersion", new.ResourceVersion, "resourceVersion must be specified for an update"))
	}

	allErrs = append(allErrs, ValidateImmutableField(new.Name, old.Name, "name")...)
	allErrs = append(allErrs, ValidateImmutableField(new.Namespace, old.Namespace, "namespace")...)
	allErrs = append(allErrs, ValidateImmutableField(new.UID, old.UID, "uid")...)
	allErrs = append(allErrs, ValidateImmutableField(new.CreationTimestamp, old.CreationTimestamp, "creationTimestamp")...)

	allErrs = append(allErrs, ValidateLabels(new.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(new.Annotations, "annotations")...)

	return allErrs
}

func validateVolumes(volumes []api.Volume) (sets.String, validation.ErrorList) {
	allErrs := validation.ErrorList{}

	allNames := sets.String{}
	for i, vol := range volumes {
		el := validateSource(&vol.VolumeSource).Prefix("source")
		if len(vol.Name) == 0 {
			el = append(el, validation.NewFieldRequired("name"))
		} else if !validation.IsDNS1123Label(vol.Name) {
			el = append(el, validation.NewFieldInvalid("name", vol.Name, DNS1123LabelErrorMsg))
		} else if allNames.Has(vol.Name) {
			el = append(el, validation.NewFieldDuplicate("name", vol.Name))
		}
		if len(el) == 0 {
			allNames.Insert(vol.Name)
		} else {
			allErrs = append(allErrs, el.PrefixIndex(i)...)
		}
	}
	return allNames, allErrs
}

func validateSource(source *api.VolumeSource) validation.ErrorList {
	numVolumes := 0
	allErrs := validation.ErrorList{}
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
	if source.Flocker != nil {
		numVolumes++
		allErrs = append(allErrs, validateFlocker(source.Flocker).Prefix("flocker")...)
	}
	if source.PersistentVolumeClaim != nil {
		numVolumes++
		allErrs = append(allErrs, validatePersistentClaimVolumeSource(source.PersistentVolumeClaim).Prefix("persistentVolumeClaim")...)
	}
	if source.RBD != nil {
		numVolumes++
		allErrs = append(allErrs, validateRBD(source.RBD).Prefix("rbd")...)
	}
	if source.Cinder != nil {
		numVolumes++
		allErrs = append(allErrs, validateCinderVolumeSource(source.Cinder).Prefix("cinder")...)
	}
	if source.CephFS != nil {
		numVolumes++
		allErrs = append(allErrs, validateCephFS(source.CephFS).Prefix("cephfs")...)
	}
	if source.DownwardAPI != nil {
		numVolumes++
		allErrs = append(allErrs, validateDownwardAPIVolumeSource(source.DownwardAPI).Prefix("downwardApi")...)
	}
	if source.FC != nil {
		numVolumes++
		allErrs = append(allErrs, validateFCVolumeSource(source.FC).Prefix("fc")...)
	}
	if numVolumes != 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("", source, "exactly 1 volume type is required"))
	}

	return allErrs
}

func validateHostPathVolumeSource(hostPath *api.HostPathVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if hostPath.Path == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("path"))
	}
	return allErrs
}

func validateGitRepoVolumeSource(gitRepo *api.GitRepoVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if gitRepo.Repository == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("repository"))
	}
	return allErrs
}

func validateISCSIVolumeSource(iscsi *api.ISCSIVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if iscsi.TargetPortal == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("targetPortal"))
	}
	if iscsi.IQN == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("iqn"))
	}
	if iscsi.FSType == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("fsType"))
	}
	if iscsi.Lun < 0 || iscsi.Lun > 255 {
		allErrs = append(allErrs, validation.NewFieldInvalid("lun", iscsi.Lun, ""))
	}
	return allErrs
}

func validateFCVolumeSource(fc *api.FCVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(fc.TargetWWNs) < 1 {
		allErrs = append(allErrs, validation.NewFieldRequired("targetWWNs"))
	}
	if fc.FSType == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("fsType"))
	}

	if fc.Lun == nil {
		allErrs = append(allErrs, validation.NewFieldRequired("lun"))
	} else {
		if *fc.Lun < 0 || *fc.Lun > 255 {
			allErrs = append(allErrs, validation.NewFieldInvalid("lun", fc.Lun, ""))
		}
	}
	return allErrs
}

func validateGCEPersistentDiskVolumeSource(PD *api.GCEPersistentDiskVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if PD.PDName == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("pdName"))
	}
	if PD.FSType == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("fsType"))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, validation.NewFieldInvalid("partition", PD.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateAWSElasticBlockStoreVolumeSource(PD *api.AWSElasticBlockStoreVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if PD.VolumeID == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("volumeID"))
	}
	if PD.FSType == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("fsType"))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, validation.NewFieldInvalid("partition", PD.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateSecretVolumeSource(secretSource *api.SecretVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if secretSource.SecretName == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("secretName"))
	}
	return allErrs
}

func validatePersistentClaimVolumeSource(claim *api.PersistentVolumeClaimVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if claim.ClaimName == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("claimName"))
	}
	return allErrs
}

func validateNFS(nfs *api.NFSVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if nfs.Server == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("server"))
	}
	if nfs.Path == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("path"))
	}
	if !path.IsAbs(nfs.Path) {
		allErrs = append(allErrs, validation.NewFieldInvalid("path", nfs.Path, "must be an absolute path"))
	}
	return allErrs
}

func validateGlusterfs(glusterfs *api.GlusterfsVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if glusterfs.EndpointsName == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("endpoints"))
	}
	if glusterfs.Path == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("path"))
	}
	return allErrs
}

func validateFlocker(flocker *api.FlockerVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if flocker.DatasetName == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("datasetName"))
	}
	if strings.Contains(flocker.DatasetName, "/") {
		allErrs = append(allErrs, validation.NewFieldInvalid("datasetName", flocker.DatasetName, "must not contain '/'"))
	}
	return allErrs
}

var validDownwardAPIFieldPathExpressions = sets.NewString("metadata.name", "metadata.namespace", "metadata.labels", "metadata.annotations")

func validateDownwardAPIVolumeSource(downwardAPIVolume *api.DownwardAPIVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	for _, downwardAPIVolumeFile := range downwardAPIVolume.Items {
		if len(downwardAPIVolumeFile.Path) == 0 {
			allErrs = append(allErrs, validation.NewFieldRequired("path"))
		}
		if path.IsAbs(downwardAPIVolumeFile.Path) {
			allErrs = append(allErrs, validation.NewFieldForbidden("path", "must not be an absolute path"))
		}
		items := strings.Split(downwardAPIVolumeFile.Path, string(os.PathSeparator))
		for _, item := range items {
			if item == ".." {
				allErrs = append(allErrs, validation.NewFieldInvalid("path", downwardAPIVolumeFile.Path, "must not contain \"..\"."))
			}
		}
		if strings.HasPrefix(items[0], "..") && len(items[0]) > 2 {
			allErrs = append(allErrs, validation.NewFieldInvalid("path", downwardAPIVolumeFile.Path, "must not start with \"..\"."))
		}
		allErrs = append(allErrs, validateObjectFieldSelector(&downwardAPIVolumeFile.FieldRef, &validDownwardAPIFieldPathExpressions).Prefix("FieldRef")...)
	}
	return allErrs
}

func validateRBD(rbd *api.RBDVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(rbd.CephMonitors) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("monitors"))
	}
	if rbd.RBDImage == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("image"))
	}
	if rbd.FSType == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("fsType"))
	}
	return allErrs
}

func validateCinderVolumeSource(cd *api.CinderVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if cd.VolumeID == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("volumeID"))
	}
	if cd.FSType == "" || (cd.FSType != "ext3" && cd.FSType != "ext4") {
		allErrs = append(allErrs, validation.NewFieldRequired("fsType required and should be of type ext3 or ext4"))
	}
	return allErrs
}

func validateCephFS(cephfs *api.CephFSVolumeSource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(cephfs.Monitors) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("monitors"))
	}
	return allErrs
}

func ValidatePersistentVolumeName(name string, prefix bool) (bool, string) {
	return NameIsDNSSubdomain(name, prefix)
}

func ValidatePersistentVolume(pv *api.PersistentVolume) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pv.ObjectMeta, false, ValidatePersistentVolumeName).Prefix("metadata")...)

	if len(pv.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("persistentVolume.AccessModes"))
	}

	for _, mode := range pv.Spec.AccessModes {
		if mode != api.ReadWriteOnce && mode != api.ReadOnlyMany && mode != api.ReadWriteMany {
			allErrs = append(allErrs, validation.NewFieldInvalid("persistentVolume.Spec.AccessModes", mode, fmt.Sprintf("only %s, %s, and %s are valid", api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany)))
		}
	}

	if len(pv.Spec.Capacity) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("persistentVolume.Capacity"))
	}

	if _, ok := pv.Spec.Capacity[api.ResourceStorage]; !ok || len(pv.Spec.Capacity) > 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("", pv.Spec.Capacity, fmt.Sprintf("only %s is expected", api.ResourceStorage)))
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
	if pv.Spec.Flocker != nil {
		numVolumes++
		allErrs = append(allErrs, validateFlocker(pv.Spec.Flocker).Prefix("flocker")...)
	}
	if pv.Spec.NFS != nil {
		numVolumes++
		allErrs = append(allErrs, validateNFS(pv.Spec.NFS).Prefix("nfs")...)
	}
	if pv.Spec.RBD != nil {
		numVolumes++
		allErrs = append(allErrs, validateRBD(pv.Spec.RBD).Prefix("rbd")...)
	}
	if pv.Spec.CephFS != nil {
		numVolumes++
		allErrs = append(allErrs, validateCephFS(pv.Spec.CephFS).Prefix("cephfs")...)
	}
	if pv.Spec.ISCSI != nil {
		numVolumes++
		allErrs = append(allErrs, validateISCSIVolumeSource(pv.Spec.ISCSI).Prefix("iscsi")...)
	}
	if pv.Spec.Cinder != nil {
		numVolumes++
		allErrs = append(allErrs, validateCinderVolumeSource(pv.Spec.Cinder).Prefix("cinder")...)
	}
	if pv.Spec.FC != nil {
		numVolumes++
		allErrs = append(allErrs, validateFCVolumeSource(pv.Spec.FC).Prefix("fc")...)
	}
	if numVolumes != 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("", pv.Spec.PersistentVolumeSource, "exactly 1 volume type is required"))
	}
	return allErrs
}

// ValidatePersistentVolumeUpdate tests to see if the update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeUpdate(newPv, oldPv *api.PersistentVolume) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = ValidatePersistentVolume(newPv)
	newPv.Status = oldPv.Status
	return allErrs
}

// ValidatePersistentVolumeStatusUpdate tests to see if the status update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeStatusUpdate(newPv, oldPv *api.PersistentVolume) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newPv.ObjectMeta, &oldPv.ObjectMeta).Prefix("metadata")...)
	if newPv.ResourceVersion == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("resourceVersion"))
	}
	newPv.Spec = oldPv.Spec
	return allErrs
}

func ValidatePersistentVolumeClaim(pvc *api.PersistentVolumeClaim) validation.ErrorList {
	allErrs := ValidateObjectMeta(&pvc.ObjectMeta, true, ValidatePersistentVolumeName)
	if len(pvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("persistentVolumeClaim.Spec.AccessModes", pvc.Spec.AccessModes, "at least 1 PersistentVolumeAccessMode is required"))
	}
	for _, mode := range pvc.Spec.AccessModes {
		if mode != api.ReadWriteOnce && mode != api.ReadOnlyMany && mode != api.ReadWriteMany {
			allErrs = append(allErrs, validation.NewFieldInvalid("persistentVolumeClaim.Spec.AccessModes", mode, fmt.Sprintf("only %s, %s, and %s are valid", api.ReadWriteOnce, api.ReadOnlyMany, api.ReadWriteMany)))
		}
	}
	if _, ok := pvc.Spec.Resources.Requests[api.ResourceStorage]; !ok {
		allErrs = append(allErrs, validation.NewFieldInvalid("persistentVolumeClaim.Spec.Resources.Requests", pvc.Spec.Resources.Requests, "No Storage size specified"))
	}
	return allErrs
}

func ValidatePersistentVolumeClaimUpdate(newPvc, oldPvc *api.PersistentVolumeClaim) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = ValidatePersistentVolumeClaim(newPvc)
	newPvc.Status = oldPvc.Status
	return allErrs
}

func ValidatePersistentVolumeClaimStatusUpdate(newPvc, oldPvc *api.PersistentVolumeClaim) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newPvc.ObjectMeta, &oldPvc.ObjectMeta).Prefix("metadata")...)
	if newPvc.ResourceVersion == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("resourceVersion"))
	}
	if len(newPvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("persistentVolume.AccessModes"))
	}
	for _, qty := range newPvc.Status.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty)...)
	}
	newPvc.Spec = oldPvc.Spec
	return allErrs
}

var supportedPortProtocols = sets.NewString(string(api.ProtocolTCP), string(api.ProtocolUDP))

func validatePorts(ports []api.ContainerPort) validation.ErrorList {
	allErrs := validation.ErrorList{}

	allNames := sets.String{}
	for i, port := range ports {
		pErrs := validation.ErrorList{}
		if len(port.Name) > 0 {
			if !validation.IsValidPortName(port.Name) {
				pErrs = append(pErrs, validation.NewFieldInvalid("name", port.Name, PortNameErrorMsg))
			} else if allNames.Has(port.Name) {
				pErrs = append(pErrs, validation.NewFieldDuplicate("name", port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if port.ContainerPort == 0 {
			pErrs = append(pErrs, validation.NewFieldInvalid("containerPort", port.ContainerPort, PortRangeErrorMsg))
		} else if !validation.IsValidPortNum(port.ContainerPort) {
			pErrs = append(pErrs, validation.NewFieldInvalid("containerPort", port.ContainerPort, PortRangeErrorMsg))
		}
		if port.HostPort != 0 && !validation.IsValidPortNum(port.HostPort) {
			pErrs = append(pErrs, validation.NewFieldInvalid("hostPort", port.HostPort, PortRangeErrorMsg))
		}
		if len(port.Protocol) == 0 {
			pErrs = append(pErrs, validation.NewFieldRequired("protocol"))
		} else if !supportedPortProtocols.Has(string(port.Protocol)) {
			pErrs = append(pErrs, validation.NewFieldNotSupported("protocol", port.Protocol, supportedPortProtocols.List()))
		}
		allErrs = append(allErrs, pErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateEnv(vars []api.EnvVar) validation.ErrorList {
	allErrs := validation.ErrorList{}

	for i, ev := range vars {
		vErrs := validation.ErrorList{}
		if len(ev.Name) == 0 {
			vErrs = append(vErrs, validation.NewFieldRequired("name"))
		} else if !validation.IsCIdentifier(ev.Name) {
			vErrs = append(vErrs, validation.NewFieldInvalid("name", ev.Name, cIdentifierErrorMsg))
		}
		vErrs = append(vErrs, validateEnvVarValueFrom(ev).Prefix("valueFrom")...)
		allErrs = append(allErrs, vErrs.PrefixIndex(i)...)
	}
	return allErrs
}

var validFieldPathExpressionsEnv = sets.NewString("metadata.name", "metadata.namespace", "status.podIP")

func validateEnvVarValueFrom(ev api.EnvVar) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if ev.ValueFrom == nil {
		return allErrs
	}

	numSources := 0

	switch {
	case ev.ValueFrom.FieldRef != nil:
		numSources++
		allErrs = append(allErrs, validateObjectFieldSelector(ev.ValueFrom.FieldRef, &validFieldPathExpressionsEnv).Prefix("fieldRef")...)
	}

	if ev.Value != "" && numSources != 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("", "", "sources cannot be specified when value is not empty"))
	}

	return allErrs
}

func validateObjectFieldSelector(fs *api.ObjectFieldSelector, expressions *sets.String) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if fs.APIVersion == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("apiVersion"))
	} else if fs.FieldPath == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("fieldPath"))
	} else {
		internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(fs.APIVersion, "Pod", fs.FieldPath, "")
		if err != nil {
			allErrs = append(allErrs, validation.NewFieldInvalid("fieldPath", fs.FieldPath, "error converting fieldPath"))
		} else if !expressions.Has(internalFieldPath) {
			allErrs = append(allErrs, validation.NewFieldNotSupported("fieldPath", internalFieldPath, expressions.List()))
		}
	}

	return allErrs
}

func validateVolumeMounts(mounts []api.VolumeMount, volumes sets.String) validation.ErrorList {
	allErrs := validation.ErrorList{}

	for i, mnt := range mounts {
		mErrs := validation.ErrorList{}
		if len(mnt.Name) == 0 {
			mErrs = append(mErrs, validation.NewFieldRequired("name"))
		} else if !volumes.Has(mnt.Name) {
			mErrs = append(mErrs, validation.NewFieldNotFound("name", mnt.Name))
		}
		if len(mnt.MountPath) == 0 {
			mErrs = append(mErrs, validation.NewFieldRequired("mountPath"))
		}
		allErrs = append(allErrs, mErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateProbe(probe *api.Probe) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if probe == nil {
		return allErrs
	}
	allErrs = append(allErrs, validateHandler(&probe.Handler)...)
	allErrs = append(allErrs, ValidatePositiveField(probe.InitialDelaySeconds, "initialDelaySeconds")...)
	allErrs = append(allErrs, ValidatePositiveField(probe.TimeoutSeconds, "timeoutSeconds")...)
	allErrs = append(allErrs, ValidatePositiveField(int64(probe.PeriodSeconds), "periodSeconds")...)
	allErrs = append(allErrs, ValidatePositiveField(int64(probe.SuccessThreshold), "successThreshold")...)
	allErrs = append(allErrs, ValidatePositiveField(int64(probe.FailureThreshold), "failureThreshold")...)
	return allErrs
}

// AccumulateUniqueHostPorts extracts each HostPort of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniqueHostPorts(containers []api.Container, accumulator *sets.String) validation.ErrorList {
	allErrs := validation.ErrorList{}

	for ci, ctr := range containers {
		cErrs := validation.ErrorList{}
		for pi := range ctr.Ports {
			port := ctr.Ports[pi].HostPort
			if port == 0 {
				continue
			}
			str := fmt.Sprintf("%d/%s", port, ctr.Ports[pi].Protocol)
			if accumulator.Has(str) {
				cErrs = append(cErrs, validation.NewFieldDuplicate("port", str))
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
func checkHostPortConflicts(containers []api.Container) validation.ErrorList {
	allPorts := sets.String{}
	return AccumulateUniqueHostPorts(containers, &allPorts)
}

func validateExecAction(exec *api.ExecAction) validation.ErrorList {
	allErrors := validation.ErrorList{}
	if len(exec.Command) == 0 {
		allErrors = append(allErrors, validation.NewFieldRequired("command"))
	}
	return allErrors
}

func validateHTTPGetAction(http *api.HTTPGetAction) validation.ErrorList {
	allErrors := validation.ErrorList{}
	if len(http.Path) == 0 {
		allErrors = append(allErrors, validation.NewFieldRequired("path"))
	}
	if http.Port.Type == intstr.Int && !validation.IsValidPortNum(http.Port.IntVal) {
		allErrors = append(allErrors, validation.NewFieldInvalid("port", http.Port, PortRangeErrorMsg))
	} else if http.Port.Type == intstr.String && !validation.IsValidPortName(http.Port.StrVal) {
		allErrors = append(allErrors, validation.NewFieldInvalid("port", http.Port.StrVal, PortNameErrorMsg))
	}
	supportedSchemes := sets.NewString(string(api.URISchemeHTTP), string(api.URISchemeHTTPS))
	if !supportedSchemes.Has(string(http.Scheme)) {
		allErrors = append(allErrors, validation.NewFieldInvalid("scheme", http.Scheme, fmt.Sprintf("must be one of %v", supportedSchemes.List())))
	}
	return allErrors
}

func validateTCPSocketAction(tcp *api.TCPSocketAction) validation.ErrorList {
	allErrors := validation.ErrorList{}
	if tcp.Port.Type == intstr.Int && !validation.IsValidPortNum(tcp.Port.IntVal) {
		allErrors = append(allErrors, validation.NewFieldInvalid("port", tcp.Port, PortRangeErrorMsg))
	} else if tcp.Port.Type == intstr.String && !validation.IsValidPortName(tcp.Port.StrVal) {
		allErrors = append(allErrors, validation.NewFieldInvalid("port", tcp.Port.StrVal, PortNameErrorMsg))
	}
	return allErrors
}

func validateHandler(handler *api.Handler) validation.ErrorList {
	numHandlers := 0
	allErrors := validation.ErrorList{}
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
		allErrors = append(allErrors, validation.NewFieldInvalid("", handler, "exactly 1 handler type is required"))
	}
	return allErrors
}

func validateLifecycle(lifecycle *api.Lifecycle) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if lifecycle.PostStart != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PostStart).Prefix("postStart")...)
	}
	if lifecycle.PreStop != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PreStop).Prefix("preStop")...)
	}
	return allErrs
}

func validatePullPolicy(ctr *api.Container) validation.ErrorList {
	allErrors := validation.ErrorList{}

	switch ctr.ImagePullPolicy {
	case api.PullAlways, api.PullIfNotPresent, api.PullNever:
		break
	case "":
		allErrors = append(allErrors, validation.NewFieldRequired(""))
	default:
		validValues := []string{string(api.PullAlways), string(api.PullIfNotPresent), string(api.PullNever)}
		allErrors = append(allErrors, validation.NewFieldNotSupported("", ctr.ImagePullPolicy, validValues))
	}

	return allErrors
}

func validateContainers(containers []api.Container, volumes sets.String) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if len(containers) == 0 {
		return append(allErrs, validation.NewFieldRequired(""))
	}

	allNames := sets.String{}
	for i, ctr := range containers {
		cErrs := validation.ErrorList{}
		if len(ctr.Name) == 0 {
			cErrs = append(cErrs, validation.NewFieldRequired("name"))
		} else if !validation.IsDNS1123Label(ctr.Name) {
			cErrs = append(cErrs, validation.NewFieldInvalid("name", ctr.Name, DNS1123LabelErrorMsg))
		} else if allNames.Has(ctr.Name) {
			cErrs = append(cErrs, validation.NewFieldDuplicate("name", ctr.Name))
		} else {
			allNames.Insert(ctr.Name)
		}
		if len(ctr.Image) == 0 {
			cErrs = append(cErrs, validation.NewFieldRequired("image"))
		}
		if ctr.Lifecycle != nil {
			cErrs = append(cErrs, validateLifecycle(ctr.Lifecycle).Prefix("lifecycle")...)
		}
		cErrs = append(cErrs, validateProbe(ctr.LivenessProbe).Prefix("livenessProbe")...)
		// Liveness-specific validation
		if ctr.LivenessProbe != nil && ctr.LivenessProbe.SuccessThreshold != 1 {
			allErrs = append(allErrs, validation.NewFieldForbidden("livenessProbe.successThreshold", "must be 1"))
		}

		cErrs = append(cErrs, validateProbe(ctr.ReadinessProbe).Prefix("readinessProbe")...)
		cErrs = append(cErrs, validatePorts(ctr.Ports).Prefix("ports")...)
		cErrs = append(cErrs, validateEnv(ctr.Env).Prefix("env")...)
		cErrs = append(cErrs, validateVolumeMounts(ctr.VolumeMounts, volumes).Prefix("volumeMounts")...)
		cErrs = append(cErrs, validatePullPolicy(&ctr).Prefix("imagePullPolicy")...)
		cErrs = append(cErrs, ValidateResourceRequirements(&ctr.Resources).Prefix("resources")...)
		cErrs = append(cErrs, ValidateSecurityContext(ctr.SecurityContext).Prefix("securityContext")...)
		allErrs = append(allErrs, cErrs.PrefixIndex(i)...)
	}
	// Check for colliding ports across all containers.
	allErrs = append(allErrs, checkHostPortConflicts(containers)...)

	return allErrs
}

func validateRestartPolicy(restartPolicy *api.RestartPolicy) validation.ErrorList {
	allErrors := validation.ErrorList{}
	switch *restartPolicy {
	case api.RestartPolicyAlways, api.RestartPolicyOnFailure, api.RestartPolicyNever:
		break
	case "":
		allErrors = append(allErrors, validation.NewFieldRequired(""))
	default:
		validValues := []string{string(api.RestartPolicyAlways), string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}
		allErrors = append(allErrors, validation.NewFieldNotSupported("", *restartPolicy, validValues))
	}

	return allErrors
}

func validateDNSPolicy(dnsPolicy *api.DNSPolicy) validation.ErrorList {
	allErrors := validation.ErrorList{}
	switch *dnsPolicy {
	case api.DNSClusterFirst, api.DNSDefault:
		break
	case "":
		allErrors = append(allErrors, validation.NewFieldRequired(""))
	default:
		validValues := []string{string(api.DNSClusterFirst), string(api.DNSDefault)}
		allErrors = append(allErrors, validation.NewFieldNotSupported("", dnsPolicy, validValues))
	}
	return allErrors
}

func validateHostNetwork(hostNetwork bool, containers []api.Container) validation.ErrorList {
	allErrors := validation.ErrorList{}
	if hostNetwork {
		for _, container := range containers {
			for _, port := range container.Ports {
				if port.HostPort != port.ContainerPort {
					allErrors = append(allErrors, validation.NewFieldInvalid("containerPort", port.ContainerPort, "containerPort must match hostPort if hostNetwork is set to true"))
				}
			}
		}
	}
	return allErrors
}

// validateImagePullSecrets checks to make sure the pull secrets are well formed.  Right now, we only expect name to be set (it's the only field).  If this ever changes
// and someone decides to set those fields, we'd like to know.
func validateImagePullSecrets(imagePullSecrets []api.LocalObjectReference) validation.ErrorList {
	allErrors := validation.ErrorList{}
	for i, currPullSecret := range imagePullSecrets {
		strippedRef := api.LocalObjectReference{Name: currPullSecret.Name}

		if !reflect.DeepEqual(strippedRef, currPullSecret) {
			allErrors = append(allErrors, validation.NewFieldInvalid(fmt.Sprintf("[%d]", i), currPullSecret, "only name may be set"))
		}
	}
	return allErrors
}

// ValidatePod tests if required fields in the pod are set.
func ValidatePod(pod *api.Pod) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodSpec(&pod.Spec).Prefix("spec")...)

	return allErrs
}

// ValidatePodSpec tests that the specified PodSpec has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidatePodSpec(spec *api.PodSpec) validation.ErrorList {
	allErrs := validation.ErrorList{}

	allVolumes, vErrs := validateVolumes(spec.Volumes)
	allErrs = append(allErrs, vErrs.Prefix("volumes")...)
	allErrs = append(allErrs, validateContainers(spec.Containers, allVolumes).Prefix("containers")...)
	allErrs = append(allErrs, validateRestartPolicy(&spec.RestartPolicy).Prefix("restartPolicy")...)
	allErrs = append(allErrs, validateDNSPolicy(&spec.DNSPolicy).Prefix("dnsPolicy")...)
	allErrs = append(allErrs, ValidateLabels(spec.NodeSelector, "nodeSelector")...)
	allErrs = append(allErrs, ValidatePodSecurityContext(spec.SecurityContext, spec).Prefix("securityContext")...)
	allErrs = append(allErrs, validateImagePullSecrets(spec.ImagePullSecrets).Prefix("imagePullSecrets")...)
	if len(spec.ServiceAccountName) > 0 {
		if ok, msg := ValidateServiceAccountName(spec.ServiceAccountName, false); !ok {
			allErrs = append(allErrs, validation.NewFieldInvalid("serviceAccountName", spec.ServiceAccountName, msg))
		}
	}

	if spec.ActiveDeadlineSeconds != nil {
		if *spec.ActiveDeadlineSeconds <= 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("activeDeadlineSeconds", spec.ActiveDeadlineSeconds, "activeDeadlineSeconds must be a positive integer greater than 0"))
		}
	}
	return allErrs
}

// ValidatePodSecurityContext test that the specified PodSecurityContext has valid data.
func ValidatePodSecurityContext(securityContext *api.PodSecurityContext, spec *api.PodSpec) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if securityContext != nil {
		allErrs = append(allErrs, validateHostNetwork(securityContext.HostNetwork, spec.Containers).Prefix("hostNetwork")...)
	}

	return allErrs
}

// ValidatePodUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodUpdate(newPod, oldPod *api.Pod) validation.ErrorList {
	allErrs := validation.ErrorList{}

	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newPod.ObjectMeta, &oldPod.ObjectMeta).Prefix("metadata")...)

	if len(newPod.Spec.Containers) != len(oldPod.Spec.Containers) {
		//TODO: Pinpoint the specific container that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, validation.NewFieldInvalid("spec.containers", "content of spec.containers is not printed out, please refer to the \"details\"", "may not add or remove containers"))
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
		//TODO: Pinpoint the specific field that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, validation.NewFieldInvalid("spec", "content of spec is not printed out, please refer to the \"details\"", "may not update fields other than container.image"))
	}

	newPod.Status = oldPod.Status
	return allErrs
}

// ValidatePodStatusUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodStatusUpdate(newPod, oldPod *api.Pod) validation.ErrorList {
	allErrs := validation.ErrorList{}

	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newPod.ObjectMeta, &oldPod.ObjectMeta).Prefix("metadata")...)

	// TODO: allow change when bindings are properly decoupled from pods
	if newPod.Spec.NodeName != oldPod.Spec.NodeName {
		allErrs = append(allErrs, validation.NewFieldInvalid("status.nodeName", newPod.Spec.NodeName, "pod nodename cannot be changed directly"))
	}

	// For status update we ignore changes to pod spec.
	newPod.Spec = oldPod.Spec

	return allErrs
}

// ValidatePodTemplate tests if required fields in the pod template are set.
func ValidatePodTemplate(pod *api.PodTemplate) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodTemplateSpec(&pod.Template).Prefix("template")...)
	return allErrs
}

// ValidatePodTemplateUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodTemplateUpdate(newPod, oldPod *api.PodTemplate) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPod.ObjectMeta, &newPod.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodTemplateSpec(&newPod.Template).Prefix("template")...)
	return allErrs
}

var supportedSessionAffinityType = sets.NewString(string(api.ServiceAffinityClientIP), string(api.ServiceAffinityNone))
var supportedServiceType = sets.NewString(string(api.ServiceTypeClusterIP), string(api.ServiceTypeNodePort),
	string(api.ServiceTypeLoadBalancer))

// ValidateService tests if required fields in the service are set.
func ValidateService(service *api.Service) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&service.ObjectMeta, true, ValidateServiceName).Prefix("metadata")...)

	if len(service.Spec.Ports) == 0 && service.Spec.ClusterIP != api.ClusterIPNone {
		allErrs = append(allErrs, validation.NewFieldRequired("spec.ports"))
	}
	if service.Spec.Type == api.ServiceTypeLoadBalancer {
		for ix := range service.Spec.Ports {
			port := &service.Spec.Ports[ix]
			if port.Port == 10250 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].port", ix), port.Port, "can not expose port 10250 externally since it is used by kubelet"))
			}
		}
	}
	allPortNames := sets.String{}
	for i := range service.Spec.Ports {
		allErrs = append(allErrs, validateServicePort(&service.Spec.Ports[i], len(service.Spec.Ports) > 1, &allPortNames).PrefixIndex(i).Prefix("spec.ports")...)
	}

	if service.Spec.Selector != nil {
		allErrs = append(allErrs, ValidateLabels(service.Spec.Selector, "spec.selector")...)
	}

	if service.Spec.SessionAffinity == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("spec.sessionAffinity"))
	} else if !supportedSessionAffinityType.Has(string(service.Spec.SessionAffinity)) {
		allErrs = append(allErrs, validation.NewFieldNotSupported("spec.sessionAffinity", service.Spec.SessionAffinity, supportedSessionAffinityType.List()))
	}

	if api.IsServiceIPSet(service) {
		if ip := net.ParseIP(service.Spec.ClusterIP); ip == nil {
			allErrs = append(allErrs, validation.NewFieldInvalid("spec.clusterIP", service.Spec.ClusterIP, "clusterIP should be empty, 'None', or a valid IP address"))
		}
	}

	for _, ip := range service.Spec.ExternalIPs {
		if ip == "0.0.0.0" {
			allErrs = append(allErrs, validation.NewFieldInvalid("spec.externalIPs", ip, "is not an IP address"))
		}
		allErrs = append(allErrs, validateIpIsNotLinkLocalOrLoopback(ip, "spec.externalIPs")...)
	}

	if service.Spec.Type == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("spec.type"))
	} else if !supportedServiceType.Has(string(service.Spec.Type)) {
		allErrs = append(allErrs, validation.NewFieldNotSupported("spec.type", service.Spec.Type, supportedServiceType.List()))
	}

	if service.Spec.Type == api.ServiceTypeLoadBalancer {
		for i := range service.Spec.Ports {
			if service.Spec.Ports[i].Protocol != api.ProtocolTCP {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].protocol", i), service.Spec.Ports[i].Protocol, "cannot create an external load balancer with non-TCP ports"))
			}
		}
	}

	if service.Spec.Type == api.ServiceTypeClusterIP {
		for i := range service.Spec.Ports {
			if service.Spec.Ports[i].NodePort != 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].nodePort", i), service.Spec.Ports[i].NodePort, "cannot specify a node port with services of type ClusterIP"))
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
			allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.ports[%d].nodePort", i), port.NodePort, "duplicate nodePort specified"))
		}
		nodePorts[key] = true
	}

	return allErrs
}

func validateServicePort(sp *api.ServicePort, requireName bool, allNames *sets.String) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if requireName && sp.Name == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("name"))
	} else if sp.Name != "" {
		if !validation.IsDNS1123Label(sp.Name) {
			allErrs = append(allErrs, validation.NewFieldInvalid("name", sp.Name, DNS1123LabelErrorMsg))
		} else if allNames.Has(sp.Name) {
			allErrs = append(allErrs, validation.NewFieldDuplicate("name", sp.Name))
		} else {
			allNames.Insert(sp.Name)
		}
	}

	if !validation.IsValidPortNum(sp.Port) {
		allErrs = append(allErrs, validation.NewFieldInvalid("port", sp.Port, PortRangeErrorMsg))
	}

	if len(sp.Protocol) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("protocol"))
	} else if !supportedPortProtocols.Has(string(sp.Protocol)) {
		allErrs = append(allErrs, validation.NewFieldNotSupported("protocol", sp.Protocol, supportedPortProtocols.List()))
	}

	if sp.TargetPort.Type == intstr.Int && !validation.IsValidPortNum(sp.TargetPort.IntVal) {
		allErrs = append(allErrs, validation.NewFieldInvalid("targetPort", sp.TargetPort, PortRangeErrorMsg))
	}
	if sp.TargetPort.Type == intstr.String && !validation.IsValidPortName(sp.TargetPort.StrVal) {
		allErrs = append(allErrs, validation.NewFieldInvalid("targetPort", sp.TargetPort, PortNameErrorMsg))
	}

	return allErrs
}

// ValidateServiceUpdate tests if required fields in the service are set during an update
func ValidateServiceUpdate(service, oldService *api.Service) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&service.ObjectMeta, &oldService.ObjectMeta).Prefix("metadata")...)

	if api.IsServiceIPSet(oldService) {
		allErrs = append(allErrs, ValidateImmutableField(service.Spec.ClusterIP, oldService.Spec.ClusterIP, "spec.clusterIP")...)
	}

	allErrs = append(allErrs, ValidateService(service)...)
	return allErrs
}

// ValidateReplicationController tests if required fields in the replication controller are set.
func ValidateReplicationController(controller *api.ReplicationController) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&controller.ObjectMeta, true, ValidateReplicationControllerName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateReplicationControllerUpdate tests if required fields in the replication controller are set.
func ValidateReplicationControllerUpdate(controller, oldController *api.ReplicationController) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateReplicationControllerStatusUpdate tests if required fields in the replication controller are set.
func ValidateReplicationControllerStatusUpdate(controller, oldController *api.ReplicationController) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePositiveField(int64(controller.Status.Replicas), "status.replicas")...)
	allErrs = append(allErrs, ValidatePositiveField(int64(controller.Status.ObservedGeneration), "status.observedGeneration")...)
	return allErrs
}

// Validates that the given selector is non-empty.
func ValidateNonEmptySelector(selectorMap map[string]string, fieldName string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	selector := labels.Set(selectorMap).AsSelector()
	if selector.Empty() {
		allErrs = append(allErrs, validation.NewFieldRequired(fieldName))
	}
	return allErrs
}

// Validates the given template and ensures that it is in accordance with the desrired selector and replicas.
func ValidatePodTemplateSpecForRC(template *api.PodTemplateSpec, selectorMap map[string]string, replicas int, fieldName string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if template == nil {
		allErrs = append(allErrs, validation.NewFieldRequired(fieldName))
	} else {
		selector := labels.Set(selectorMap).AsSelector()
		if !selector.Empty() {
			// Verify that the RC selector matches the labels in template.
			labels := labels.Set(template.Labels)
			if !selector.Matches(labels) {
				allErrs = append(allErrs, validation.NewFieldInvalid(fieldName+".metadata.labels", template.Labels, "selector does not match labels in "+fieldName))
			}
		}
		allErrs = append(allErrs, ValidatePodTemplateSpec(template).Prefix(fieldName)...)
		if replicas > 1 {
			allErrs = append(allErrs, ValidateReadOnlyPersistentDisks(template.Spec.Volumes).Prefix(fieldName+".spec.volumes")...)
		}
		// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
		if template.Spec.RestartPolicy != api.RestartPolicyAlways {
			allErrs = append(allErrs, validation.NewFieldNotSupported(fieldName+".spec.restartPolicy", template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
		}
	}
	return allErrs
}

// ValidateReplicationControllerSpec tests if required fields in the replication controller spec are set.
func ValidateReplicationControllerSpec(spec *api.ReplicationControllerSpec) validation.ErrorList {
	allErrs := validation.ErrorList{}

	allErrs = append(allErrs, ValidateNonEmptySelector(spec.Selector, "selector")...)
	allErrs = append(allErrs, ValidatePositiveField(int64(spec.Replicas), "replicas")...)
	allErrs = append(allErrs, ValidatePodTemplateSpecForRC(spec.Template, spec.Selector, spec.Replicas, "template")...)
	return allErrs
}

// ValidatePodTemplateSpec validates the spec of a pod template
func ValidatePodTemplateSpec(spec *api.PodTemplateSpec) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateLabels(spec.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(spec.Annotations, "annotations")...)
	allErrs = append(allErrs, ValidatePodSpec(&spec.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateReadOnlyPersistentDisks(volumes []api.Volume) validation.ErrorList {
	allErrs := validation.ErrorList{}
	for _, vol := range volumes {
		if vol.GCEPersistentDisk != nil {
			if vol.GCEPersistentDisk.ReadOnly == false {
				allErrs = append(allErrs, validation.NewFieldInvalid("GCEPersistentDisk.ReadOnly", false, "ReadOnly must be true for replicated pods > 1, as GCE PD can only be mounted on multiple machines if it is read-only."))
			}
		}
		// TODO: What to do for AWS?  It doesn't support replicas
	}
	return allErrs
}

// ValidateNode tests if required fields in the node are set.
func ValidateNode(node *api.Node) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&node.ObjectMeta, false, ValidateNodeName).Prefix("metadata")...)

	// Only validate spec. All status fields are optional and can be updated later.

	// external ID is required.
	if len(node.Spec.ExternalID) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("spec.ExternalID"))
	}

	// TODO(rjnagal): Ignore PodCIDR till its completely implemented.
	return allErrs
}

// ValidateNodeUpdate tests to make sure a node update can be applied.  Modifies oldNode.
func ValidateNodeUpdate(node, oldNode *api.Node) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&node.ObjectMeta, &oldNode.ObjectMeta).Prefix("metadata")...)

	// TODO: Enable the code once we have better api object.status update model. Currently,
	// anyone can update node status.
	// if !api.Semantic.DeepEqual(node.Status, api.NodeStatus{}) {
	// 	allErrs = append(allErrs, validation.NewFieldInvalid("status", node.Status, "status must be empty"))
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

	// TODO: Add a 'real' error type for this error and provide print actual diffs.
	if !api.Semantic.DeepEqual(oldNode, node) {
		glog.V(4).Infof("Update failed validation %#v vs %#v", oldNode, node)
		allErrs = append(allErrs, fmt.Errorf("update contains more than labels or capacity changes"))
	}

	return allErrs
}

// Validate compute resource typename.
// Refer to docs/design/resources.md for more details.
func validateResourceName(value string, field string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if !validation.IsQualifiedName(value) {
		return append(allErrs, validation.NewFieldInvalid(field, value, "resource typename: "+qualifiedNameErrorMsg))
	}

	if len(strings.Split(value, "/")) == 1 {
		if !api.IsStandardResourceName(value) {
			return append(allErrs, validation.NewFieldInvalid(field, value, "is neither a standard resource type nor is fully qualified"))
		}
	}

	return validation.ErrorList{}
}

// ValidateLimitRange tests if required fields in the LimitRange are set.
func ValidateLimitRange(limitRange *api.LimitRange) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&limitRange.ObjectMeta, true, ValidateLimitRangeName).Prefix("metadata")...)

	// ensure resource names are properly qualified per docs/design/resources.md
	limitTypeSet := map[api.LimitType]bool{}
	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		_, found := limitTypeSet[limit.Type]
		if found {
			allErrs = append(allErrs, validation.NewFieldDuplicate(fmt.Sprintf("spec.limits[%d].type", i), limit.Type))
		}
		limitTypeSet[limit.Type] = true

		keys := sets.String{}
		min := map[string]resource.Quantity{}
		max := map[string]resource.Quantity{}
		defaults := map[string]resource.Quantity{}
		defaultRequests := map[string]resource.Quantity{}
		maxLimitRequestRatios := map[string]resource.Quantity{}

		for k, q := range limit.Max {
			allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].max[%s]", i, k))...)
			keys.Insert(string(k))
			max[string(k)] = q
		}
		for k, q := range limit.Min {
			allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].min[%s]", i, k))...)
			keys.Insert(string(k))
			min[string(k)] = q
		}

		if limit.Type == api.LimitTypePod {
			if len(limit.Default) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid("spec.limits[%d].default", limit.Default, "Default is not supported when limit type is Pod"))
			}
			if len(limit.DefaultRequest) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid("spec.limits[%d].defaultRequest", limit.DefaultRequest, "DefaultRequest is not supported when limit type is Pod"))
			}
		} else {
			for k, q := range limit.Default {
				allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].default[%s]", i, k))...)
				keys.Insert(string(k))
				defaults[string(k)] = q
			}
			for k, q := range limit.DefaultRequest {
				allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].defaultRequest[%s]", i, k))...)
				keys.Insert(string(k))
				defaultRequests[string(k)] = q
			}
		}

		for k, q := range limit.MaxLimitRequestRatio {
			allErrs = append(allErrs, validateResourceName(string(k), fmt.Sprintf("spec.limits[%d].maxLimitRequestRatio[%s]", i, k))...)
			keys.Insert(string(k))
			maxLimitRequestRatios[string(k)] = q
		}

		for k := range keys {
			minQuantity, minQuantityFound := min[k]
			maxQuantity, maxQuantityFound := max[k]
			defaultQuantity, defaultQuantityFound := defaults[k]
			defaultRequestQuantity, defaultRequestQuantityFound := defaultRequests[k]
			maxRatio, maxRatioFound := maxLimitRequestRatios[k]

			if minQuantityFound && maxQuantityFound && minQuantity.Cmp(maxQuantity) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].min[%s]", i, k), minQuantity, fmt.Sprintf("min value %s is greater than max value %s", minQuantity.String(), maxQuantity.String())))
			}

			if defaultRequestQuantityFound && minQuantityFound && minQuantity.Cmp(defaultRequestQuantity) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].defaultRequest[%s]", i, k), defaultRequestQuantity, fmt.Sprintf("min value %s is greater than default request value %s", minQuantity.String(), defaultRequestQuantity.String())))
			}

			if defaultRequestQuantityFound && maxQuantityFound && defaultRequestQuantity.Cmp(maxQuantity) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].defaultRequest[%s]", i, k), defaultRequestQuantity, fmt.Sprintf("default request value %s is greater than max value %s", defaultRequestQuantity.String(), maxQuantity.String())))
			}

			if defaultRequestQuantityFound && defaultQuantityFound && defaultRequestQuantity.Cmp(defaultQuantity) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].defaultRequest[%s]", i, k), defaultRequestQuantity, fmt.Sprintf("default request value %s is greater than default limit value %s", defaultRequestQuantity.String(), defaultQuantity.String())))
			}

			if defaultQuantityFound && minQuantityFound && minQuantity.Cmp(defaultQuantity) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].default[%s]", i, k), minQuantity, fmt.Sprintf("min value %s is greater than default value %s", minQuantity.String(), defaultQuantity.String())))
			}

			if defaultQuantityFound && maxQuantityFound && defaultQuantity.Cmp(maxQuantity) > 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].default[%s]", i, k), maxQuantity, fmt.Sprintf("default value %s is greater than max value %s", defaultQuantity.String(), maxQuantity.String())))
			}
			if maxRatioFound && maxRatio.Cmp(*resource.NewQuantity(1, resource.DecimalSI)) < 0 {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].maxLimitRequestRatio[%s]", i, k), maxRatio, fmt.Sprintf("maxLimitRequestRatio %s is less than 1", maxRatio.String())))
			}
			if maxRatioFound && minQuantityFound && maxQuantityFound {
				maxRatioValue := float64(maxRatio.Value())
				minQuantityValue := minQuantity.Value()
				maxQuantityValue := maxQuantity.Value()
				if maxRatio.Value() < resource.MaxMilliValue && minQuantityValue < resource.MaxMilliValue && maxQuantityValue < resource.MaxMilliValue {
					maxRatioValue = float64(maxRatio.MilliValue()) / 1000
					minQuantityValue = minQuantity.MilliValue()
					maxQuantityValue = maxQuantity.MilliValue()
				}
				maxRatioLimit := float64(maxQuantityValue) / float64(minQuantityValue)
				if maxRatioValue > maxRatioLimit {
					allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("spec.limits[%d].maxLimitRequestRatio[%s]", i, k), maxRatio, fmt.Sprintf("maxLimitRequestRatio %s is greater than max/min = %f", maxRatio.String(), maxRatioLimit)))
				}
			}
		}
	}

	return allErrs
}

// ValidateServiceAccount tests if required fields in the ServiceAccount are set.
func ValidateServiceAccount(serviceAccount *api.ServiceAccount) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&serviceAccount.ObjectMeta, true, ValidateServiceAccountName).Prefix("metadata")...)
	return allErrs
}

// ValidateServiceAccountUpdate tests if required fields in the ServiceAccount are set.
func ValidateServiceAccountUpdate(newServiceAccount, oldServiceAccount *api.ServiceAccount) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newServiceAccount.ObjectMeta, &oldServiceAccount.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateServiceAccount(newServiceAccount)...)
	return allErrs
}

const SecretKeyFmt string = "\\.?" + validation.DNS1123LabelFmt + "(\\." + validation.DNS1123LabelFmt + ")*"

var secretKeyRegexp = regexp.MustCompile("^" + SecretKeyFmt + "$")

// IsSecretKey tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123), except that a leading dot is allowed
func IsSecretKey(value string) bool {
	return len(value) <= validation.DNS1123SubdomainMaxLength && secretKeyRegexp.MatchString(value)
}

// ValidateSecret tests if required fields in the Secret are set.
func ValidateSecret(secret *api.Secret) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&secret.ObjectMeta, true, ValidateSecretName).Prefix("metadata")...)

	totalSize := 0
	for key, value := range secret.Data {
		if !IsSecretKey(key) {
			allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("data[%s]", key), key, fmt.Sprintf("must have at most %d characters and match regex %s", validation.DNS1123SubdomainMaxLength, SecretKeyFmt)))
		}

		totalSize += len(value)
	}

	if totalSize > api.MaxSecretSize {
		allErrs = append(allErrs, validation.NewFieldForbidden("data", "Maximum secret size exceeded"))
	}

	switch secret.Type {
	case api.SecretTypeServiceAccountToken:
		// Only require Annotations[kubernetes.io/service-account.name]
		// Additional fields (like Annotations[kubernetes.io/service-account.uid] and Data[token]) might be contributed later by a controller loop
		if value := secret.Annotations[api.ServiceAccountNameKey]; len(value) == 0 {
			allErrs = append(allErrs, validation.NewFieldRequired(fmt.Sprintf("metadata.annotations[%s]", api.ServiceAccountNameKey)))
		}
	case api.SecretTypeOpaque, "":
	// no-op
	case api.SecretTypeDockercfg:
		dockercfgBytes, exists := secret.Data[api.DockerConfigKey]
		if !exists {
			allErrs = append(allErrs, validation.NewFieldRequired(fmt.Sprintf("data[%s]", api.DockerConfigKey)))
			break
		}

		// make sure that the content is well-formed json.
		if err := json.Unmarshal(dockercfgBytes, &map[string]interface{}{}); err != nil {
			allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("data[%s]", api.DockerConfigKey), "<secret contents redacted>", err.Error()))
		}

	default:
		// no-op
	}

	return allErrs
}

// ValidateSecretUpdate tests if required fields in the Secret are set.
func ValidateSecretUpdate(newSecret, oldSecret *api.Secret) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newSecret.ObjectMeta, &oldSecret.ObjectMeta).Prefix("metadata")...)

	if len(newSecret.Type) == 0 {
		newSecret.Type = oldSecret.Type
	}

	allErrs = append(allErrs, ValidateImmutableField(newSecret.Type, oldSecret.Type, "type")...)

	allErrs = append(allErrs, ValidateSecret(newSecret)...)
	return allErrs
}

func validateBasicResource(quantity resource.Quantity) validation.ErrorList {
	if quantity.Value() < 0 {
		return validation.ErrorList{validation.NewFieldInvalid("", quantity.Value(), "must be a valid resource quantity")}
	}
	return validation.ErrorList{}
}

// Validates resource requirement spec.
func ValidateResourceRequirements(requirements *api.ResourceRequirements) validation.ErrorList {
	allErrs := validation.ErrorList{}
	for resourceName, quantity := range requirements.Limits {
		// Validate resource name.
		allErrs = append(allErrs, validateResourceName(resourceName.String(), fmt.Sprintf("resources.limits[%s]", resourceName))...)
		if api.IsStandardResourceName(resourceName.String()) {
			allErrs = append(allErrs, validateBasicResource(quantity).Prefix(fmt.Sprintf("Resource %s: ", resourceName))...)
		}
		// Check that request <= limit.
		requestQuantity, exists := requirements.Requests[resourceName]
		if exists {
			var requestValue, limitValue int64
			requestValue = requestQuantity.Value()
			limitValue = quantity.Value()
			// Do a more precise comparison if possible (if the value won't overflow).
			if requestValue <= resource.MaxMilliValue && limitValue <= resource.MaxMilliValue {
				requestValue = requestQuantity.MilliValue()
				limitValue = quantity.MilliValue()
			}
			if limitValue < requestValue {
				allErrs = append(allErrs, validation.NewFieldInvalid(fmt.Sprintf("resources.limits[%s]", resourceName), quantity.String(), "limit cannot be smaller than request"))
			}
		}
	}
	for resourceName, quantity := range requirements.Requests {
		// Validate resource name.
		allErrs = append(allErrs, validateResourceName(resourceName.String(), fmt.Sprintf("resources.requests[%s]", resourceName))...)
		if api.IsStandardResourceName(resourceName.String()) {
			allErrs = append(allErrs, validateBasicResource(quantity).Prefix(fmt.Sprintf("Resource %s: ", resourceName))...)
		}
	}
	return allErrs
}

// ValidateResourceQuota tests if required fields in the ResourceQuota are set.
func ValidateResourceQuota(resourceQuota *api.ResourceQuota) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&resourceQuota.ObjectMeta, true, ValidateResourceQuotaName).Prefix("metadata")...)

	for k, v := range resourceQuota.Spec.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(resourceQuota.TypeMeta.Kind))...)
		allErrs = append(allErrs, validateResourceQuantityValue(string(k), v)...)
	}
	for k, v := range resourceQuota.Status.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(resourceQuota.TypeMeta.Kind))...)
		allErrs = append(allErrs, validateResourceQuantityValue(string(k), v)...)
	}
	for k, v := range resourceQuota.Status.Used {
		allErrs = append(allErrs, validateResourceName(string(k), string(resourceQuota.TypeMeta.Kind))...)
		allErrs = append(allErrs, validateResourceQuantityValue(string(k), v)...)
	}
	return allErrs
}

// validateResourceQuantityValue enforces that specified quantity is valid for specified resource
func validateResourceQuantityValue(resource string, value resource.Quantity) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidatePositiveQuantity(value, resource)...)
	if api.IsIntegerResourceName(resource) {
		if value.MilliValue()%int64(1000) != int64(0) {
			allErrs = append(allErrs, validation.NewFieldInvalid(resource, value, isNotIntegerErrorMsg))
		}
	}
	return allErrs
}

// ValidateResourceQuotaUpdate tests to see if the update is legal for an end user to make.
// newResourceQuota is updated with fields that cannot be changed.
func ValidateResourceQuotaUpdate(newResourceQuota, oldResourceQuota *api.ResourceQuota) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newResourceQuota.ObjectMeta, &oldResourceQuota.ObjectMeta).Prefix("metadata")...)
	for k, v := range newResourceQuota.Spec.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(newResourceQuota.TypeMeta.Kind))...)
		allErrs = append(allErrs, validateResourceQuantityValue(string(k), v)...)
	}
	newResourceQuota.Status = oldResourceQuota.Status
	return allErrs
}

// ValidateResourceQuotaStatusUpdate tests to see if the status update is legal for an end user to make.
// newResourceQuota is updated with fields that cannot be changed.
func ValidateResourceQuotaStatusUpdate(newResourceQuota, oldResourceQuota *api.ResourceQuota) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newResourceQuota.ObjectMeta, &oldResourceQuota.ObjectMeta).Prefix("metadata")...)
	if newResourceQuota.ResourceVersion == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("resourceVersion"))
	}
	for k, v := range newResourceQuota.Status.Hard {
		allErrs = append(allErrs, validateResourceName(string(k), string(newResourceQuota.TypeMeta.Kind))...)
		allErrs = append(allErrs, validateResourceQuantityValue(string(k), v)...)
	}
	for k, v := range newResourceQuota.Status.Used {
		allErrs = append(allErrs, validateResourceName(string(k), string(newResourceQuota.TypeMeta.Kind))...)
		allErrs = append(allErrs, validateResourceQuantityValue(string(k), v)...)
	}
	newResourceQuota.Spec = oldResourceQuota.Spec
	return allErrs
}

// ValidateNamespace tests if required fields are set.
func ValidateNamespace(namespace *api.Namespace) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&namespace.ObjectMeta, false, ValidateNamespaceName).Prefix("metadata")...)
	for i := range namespace.Spec.Finalizers {
		allErrs = append(allErrs, validateFinalizerName(string(namespace.Spec.Finalizers[i]))...)
	}
	return allErrs
}

// Validate finalizer names
func validateFinalizerName(stringValue string) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if !validation.IsQualifiedName(stringValue) {
		return append(allErrs, validation.NewFieldInvalid("spec.finalizers", stringValue, qualifiedNameErrorMsg))
	}

	if len(strings.Split(stringValue, "/")) == 1 {
		if !api.IsStandardFinalizerName(stringValue) {
			return append(allErrs, validation.NewFieldInvalid("spec.finalizers", stringValue, fmt.Sprintf("finalizer name is neither a standard finalizer name nor is it fully qualified")))
		}
	}

	return validation.ErrorList{}
}

// ValidateNamespaceUpdate tests to make sure a namespace update can be applied.
// newNamespace is updated with fields that cannot be changed
func ValidateNamespaceUpdate(newNamespace *api.Namespace, oldNamespace *api.Namespace) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newNamespace.ObjectMeta, &oldNamespace.ObjectMeta).Prefix("metadata")...)
	newNamespace.Spec.Finalizers = oldNamespace.Spec.Finalizers
	newNamespace.Status = oldNamespace.Status
	return allErrs
}

// ValidateNamespaceStatusUpdate tests to see if the update is legal for an end user to make. newNamespace is updated with fields
// that cannot be changed.
func ValidateNamespaceStatusUpdate(newNamespace, oldNamespace *api.Namespace) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newNamespace.ObjectMeta, &oldNamespace.ObjectMeta).Prefix("metadata")...)
	newNamespace.Spec = oldNamespace.Spec
	if newNamespace.DeletionTimestamp.IsZero() {
		if newNamespace.Status.Phase != api.NamespaceActive {
			allErrs = append(allErrs, validation.NewFieldInvalid("Status.Phase", newNamespace.Status.Phase, "A namespace may only be in active status if it does not have a deletion timestamp."))
		}
	} else {
		if newNamespace.Status.Phase != api.NamespaceTerminating {
			allErrs = append(allErrs, validation.NewFieldInvalid("Status.Phase", newNamespace.Status.Phase, "A namespace may only be in terminating status if it has a deletion timestamp."))
		}
	}
	return allErrs
}

// ValidateNamespaceFinalizeUpdate tests to see if the update is legal for an end user to make.
// newNamespace is updated with fields that cannot be changed.
func ValidateNamespaceFinalizeUpdate(newNamespace, oldNamespace *api.Namespace) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newNamespace.ObjectMeta, &oldNamespace.ObjectMeta).Prefix("metadata")...)
	for i := range newNamespace.Spec.Finalizers {
		allErrs = append(allErrs, validateFinalizerName(string(newNamespace.Spec.Finalizers[i]))...)
	}
	newNamespace.Status = oldNamespace.Status
	return allErrs
}

// ValidateEndpoints tests if required fields are set.
func ValidateEndpoints(endpoints *api.Endpoints) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&endpoints.ObjectMeta, true, ValidateEndpointsName).Prefix("metadata")...)
	allErrs = append(allErrs, validateEndpointSubsets(endpoints.Subsets).Prefix("subsets")...)
	return allErrs
}

func validateEndpointSubsets(subsets []api.EndpointSubset) validation.ErrorList {
	allErrs := validation.ErrorList{}

	for i := range subsets {
		ss := &subsets[i]

		ssErrs := validation.ErrorList{}

		if len(ss.Addresses) == 0 && len(ss.NotReadyAddresses) == 0 {
			ssErrs = append(ssErrs, validation.NewFieldRequired("addresses or notReadyAddresses"))
		}
		if len(ss.Ports) == 0 {
			ssErrs = append(ssErrs, validation.NewFieldRequired("ports"))
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

func validateEndpointAddress(address *api.EndpointAddress) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if !validation.IsValidIPv4(address.IP) {
		allErrs = append(allErrs, validation.NewFieldInvalid("ip", address.IP, "invalid IPv4 address"))
		return allErrs
	}
	return validateIpIsNotLinkLocalOrLoopback(address.IP, "ip")
}

func validateIpIsNotLinkLocalOrLoopback(ipAddress, fieldName string) validation.ErrorList {
	// We disallow some IPs as endpoints or external-ips.  Specifically, loopback addresses are
	// nonsensical and link-local addresses tend to be used for node-centric purposes (e.g. metadata service).
	allErrs := validation.ErrorList{}
	ip := net.ParseIP(ipAddress)
	if ip == nil {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, ipAddress, "not a valid IP address"))
		return allErrs
	}
	if ip.IsLoopback() {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, ipAddress, "may not be in the loopback range (127.0.0.0/8)"))
	}
	if ip.IsLinkLocalUnicast() {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, ipAddress, "may not be in the link-local range (169.254.0.0/16)"))
	}
	if ip.IsLinkLocalMulticast() {
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, ipAddress, "may not be in the link-local multicast range (224.0.0.0/24)"))
	}
	return allErrs
}

func validateEndpointPort(port *api.EndpointPort, requireName bool) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if requireName && port.Name == "" {
		allErrs = append(allErrs, validation.NewFieldRequired("name"))
	} else if port.Name != "" {
		if !validation.IsDNS1123Label(port.Name) {
			allErrs = append(allErrs, validation.NewFieldInvalid("name", port.Name, DNS1123LabelErrorMsg))
		}
	}
	if !validation.IsValidPortNum(port.Port) {
		allErrs = append(allErrs, validation.NewFieldInvalid("port", port.Port, PortRangeErrorMsg))
	}
	if len(port.Protocol) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("protocol"))
	} else if !supportedPortProtocols.Has(string(port.Protocol)) {
		allErrs = append(allErrs, validation.NewFieldNotSupported("protocol", port.Protocol, supportedPortProtocols.List()))
	}
	return allErrs
}

// ValidateEndpointsUpdate tests to make sure an endpoints update can be applied.
func ValidateEndpointsUpdate(newEndpoints, oldEndpoints *api.Endpoints) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newEndpoints.ObjectMeta, &oldEndpoints.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, validateEndpointSubsets(newEndpoints.Subsets).Prefix("subsets")...)
	return allErrs
}

// ValidateSecurityContext ensure the security context contains valid settings
func ValidateSecurityContext(sc *api.SecurityContext) validation.ErrorList {
	allErrs := validation.ErrorList{}
	//this should only be true for testing since SecurityContext is defaulted by the api
	if sc == nil {
		return allErrs
	}

	if sc.Privileged != nil {
		if *sc.Privileged && !capabilities.Get().AllowPrivileged {
			allErrs = append(allErrs, validation.NewFieldForbidden("privileged", sc.Privileged))
		}
	}

	if sc.RunAsUser != nil {
		if *sc.RunAsUser < 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("runAsUser", *sc.RunAsUser, "runAsUser cannot be negative"))
		}
	}
	return allErrs
}

func ValidatePodLogOptions(opts *api.PodLogOptions) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if opts.TailLines != nil && *opts.TailLines < 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("tailLines", *opts.TailLines, "tailLines must be a non-negative integer or nil"))
	}
	if opts.LimitBytes != nil && *opts.LimitBytes < 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("limitBytes", *opts.LimitBytes, "limitBytes must be a positive integer or nil"))
	}
	switch {
	case opts.SinceSeconds != nil && opts.SinceTime != nil:
		allErrs = append(allErrs, validation.NewFieldInvalid("sinceSeconds", *opts.SinceSeconds, "only one of sinceTime or sinceSeconds can be provided"))
		allErrs = append(allErrs, validation.NewFieldInvalid("sinceTime", *opts.SinceTime, "only one of sinceTime or sinceSeconds can be provided"))
	case opts.SinceSeconds != nil:
		if *opts.SinceSeconds < 1 {
			allErrs = append(allErrs, validation.NewFieldInvalid("sinceSeconds", *opts.SinceSeconds, "sinceSeconds must be a positive integer"))
		}
	}
	return allErrs
}

// ValidateLoadBalancerStatus validates required fields on a LoadBalancerStatus
func ValidateLoadBalancerStatus(status *api.LoadBalancerStatus) validation.ErrorList {
	allErrs := validation.ErrorList{}
	for _, ingress := range status.Ingress {
		if len(ingress.IP) > 0 {
			if isIP := (net.ParseIP(ingress.IP) != nil); !isIP {
				allErrs = append(allErrs, validation.NewFieldInvalid("ingress.ip", ingress.IP, "must be an IP address"))
			}
		}
		if len(ingress.Hostname) > 0 {
			if valid, errMsg := NameIsDNSSubdomain(ingress.Hostname, false); !valid {
				allErrs = append(allErrs, validation.NewFieldInvalid("ingress.hostname", ingress.Hostname, errMsg))
			}
			if isIP := (net.ParseIP(ingress.Hostname) != nil); isIP {
				allErrs = append(allErrs, validation.NewFieldInvalid("ingress.hostname", ingress.Hostname, "must be a DNS name, not an IP address"))
			}
		}
	}
	return allErrs
}
