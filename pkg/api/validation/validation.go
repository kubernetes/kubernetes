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
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path"
	"reflect"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/endpoints"
	utilpod "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/api/resource"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/unversioned"
	unversionedvalidation "k8s.io/kubernetes/pkg/api/unversioned/validation"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// TODO: delete this global variable when we enable the validation of common
// fields by default.
var RepairMalformedUpdates bool = true

const isNegativeErrorMsg string = `must be greater than or equal to 0`
const isInvalidQuotaResource string = `must be a standard resource for quota`
const fieldImmutableErrorMsg string = `field is immutable`
const isNotIntegerErrorMsg string = `must be an integer`

var pdPartitionErrorMsg string = validation.InclusiveRangeError(1, 255)

const totalAnnotationSizeLimitB int = 256 * (1 << 10) // 256 kB

// BannedOwners is a black list of object that are not allowed to be owners.
var BannedOwners = map[unversioned.GroupVersionKind]struct{}{
	v1.SchemeGroupVersion.WithKind("Event"): {},
}

// ValidateHasLabel requires that api.ObjectMeta has a Label with key and expectedValue
func ValidateHasLabel(meta api.ObjectMeta, fldPath *field.Path, key, expectedValue string) field.ErrorList {
	allErrs := field.ErrorList{}
	actualValue, found := meta.Labels[key]
	if !found {
		allErrs = append(allErrs, field.Required(fldPath.Child("labels").Key(key),
			fmt.Sprintf("must be '%s'", expectedValue)))
		return allErrs
	}
	if actualValue != expectedValue {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("labels").Key(key), meta.Labels,
			fmt.Sprintf("must be '%s'", expectedValue)))
	}
	return allErrs
}

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	var totalSize int64
	for k, v := range annotations {
		for _, msg := range validation.IsQualifiedName(strings.ToLower(k)) {
			allErrs = append(allErrs, field.Invalid(fldPath, k, msg))
		}
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}
	if totalSize > (int64)(totalAnnotationSizeLimitB) {
		allErrs = append(allErrs, field.TooLong(fldPath, "", totalAnnotationSizeLimitB))
	}
	return allErrs
}

func ValidateDNS1123Label(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsDNS1123Label(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	return allErrs
}

func ValidatePodSpecificAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if annotations[api.AffinityAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateAffinityInPodAnnotations(annotations, fldPath)...)
	}

	if annotations[api.TolerationsAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateTolerationsInPodAnnotations(annotations, fldPath)...)
	}

	// TODO: remove these after we EOL the annotations.
	if hostname, exists := annotations[utilpod.PodHostnameAnnotation]; exists {
		allErrs = append(allErrs, ValidateDNS1123Label(hostname, fldPath.Key(utilpod.PodHostnameAnnotation))...)
	}
	if subdomain, exists := annotations[utilpod.PodSubdomainAnnotation]; exists {
		allErrs = append(allErrs, ValidateDNS1123Label(subdomain, fldPath.Key(utilpod.PodSubdomainAnnotation))...)
	}

	allErrs = append(allErrs, ValidateSeccompPodAnnotations(annotations, fldPath)...)

	return allErrs
}

func ValidateEndpointsSpecificAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO: remove this after we EOL the annotation.
	hostnamesMap, exists := annotations[endpoints.PodHostnamesAnnotation]
	if exists && !isValidHostnamesMap(hostnamesMap) {
		allErrs = append(allErrs, field.Invalid(fldPath, endpoints.PodHostnamesAnnotation,
			`must be a valid json representation of map[string(IP)][HostRecord] e.g. "{"10.245.1.6":{"HostName":"my-webserver"}}"`))
	}

	return allErrs
}

func validateOwnerReference(ownerReference api.OwnerReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	gvk := unversioned.FromAPIVersionAndKind(ownerReference.APIVersion, ownerReference.Kind)
	// gvk.Group is empty for the legacy group.
	if len(gvk.Version) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("apiVersion"), ownerReference.APIVersion, "version must not be empty"))
	}
	if len(gvk.Kind) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("kind"), ownerReference.Kind, "kind must not be empty"))
	}
	if len(ownerReference.Name) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), ownerReference.Name, "name must not be empty"))
	}
	if len(ownerReference.UID) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("uid"), ownerReference.UID, "uid must not be empty"))
	}
	if _, ok := BannedOwners[gvk]; ok {
		allErrs = append(allErrs, field.Invalid(fldPath, ownerReference, fmt.Sprintf("%s is disallowed from being an owner", gvk)))
	}
	return allErrs
}

func ValidateOwnerReferences(ownerReferences []api.OwnerReference, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	controllerName := ""
	for _, ref := range ownerReferences {
		allErrs = append(allErrs, validateOwnerReference(ref, fldPath)...)
		if ref.Controller != nil && *ref.Controller {
			if controllerName != "" {
				allErrs = append(allErrs, field.Invalid(fldPath, ownerReferences,
					fmt.Sprintf("Only one reference can have Controller set to true. Found \"true\" in references for %v and %v", controllerName, ref.Name)))
			} else {
				controllerName = ref.Name
			}
		}
	}
	return allErrs
}

// ValidateNameFunc validates that the provided name is valid for a given resource type.
// Not all resources have the same validation rules for names. Prefix is true
// if the name will have a value appended to it.  If the name is not valid,
// this returns a list of descriptions of individual characteristics of the
// value that were not valid.  Otherwise this returns an empty list or nil.
type ValidateNameFunc func(name string, prefix bool) []string

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
var ValidatePodName = NameIsDNSSubdomain

// ValidateReplicationControllerName can be used to check whether the given replication
// controller name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateReplicationControllerName = NameIsDNSSubdomain

// ValidateServiceName can be used to check whether the given service name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateServiceName = NameIsDNS1035Label

// ValidateNodeName can be used to check whether the given node name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateNodeName = NameIsDNSSubdomain

// ValidateNamespaceName can be used to check whether the given namespace name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateNamespaceName = NameIsDNSLabel

// ValidateLimitRangeName can be used to check whether the given limit range name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateLimitRangeName = NameIsDNSSubdomain

// ValidateResourceQuotaName can be used to check whether the given
// resource quota name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateResourceQuotaName = NameIsDNSSubdomain

// ValidateSecretName can be used to check whether the given secret name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateSecretName = NameIsDNSSubdomain

// ValidateServiceAccountName can be used to check whether the given service account name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateServiceAccountName = NameIsDNSSubdomain

// ValidateEndpointsName can be used to check whether the given endpoints name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateEndpointsName = NameIsDNSSubdomain

// NameIsDNSSubdomain is a ValidateNameFunc for names that must be a DNS subdomain.
func NameIsDNSSubdomain(name string, prefix bool) []string {
	if prefix {
		name = maskTrailingDash(name)
	}
	return validation.IsDNS1123Subdomain(name)
}

// NameIsDNSLabel is a ValidateNameFunc for names that must be a DNS 1123 label.
func NameIsDNSLabel(name string, prefix bool) []string {
	if prefix {
		name = maskTrailingDash(name)
	}
	return validation.IsDNS1123Label(name)
}

// NameIsDNS1035Label is a ValidateNameFunc for names that must be a DNS 952 label.
func NameIsDNS1035Label(name string, prefix bool) []string {
	if prefix {
		name = maskTrailingDash(name)
	}
	return validation.IsDNS1035Label(name)
}

// Validates that given value is not negative.
func ValidateNonnegativeField(value int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value, isNegativeErrorMsg))
	}
	return allErrs
}

// Validates that a Quantity is not negative
func ValidateNonnegativeQuantity(value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if value.Cmp(resource.Quantity{}) < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, value.String(), isNegativeErrorMsg))
	}
	return allErrs
}

func ValidateImmutableField(newVal, oldVal interface{}, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !api.Semantic.DeepEqual(oldVal, newVal) {
		allErrs = append(allErrs, field.Invalid(fldPath, newVal, fieldImmutableErrorMsg))
	}
	return allErrs
}

// ValidateObjectMeta validates an object's metadata on creation. It expects that name generation has already
// been performed.
// It doesn't return an error for rootscoped resources with namespace, because namespace should already be cleared before.
// TODO: Remove calls to this method scattered in validations of specific resources, e.g., ValidatePodUpdate.
func ValidateObjectMeta(meta *api.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(meta.GenerateName) != 0 {
		for _, msg := range nameFn(meta.GenerateName, true) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("generateName"), meta.GenerateName, msg))
		}
	}
	// If the generated name validates, but the calculated value does not, it's a problem with generation, and we
	// report it here. This may confuse users, but indicates a programming bug and still must be validated.
	// If there are multiple fields out of which one is required then add a or as a separator
	if len(meta.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "name or generateName is required"))
	} else {
		for _, msg := range nameFn(meta.Name, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("name"), meta.Name, msg))
		}
	}
	if requiresNamespace {
		if len(meta.Namespace) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("namespace"), ""))
		} else {
			for _, msg := range ValidateNamespaceName(meta.Namespace, false) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), meta.Namespace, msg))
			}
		}
	} else {
		if len(meta.Namespace) != 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("namespace"), "not allowed on this type"))
		}
	}
	allErrs = append(allErrs, ValidateNonnegativeField(meta.Generation, fldPath.Child("generation"))...)
	allErrs = append(allErrs, unversionedvalidation.ValidateLabels(meta.Labels, fldPath.Child("labels"))...)
	allErrs = append(allErrs, ValidateAnnotations(meta.Annotations, fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidateOwnerReferences(meta.OwnerReferences, fldPath.Child("ownerReferences"))...)
	for _, finalizer := range meta.Finalizers {
		allErrs = append(allErrs, validateFinalizerName(finalizer, fldPath.Child("finalizers"))...)
	}
	return allErrs
}

// ValidateObjectMetaUpdate validates an object's metadata when updated
func ValidateObjectMetaUpdate(newMeta, oldMeta *api.ObjectMeta, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if !RepairMalformedUpdates && newMeta.UID != oldMeta.UID {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("uid"), newMeta.UID, "field is immutable"))
	}
	// in the event it is left empty, set it, to allow clients more flexibility
	// TODO: remove the following code that repairs the update request when we retire the clients that modify the immutable fields.
	// Please do not copy this pattern elsewhere; validation functions should not be modifying the objects they are passed!
	if RepairMalformedUpdates {
		if len(newMeta.UID) == 0 {
			newMeta.UID = oldMeta.UID
		}
		// ignore changes to timestamp
		if oldMeta.CreationTimestamp.IsZero() {
			oldMeta.CreationTimestamp = newMeta.CreationTimestamp
		} else {
			newMeta.CreationTimestamp = oldMeta.CreationTimestamp
		}
		// an object can never remove a deletion timestamp or clear/change grace period seconds
		if !oldMeta.DeletionTimestamp.IsZero() {
			newMeta.DeletionTimestamp = oldMeta.DeletionTimestamp
		}
		if oldMeta.DeletionGracePeriodSeconds != nil && newMeta.DeletionGracePeriodSeconds == nil {
			newMeta.DeletionGracePeriodSeconds = oldMeta.DeletionGracePeriodSeconds
		}
	}

	// TODO: needs to check if newMeta==nil && oldMeta !=nil after the repair logic is removed.
	if newMeta.DeletionGracePeriodSeconds != nil && (oldMeta.DeletionGracePeriodSeconds == nil || *newMeta.DeletionGracePeriodSeconds != *oldMeta.DeletionGracePeriodSeconds) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("deletionGracePeriodSeconds"), newMeta.DeletionGracePeriodSeconds, "field is immutable; may only be changed via deletion"))
	}
	if newMeta.DeletionTimestamp != nil && (oldMeta.DeletionTimestamp == nil || !newMeta.DeletionTimestamp.Equal(*oldMeta.DeletionTimestamp)) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("deletionTimestamp"), newMeta.DeletionTimestamp, "field is immutable; may only be changed via deletion"))
	}

	// Reject updates that don't specify a resource version
	if len(newMeta.ResourceVersion) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceVersion"), newMeta.ResourceVersion, "must be specified for an update"))
	}

	// Generation shouldn't be decremented
	if newMeta.Generation < oldMeta.Generation {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("generation"), newMeta.Generation, "must not be decremented"))
	}

	allErrs = append(allErrs, ValidateImmutableField(newMeta.Name, oldMeta.Name, fldPath.Child("name"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.Namespace, oldMeta.Namespace, fldPath.Child("namespace"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.UID, oldMeta.UID, fldPath.Child("uid"))...)
	allErrs = append(allErrs, ValidateImmutableField(newMeta.CreationTimestamp, oldMeta.CreationTimestamp, fldPath.Child("creationTimestamp"))...)

	allErrs = append(allErrs, unversionedvalidation.ValidateLabels(newMeta.Labels, fldPath.Child("labels"))...)
	allErrs = append(allErrs, ValidateAnnotations(newMeta.Annotations, fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidateOwnerReferences(newMeta.OwnerReferences, fldPath.Child("ownerReferences"))...)

	return allErrs
}

func validateVolumes(volumes []api.Volume, fldPath *field.Path) (sets.String, field.ErrorList) {
	allErrs := field.ErrorList{}

	allNames := sets.String{}
	for i, vol := range volumes {
		idxPath := fldPath.Index(i)
		namePath := idxPath.Child("name")
		el := validateVolumeSource(&vol.VolumeSource, idxPath)
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
		} else {
			allErrs = append(allErrs, el...)
		}

	}
	return allNames, allErrs
}

func validateVolumeSource(source *api.VolumeSource, fldPath *field.Path) field.ErrorList {
	numVolumes := 0
	allErrs := field.ErrorList{}
	if source.EmptyDir != nil {
		numVolumes++
		// EmptyDirs have nothing to validate
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
	}
	if source.Glusterfs != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("glusterfs"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateGlusterfs(source.Glusterfs, fldPath.Child("glusterfs"))...)
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
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("flexVolume"), "may not specifiy more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateFlexVolumeSource(source.FlexVolume, fldPath.Child("flexVolume"))...)
		}
	}
	if source.ConfigMap != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("configMap"), "may not specifiy more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateConfigMapVolumeSource(source.ConfigMap, fldPath.Child("configMap"))...)
		}
	}
	if source.AzureFile != nil {
		numVolumes++
		allErrs = append(allErrs, validateAzureFile(source.AzureFile, fldPath.Child("azureFile"))...)
	}
	if source.VsphereVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("vsphereVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateVsphereVolumeSource(source.VsphereVolume, fldPath.Child("vsphereVolume"))...)
		}
	}
	if numVolumes == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "must specify a volume type"))
	}

	return allErrs
}

func validateHostPathVolumeSource(hostPath *api.HostPathVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(hostPath.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	return allErrs
}

func validateGitRepoVolumeSource(gitRepo *api.GitRepoVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(gitRepo.Repository) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("repository"), ""))
	}

	pathErrs := validateLocalDescendingPath(gitRepo.Directory, fldPath.Child("directory"))
	allErrs = append(allErrs, pathErrs...)
	return allErrs
}

func validateISCSIVolumeSource(iscsi *api.ISCSIVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(iscsi.TargetPortal) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetPortal"), ""))
	}
	if len(iscsi.IQN) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("iqn"), ""))
	}
	if iscsi.Lun < 0 || iscsi.Lun > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("lun"), iscsi.Lun, validation.InclusiveRangeError(0, 255)))
	}
	return allErrs
}

func validateFCVolumeSource(fc *api.FCVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(fc.TargetWWNs) < 1 {
		allErrs = append(allErrs, field.Required(fldPath.Child("targetWWNs"), ""))
	}

	if fc.Lun == nil {
		allErrs = append(allErrs, field.Required(fldPath.Child("lun"), ""))
	} else {
		if *fc.Lun < 0 || *fc.Lun > 255 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("lun"), fc.Lun, validation.InclusiveRangeError(0, 255)))
		}
	}
	return allErrs
}

func validateGCEPersistentDiskVolumeSource(pd *api.GCEPersistentDiskVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(pd.PDName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("pdName"), ""))
	}
	if pd.Partition < 0 || pd.Partition > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("partition"), pd.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateAWSElasticBlockStoreVolumeSource(PD *api.AWSElasticBlockStoreVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(PD.VolumeID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeID"), ""))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("partition"), PD.Partition, pdPartitionErrorMsg))
	}
	return allErrs
}

func validateSecretVolumeSource(secretSource *api.SecretVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(secretSource.SecretName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretName"), ""))
	}
	itemsPath := fldPath.Child("items")
	for i, kp := range secretSource.Items {
		itemPath := itemsPath.Index(i)
		allErrs = append(allErrs, validateKeyToPath(&kp, itemPath)...)
	}
	return allErrs
}

func validateConfigMapVolumeSource(configMapSource *api.ConfigMapVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(configMapSource.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}
	itemsPath := fldPath.Child("items")
	for i, kp := range configMapSource.Items {
		itemPath := itemsPath.Index(i)
		allErrs = append(allErrs, validateKeyToPath(&kp, itemPath)...)
	}
	return allErrs
}

func validateKeyToPath(kp *api.KeyToPath, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(kp.Key) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("key"), ""))
	}
	if len(kp.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	allErrs = append(allErrs, validateLocalNonReservedPath(kp.Path, fldPath.Child("path"))...)
	return allErrs
}

func validatePersistentClaimVolumeSource(claim *api.PersistentVolumeClaimVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(claim.ClaimName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("claimName"), ""))
	}
	return allErrs
}

func validateNFSVolumeSource(nfs *api.NFSVolumeSource, fldPath *field.Path) field.ErrorList {
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

func validateGlusterfs(glusterfs *api.GlusterfsVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(glusterfs.EndpointsName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("endpoints"), ""))
	}
	if len(glusterfs.Path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
	}
	return allErrs
}

func validateFlockerVolumeSource(flocker *api.FlockerVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(flocker.DatasetName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("datasetName"), ""))
	}
	if strings.Contains(flocker.DatasetName, "/") {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("datasetName"), flocker.DatasetName, "must not contain '/'"))
	}
	return allErrs
}

var validDownwardAPIFieldPathExpressions = sets.NewString(
	"metadata.name",
	"metadata.namespace",
	"metadata.labels",
	"metadata.annotations")

func validateDownwardAPIVolumeSource(downwardAPIVolume *api.DownwardAPIVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, file := range downwardAPIVolume.Items {
		if len(file.Path) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("path"), ""))
		}
		allErrs = append(allErrs, validateLocalNonReservedPath(file.Path, fldPath.Child("path"))...)
		if file.FieldRef != nil {
			allErrs = append(allErrs, validateObjectFieldSelector(file.FieldRef, &validDownwardAPIFieldPathExpressions, fldPath.Child("fieldRef"))...)
			if file.ResourceFieldRef != nil {
				allErrs = append(allErrs, field.Invalid(fldPath, "resource", "fieldRef and resourceFieldRef can not be specified simultaneously"))
			}
		} else if file.ResourceFieldRef != nil {
			allErrs = append(allErrs, validateContainerResourceFieldSelector(file.ResourceFieldRef, &validContainerResourceFieldPathExpressions, fldPath.Child("resourceFieldRef"), true)...)
		} else {
			allErrs = append(allErrs, field.Required(fldPath, "one of fieldRef and resourceFieldRef is required"))
		}
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

	// TODO: this assumes the OS of apiserver & nodes are the same
	parts := strings.Split(targetPath, string(os.PathSeparator))
	for _, item := range parts {
		if item == ".." {
			allErrs = append(allErrs, field.Invalid(fldPath, targetPath, "must not contain '..'"))
			break // even for `../../..`, one error is sufficient to make the point
		}
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

func validateRBDVolumeSource(rbd *api.RBDVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(rbd.CephMonitors) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("monitors"), ""))
	}
	if len(rbd.RBDImage) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("image"), ""))
	}
	return allErrs
}

func validateCinderVolumeSource(cd *api.CinderVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cd.VolumeID) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumeID"), ""))
	}
	return allErrs
}

func validateCephFSVolumeSource(cephfs *api.CephFSVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cephfs.Monitors) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("monitors"), ""))
	}
	return allErrs
}

func validateFlexVolumeSource(fv *api.FlexVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(fv.Driver) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("driver"), ""))
	}
	return allErrs
}

func validateAzureFile(azure *api.AzureFileVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if azure.SecretName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("secretName"), ""))
	}
	if azure.ShareName == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("shareName"), ""))
	}
	return allErrs
}

func validateVsphereVolumeSource(cd *api.VsphereVirtualDiskVolumeSource, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cd.VolumePath) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("volumePath"), ""))
	}
	return allErrs
}

// ValidatePersistentVolumeName checks that a name is appropriate for a
// PersistentVolumeName object.
var ValidatePersistentVolumeName = NameIsDNSSubdomain

var supportedAccessModes = sets.NewString(string(api.ReadWriteOnce), string(api.ReadOnlyMany), string(api.ReadWriteMany))

func ValidatePersistentVolume(pv *api.PersistentVolume) field.ErrorList {
	allErrs := ValidateObjectMeta(&pv.ObjectMeta, false, ValidatePersistentVolumeName, field.NewPath("metadata"))

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

	if _, ok := pv.Spec.Capacity[api.ResourceStorage]; !ok || len(pv.Spec.Capacity) > 1 {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("capacity"), pv.Spec.Capacity, []string{string(api.ResourceStorage)}))
	}
	capPath := specPath.Child("capacity")
	for r, qty := range pv.Spec.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty, capPath.Key(string(r)))...)
	}

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
			allErrs = append(allErrs, validateGlusterfs(pv.Spec.Glusterfs, specPath.Child("glusterfs"))...)
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
			allErrs = append(allErrs, validateRBDVolumeSource(pv.Spec.RBD, specPath.Child("rbd"))...)
		}
	}
	if pv.Spec.CephFS != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("cephFS"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateCephFSVolumeSource(pv.Spec.CephFS, specPath.Child("cephfs"))...)
		}
	}
	if pv.Spec.ISCSI != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("iscsi"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateISCSIVolumeSource(pv.Spec.ISCSI, specPath.Child("iscsi"))...)
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
		allErrs = append(allErrs, validateFlexVolumeSource(pv.Spec.FlexVolume, specPath.Child("flexVolume"))...)
	}
	if pv.Spec.AzureFile != nil {
		numVolumes++
		allErrs = append(allErrs, validateAzureFile(pv.Spec.AzureFile, specPath.Child("azureFile"))...)
	}
	if pv.Spec.VsphereVolume != nil {
		if numVolumes > 0 {
			allErrs = append(allErrs, field.Forbidden(specPath.Child("vsphereVolume"), "may not specify more than 1 volume type"))
		} else {
			numVolumes++
			allErrs = append(allErrs, validateVsphereVolumeSource(pv.Spec.VsphereVolume, specPath.Child("vsphereVolume"))...)
		}
	}
	if numVolumes == 0 {
		allErrs = append(allErrs, field.Required(specPath, "must specify a volume type"))
	}
	return allErrs
}

// ValidatePersistentVolumeUpdate tests to see if the update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeUpdate(newPv, oldPv *api.PersistentVolume) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = ValidatePersistentVolume(newPv)
	newPv.Status = oldPv.Status
	return allErrs
}

// ValidatePersistentVolumeStatusUpdate tests to see if the status update is legal for an end user to make.
// newPv is updated with fields that cannot be changed.
func ValidatePersistentVolumeStatusUpdate(newPv, oldPv *api.PersistentVolume) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPv.ObjectMeta, &oldPv.ObjectMeta, field.NewPath("metadata"))
	if len(newPv.ResourceVersion) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("resourceVersion"), ""))
	}
	newPv.Spec = oldPv.Spec
	return allErrs
}

func ValidatePersistentVolumeClaim(pvc *api.PersistentVolumeClaim) field.ErrorList {
	allErrs := ValidateObjectMeta(&pvc.ObjectMeta, true, ValidatePersistentVolumeName, field.NewPath("metadata"))
	specPath := field.NewPath("spec")
	if len(pvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("accessModes"), "at least 1 accessMode is required"))
	}
	if pvc.Spec.Selector != nil {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(pvc.Spec.Selector, specPath.Child("selector"))...)
	}
	for _, mode := range pvc.Spec.AccessModes {
		if mode != api.ReadWriteOnce && mode != api.ReadOnlyMany && mode != api.ReadWriteMany {
			allErrs = append(allErrs, field.NotSupported(specPath.Child("accessModes"), mode, supportedAccessModes.List()))
		}
	}
	if _, ok := pvc.Spec.Resources.Requests[api.ResourceStorage]; !ok {
		allErrs = append(allErrs, field.Required(specPath.Child("resources").Key(string(api.ResourceStorage)), ""))
	}
	return allErrs
}

func ValidatePersistentVolumeClaimUpdate(newPvc, oldPvc *api.PersistentVolumeClaim) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPvc.ObjectMeta, &oldPvc.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePersistentVolumeClaim(newPvc)...)
	// PVController needs to update PVC.Spec w/ VolumeName.
	// Claims are immutable in order to enforce quota, range limits, etc. without gaming the system.
	if len(oldPvc.Spec.VolumeName) == 0 {
		// volumeName changes are allowed once.
		// Reset back to empty string after equality check
		oldPvc.Spec.VolumeName = newPvc.Spec.VolumeName
		defer func() { oldPvc.Spec.VolumeName = "" }()
	}
	// changes to Spec are not allowed, but updates to label/annotations are OK.
	// no-op updates pass validation.
	if !api.Semantic.DeepEqual(newPvc.Spec, oldPvc.Spec) {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec"), "field is immutable after creation"))
	}
	newPvc.Status = oldPvc.Status
	return allErrs
}

func ValidatePersistentVolumeClaimStatusUpdate(newPvc, oldPvc *api.PersistentVolumeClaim) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPvc.ObjectMeta, &oldPvc.ObjectMeta, field.NewPath("metadata"))
	if len(newPvc.ResourceVersion) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("resourceVersion"), ""))
	}
	if len(newPvc.Spec.AccessModes) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("Spec", "accessModes"), ""))
	}
	capPath := field.NewPath("status", "capacity")
	for r, qty := range newPvc.Status.Capacity {
		allErrs = append(allErrs, validateBasicResource(qty, capPath.Key(string(r)))...)
	}
	newPvc.Spec = oldPvc.Spec
	return allErrs
}

var supportedPortProtocols = sets.NewString(string(api.ProtocolTCP), string(api.ProtocolUDP))

func validateContainerPorts(ports []api.ContainerPort, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allNames := sets.String{}
	for i, port := range ports {
		idxPath := fldPath.Index(i)
		if len(port.Name) > 0 {
			if msgs := validation.IsValidPortName(port.Name); len(msgs) != 0 {
				for i = range msgs {
					allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), port.Name, msgs[i]))
				}
			} else if allNames.Has(port.Name) {
				allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if port.ContainerPort == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("containerPort"), ""))
		} else {
			for _, msg := range validation.IsValidPortNum(int(port.ContainerPort)) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("containerPort"), port.ContainerPort, msg))
			}
		}
		if port.HostPort != 0 {
			for _, msg := range validation.IsValidPortNum(int(port.HostPort)) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostPort"), port.HostPort, msg))
			}
		}
		if len(port.Protocol) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("protocol"), ""))
		} else if !supportedPortProtocols.Has(string(port.Protocol)) {
			allErrs = append(allErrs, field.NotSupported(idxPath.Child("protocol"), port.Protocol, supportedPortProtocols.List()))
		}
	}
	return allErrs
}

func validateEnv(vars []api.EnvVar, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i, ev := range vars {
		idxPath := fldPath.Index(i)
		if len(ev.Name) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("name"), ""))
		} else {
			for _, msg := range validation.IsCIdentifier(ev.Name) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), ev.Name, msg))
			}
		}
		allErrs = append(allErrs, validateEnvVarValueFrom(ev, idxPath.Child("valueFrom"))...)
	}
	return allErrs
}

var validFieldPathExpressionsEnv = sets.NewString("metadata.name", "metadata.namespace", "status.podIP")
var validContainerResourceFieldPathExpressions = sets.NewString("limits.cpu", "limits.memory", "requests.cpu", "requests.memory")

func validateEnvVarValueFrom(ev api.EnvVar, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if ev.ValueFrom == nil {
		return allErrs
	}

	numSources := 0

	if ev.ValueFrom.FieldRef != nil {
		numSources++
		allErrs = append(allErrs, validateObjectFieldSelector(ev.ValueFrom.FieldRef, &validFieldPathExpressionsEnv, fldPath.Child("fieldRef"))...)
	}
	if ev.ValueFrom.ResourceFieldRef != nil {
		numSources++
		allErrs = append(allErrs, validateContainerResourceFieldSelector(ev.ValueFrom.ResourceFieldRef, &validContainerResourceFieldPathExpressions, fldPath.Child("resourceFieldRef"), false)...)
	}
	if ev.ValueFrom.ConfigMapKeyRef != nil {
		numSources++
		allErrs = append(allErrs, validateConfigMapKeySelector(ev.ValueFrom.ConfigMapKeyRef, fldPath.Child("configMapKeyRef"))...)
	}
	if ev.ValueFrom.SecretKeyRef != nil {
		numSources++
		allErrs = append(allErrs, validateSecretKeySelector(ev.ValueFrom.SecretKeyRef, fldPath.Child("secretKeyRef"))...)
	}

	if len(ev.Value) != 0 {
		if numSources != 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, "", "may not be specified when `value` is not empty"))
		}
	} else if numSources != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "may not have more than one field specified at a time"))
	}

	return allErrs
}

func validateObjectFieldSelector(fs *api.ObjectFieldSelector, expressions *sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(fs.APIVersion) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("apiVersion"), ""))
	} else if len(fs.FieldPath) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("fieldPath"), ""))
	} else {
		internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(fs.APIVersion, "Pod", fs.FieldPath, "")
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("fieldPath"), fs.FieldPath, fmt.Sprintf("error converting fieldPath: %v", err)))
		} else if !expressions.Has(internalFieldPath) {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("fieldPath"), internalFieldPath, expressions.List()))
		}
	}

	return allErrs
}

func validateContainerResourceFieldSelector(fs *api.ResourceFieldSelector, expressions *sets.String, fldPath *field.Path, volume bool) field.ErrorList {
	allErrs := field.ErrorList{}

	if volume && len(fs.ContainerName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("containerName"), ""))
	} else if len(fs.Resource) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("resource"), ""))
	} else if !expressions.Has(fs.Resource) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("resource"), fs.Resource, expressions.List()))
	}
	allErrs = append(allErrs, validateContainerResourceDivisor(fs.Resource, fs.Divisor, fldPath)...)
	return allErrs
}

var validContainerResourceDivisorForCPU = sets.NewString("1m", "1")
var validContainerResourceDivisorForMemory = sets.NewString("1", "1k", "1M", "1G", "1T", "1P", "1E", "1Ki", "1Mi", "1Gi", "1Ti", "1Pi", "1Ei")

func validateContainerResourceDivisor(rName string, divisor resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	unsetDivisor := resource.Quantity{}
	if unsetDivisor.Cmp(divisor) == 0 {
		return allErrs
	}
	switch rName {
	case "limits.cpu", "requests.cpu":
		if !validContainerResourceDivisorForCPU.Has(divisor.String()) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("divisor"), rName, "only divisor's values 1m and 1 are supported with the cpu resource"))
		}
	case "limits.memory", "requests.memory":
		if !validContainerResourceDivisorForMemory.Has(divisor.String()) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("divisor"), rName, "only divisor's values 1, 1k, 1M, 1G, 1T, 1P, 1E, 1Ki, 1Mi, 1Gi, 1Ti, 1Pi, 1Ei are supported with the memory resource"))
		}
	}
	return allErrs
}

func validateConfigMapKeySelector(s *api.ConfigMapKeySelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(s.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}
	if len(s.Key) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("key"), ""))
	} else {
		for _, msg := range validation.IsConfigMapKey(s.Key) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("key"), s.Key, msg))
		}
	}

	return allErrs
}

func validateSecretKeySelector(s *api.SecretKeySelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(s.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}
	if len(s.Key) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("key"), ""))
	} else {
		for _, msg := range validation.IsConfigMapKey(s.Key) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("key"), s.Key, msg))
		}
	}

	return allErrs
}

func validateVolumeMounts(mounts []api.VolumeMount, volumes sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	mountpoints := sets.NewString()

	for i, mnt := range mounts {
		idxPath := fldPath.Index(i)
		if len(mnt.Name) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("name"), ""))
		} else if !volumes.Has(mnt.Name) {
			allErrs = append(allErrs, field.NotFound(idxPath.Child("name"), mnt.Name))
		}
		if len(mnt.MountPath) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("mountPath"), ""))
		} else if strings.Contains(mnt.MountPath, ":") {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("mountPath"), mnt.MountPath, "must not contain ':'"))
		}
		if mountpoints.Has(mnt.MountPath) {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("mountPath"), mnt.MountPath, "must be unique"))
		}
		mountpoints.Insert(mnt.MountPath)
		if len(mnt.SubPath) > 0 {
			allErrs = append(allErrs, validateLocalDescendingPath(mnt.SubPath, fldPath.Child("subPath"))...)
		}
	}
	return allErrs
}

func validateProbe(probe *api.Probe, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if probe == nil {
		return allErrs
	}
	allErrs = append(allErrs, validateHandler(&probe.Handler, fldPath)...)

	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.InitialDelaySeconds), fldPath.Child("initialDelaySeconds"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.TimeoutSeconds), fldPath.Child("timeoutSeconds"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.PeriodSeconds), fldPath.Child("periodSeconds"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.SuccessThreshold), fldPath.Child("successThreshold"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(probe.FailureThreshold), fldPath.Child("failureThreshold"))...)
	return allErrs
}

// AccumulateUniqueHostPorts extracts each HostPort of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniqueHostPorts(containers []api.Container, accumulator *sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for ci, ctr := range containers {
		idxPath := fldPath.Index(ci)
		portsPath := idxPath.Child("ports")
		for pi := range ctr.Ports {
			idxPath := portsPath.Index(pi)
			port := ctr.Ports[pi].HostPort
			if port == 0 {
				continue
			}
			str := fmt.Sprintf("%d/%s", port, ctr.Ports[pi].Protocol)
			if accumulator.Has(str) {
				allErrs = append(allErrs, field.Duplicate(idxPath.Child("hostPort"), str))
			} else {
				accumulator.Insert(str)
			}
		}
	}
	return allErrs
}

// checkHostPortConflicts checks for colliding Port.HostPort values across
// a slice of containers.
func checkHostPortConflicts(containers []api.Container, fldPath *field.Path) field.ErrorList {
	allPorts := sets.String{}
	return AccumulateUniqueHostPorts(containers, &allPorts, fldPath)
}

func validateExecAction(exec *api.ExecAction, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if len(exec.Command) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("command"), ""))
	}
	return allErrors
}

var supportedHTTPSchemes = sets.NewString(string(api.URISchemeHTTP), string(api.URISchemeHTTPS))

func validateHTTPGetAction(http *api.HTTPGetAction, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if len(http.Path) == 0 {
		allErrors = append(allErrors, field.Required(fldPath.Child("path"), ""))
	}
	allErrors = append(allErrors, ValidatePortNumOrName(http.Port, fldPath.Child("port"))...)
	if !supportedHTTPSchemes.Has(string(http.Scheme)) {
		allErrors = append(allErrors, field.NotSupported(fldPath.Child("scheme"), http.Scheme, supportedHTTPSchemes.List()))
	}
	for _, header := range http.HTTPHeaders {
		for _, msg := range validation.IsHTTPHeaderName(header.Name) {
			allErrors = append(allErrors, field.Invalid(fldPath.Child("httpHeaders"), header.Name, msg))
		}
	}
	return allErrors
}

func ValidatePortNumOrName(port intstr.IntOrString, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if port.Type == intstr.Int {
		for _, msg := range validation.IsValidPortNum(port.IntValue()) {
			allErrs = append(allErrs, field.Invalid(fldPath, port.IntValue(), msg))
		}
	} else if port.Type == intstr.String {
		for _, msg := range validation.IsValidPortName(port.StrVal) {
			allErrs = append(allErrs, field.Invalid(fldPath, port.StrVal, msg))
		}
	} else {
		allErrs = append(allErrs, field.InternalError(fldPath, fmt.Errorf("unknown type: %v", port.Type)))
	}
	return allErrs
}

func validateTCPSocketAction(tcp *api.TCPSocketAction, fldPath *field.Path) field.ErrorList {
	return ValidatePortNumOrName(tcp.Port, fldPath.Child("port"))
}

func validateHandler(handler *api.Handler, fldPath *field.Path) field.ErrorList {
	numHandlers := 0
	allErrors := field.ErrorList{}
	if handler.Exec != nil {
		if numHandlers > 0 {
			allErrors = append(allErrors, field.Forbidden(fldPath.Child("exec"), "may not specify more than 1 handler type"))
		} else {
			numHandlers++
			allErrors = append(allErrors, validateExecAction(handler.Exec, fldPath.Child("exec"))...)
		}
	}
	if handler.HTTPGet != nil {
		if numHandlers > 0 {
			allErrors = append(allErrors, field.Forbidden(fldPath.Child("httpGet"), "may not specify more than 1 handler type"))
		} else {
			numHandlers++
			allErrors = append(allErrors, validateHTTPGetAction(handler.HTTPGet, fldPath.Child("httpGet"))...)
		}
	}
	if handler.TCPSocket != nil {
		if numHandlers > 0 {
			allErrors = append(allErrors, field.Forbidden(fldPath.Child("tcpSocket"), "may not specify more than 1 handler type"))
		} else {
			numHandlers++
			allErrors = append(allErrors, validateTCPSocketAction(handler.TCPSocket, fldPath.Child("tcpSocket"))...)
		}
	}
	if numHandlers == 0 {
		allErrors = append(allErrors, field.Required(fldPath, "must specify a handler type"))
	}
	return allErrors
}

func validateLifecycle(lifecycle *api.Lifecycle, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if lifecycle.PostStart != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PostStart, fldPath.Child("postStart"))...)
	}
	if lifecycle.PreStop != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PreStop, fldPath.Child("preStop"))...)
	}
	return allErrs
}

var supportedPullPolicies = sets.NewString(string(api.PullAlways), string(api.PullIfNotPresent), string(api.PullNever))

func validatePullPolicy(policy api.PullPolicy, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}

	switch policy {
	case api.PullAlways, api.PullIfNotPresent, api.PullNever:
		break
	case "":
		allErrors = append(allErrors, field.Required(fldPath, ""))
	default:
		allErrors = append(allErrors, field.NotSupported(fldPath, policy, supportedPullPolicies.List()))
	}

	return allErrors
}

func validateInitContainers(containers, otherContainers []api.Container, volumes sets.String, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(containers) > 0 {
		allErrs = append(allErrs, validateContainers(containers, volumes, fldPath)...)
	}

	allNames := sets.String{}
	for _, ctr := range otherContainers {
		allNames.Insert(ctr.Name)
	}
	for i, ctr := range containers {
		idxPath := fldPath.Index(i)
		if allNames.Has(ctr.Name) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), ctr.Name))
		}
		if len(ctr.Name) > 0 {
			allNames.Insert(ctr.Name)
		}
		if ctr.Lifecycle != nil {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("lifecycle"), ctr.Lifecycle, "must not be set for init containers"))
		}
		if ctr.LivenessProbe != nil {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("livenessProbe"), ctr.LivenessProbe, "must not be set for init containers"))
		}
		if ctr.ReadinessProbe != nil {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("readinessProbe"), ctr.ReadinessProbe, "must not be set for init containers"))
		}
	}
	return allErrs
}

func validateContainers(containers []api.Container, volumes sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(containers) == 0 {
		return append(allErrs, field.Required(fldPath, ""))
	}

	allNames := sets.String{}
	for i, ctr := range containers {
		idxPath := fldPath.Index(i)
		namePath := idxPath.Child("name")
		if len(ctr.Name) == 0 {
			allErrs = append(allErrs, field.Required(namePath, ""))
		} else {
			allErrs = append(allErrs, ValidateDNS1123Label(ctr.Name, namePath)...)
		}
		if allNames.Has(ctr.Name) {
			allErrs = append(allErrs, field.Duplicate(namePath, ctr.Name))
		} else {
			allNames.Insert(ctr.Name)
		}
		if len(ctr.Image) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("image"), ""))
		}
		if ctr.Lifecycle != nil {
			allErrs = append(allErrs, validateLifecycle(ctr.Lifecycle, idxPath.Child("lifecycle"))...)
		}
		allErrs = append(allErrs, validateProbe(ctr.LivenessProbe, idxPath.Child("livenessProbe"))...)
		// Liveness-specific validation
		if ctr.LivenessProbe != nil && ctr.LivenessProbe.SuccessThreshold != 1 {
			allErrs = append(allErrs, field.Invalid(idxPath.Child("livenessProbe", "successThreshold"), ctr.LivenessProbe.SuccessThreshold, "must be 1"))
		}

		allErrs = append(allErrs, validateProbe(ctr.ReadinessProbe, idxPath.Child("readinessProbe"))...)
		allErrs = append(allErrs, validateContainerPorts(ctr.Ports, idxPath.Child("ports"))...)
		allErrs = append(allErrs, validateEnv(ctr.Env, idxPath.Child("env"))...)
		allErrs = append(allErrs, validateVolumeMounts(ctr.VolumeMounts, volumes, idxPath.Child("volumeMounts"))...)
		allErrs = append(allErrs, validatePullPolicy(ctr.ImagePullPolicy, idxPath.Child("imagePullPolicy"))...)
		allErrs = append(allErrs, ValidateResourceRequirements(&ctr.Resources, idxPath.Child("resources"))...)
		allErrs = append(allErrs, ValidateSecurityContext(ctr.SecurityContext, idxPath.Child("securityContext"))...)
	}
	// Check for colliding ports across all containers.
	allErrs = append(allErrs, checkHostPortConflicts(containers, fldPath)...)

	return allErrs
}

func validateRestartPolicy(restartPolicy *api.RestartPolicy, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	switch *restartPolicy {
	case api.RestartPolicyAlways, api.RestartPolicyOnFailure, api.RestartPolicyNever:
		break
	case "":
		allErrors = append(allErrors, field.Required(fldPath, ""))
	default:
		validValues := []string{string(api.RestartPolicyAlways), string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}
		allErrors = append(allErrors, field.NotSupported(fldPath, *restartPolicy, validValues))
	}

	return allErrors
}

func validateDNSPolicy(dnsPolicy *api.DNSPolicy, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	switch *dnsPolicy {
	case api.DNSClusterFirst, api.DNSDefault:
		break
	case "":
		allErrors = append(allErrors, field.Required(fldPath, ""))
	default:
		validValues := []string{string(api.DNSClusterFirst), string(api.DNSDefault)}
		allErrors = append(allErrors, field.NotSupported(fldPath, dnsPolicy, validValues))
	}
	return allErrors
}

func validateHostNetwork(hostNetwork bool, containers []api.Container, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if hostNetwork {
		for i, container := range containers {
			portsPath := fldPath.Index(i).Child("ports")
			for i, port := range container.Ports {
				idxPath := portsPath.Index(i)
				if port.HostPort != port.ContainerPort {
					allErrors = append(allErrors, field.Invalid(idxPath.Child("containerPort"), port.ContainerPort, "must match `hostPort` when `hostNetwork` is true"))
				}
			}
		}
	}
	return allErrors
}

// validateImagePullSecrets checks to make sure the pull secrets are well
// formed.  Right now, we only expect name to be set (it's the only field).  If
// this ever changes and someone decides to set those fields, we'd like to
// know.
func validateImagePullSecrets(imagePullSecrets []api.LocalObjectReference, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	for i, currPullSecret := range imagePullSecrets {
		idxPath := fldPath.Index(i)
		strippedRef := api.LocalObjectReference{Name: currPullSecret.Name}
		if !reflect.DeepEqual(strippedRef, currPullSecret) {
			allErrors = append(allErrors, field.Invalid(idxPath, currPullSecret, "only name may be set"))
		}
	}
	return allErrors
}

func validateTaintEffect(effect *api.TaintEffect, allowEmpty bool, fldPath *field.Path) field.ErrorList {
	if !allowEmpty && len(*effect) == 0 {
		return field.ErrorList{field.Required(fldPath, "")}
	}

	allErrors := field.ErrorList{}
	switch *effect {
	// TODO: Replace next line with subsequent commented-out line when implement TaintEffectNoScheduleNoAdmit, TaintEffectNoScheduleNoAdmitNoExecute.
	case api.TaintEffectNoSchedule, api.TaintEffectPreferNoSchedule:
		// case api.TaintEffectNoSchedule, api.TaintEffectPreferNoSchedule, api.TaintEffectNoScheduleNoAdmit, api.TaintEffectNoScheduleNoAdmitNoExecute:
	default:
		validValues := []string{
			string(api.TaintEffectNoSchedule),
			string(api.TaintEffectPreferNoSchedule),
			// TODO: Uncomment this block when implement TaintEffectNoScheduleNoAdmit, TaintEffectNoScheduleNoAdmitNoExecute.
			// string(api.TaintEffectNoScheduleNoAdmit),
			// string(api.TaintEffectNoScheduleNoAdmitNoExecute),
		}
		allErrors = append(allErrors, field.NotSupported(fldPath, effect, validValues))
	}
	return allErrors
}

// validateTolerations tests if given tolerations have valid data.
func validateTolerations(tolerations []api.Toleration, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	for i, toleration := range tolerations {
		idxPath := fldPath.Index(i)
		// validate the toleration key
		allErrors = append(allErrors, unversionedvalidation.ValidateLabelName(toleration.Key, idxPath.Child("key"))...)

		// validate toleration operator and value
		switch toleration.Operator {
		case api.TolerationOpEqual, "":
			if errs := validation.IsValidLabelValue(toleration.Value); len(errs) != 0 {
				allErrors = append(allErrors, field.Invalid(idxPath.Child("operator"), toleration.Value, strings.Join(errs, ";")))
			}
		case api.TolerationOpExists:
			if len(toleration.Value) > 0 {
				allErrors = append(allErrors, field.Invalid(idxPath.Child("operator"), toleration, "value must be empty when `operator` is 'Exists'"))
			}
		default:
			validValues := []string{string(api.TolerationOpEqual), string(api.TolerationOpExists)}
			allErrors = append(allErrors, field.NotSupported(idxPath.Child("operator"), toleration.Operator, validValues))
		}

		// validate toleration effect
		if len(toleration.Effect) > 0 {
			allErrors = append(allErrors, validateTaintEffect(&toleration.Effect, true, idxPath.Child("effect"))...)
		}
	}
	return allErrors
}

// ValidatePod tests if required fields in the pod are set.
func ValidatePod(pod *api.Pod) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName, fldPath)
	allErrs = append(allErrs, ValidatePodSpecificAnnotations(pod.ObjectMeta.Annotations, fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidatePodSpec(&pod.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidatePodSpec tests that the specified PodSpec has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidatePodSpec(spec *api.PodSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allVolumes, vErrs := validateVolumes(spec.Volumes, fldPath.Child("volumes"))
	allErrs = append(allErrs, vErrs...)
	allErrs = append(allErrs, validateContainers(spec.Containers, allVolumes, fldPath.Child("containers"))...)
	allErrs = append(allErrs, validateInitContainers(spec.InitContainers, spec.Containers, allVolumes, fldPath.Child("initContainers"))...)
	allErrs = append(allErrs, validateRestartPolicy(&spec.RestartPolicy, fldPath.Child("restartPolicy"))...)
	allErrs = append(allErrs, validateDNSPolicy(&spec.DNSPolicy, fldPath.Child("dnsPolicy"))...)
	allErrs = append(allErrs, unversionedvalidation.ValidateLabels(spec.NodeSelector, fldPath.Child("nodeSelector"))...)
	allErrs = append(allErrs, ValidatePodSecurityContext(spec.SecurityContext, spec, fldPath, fldPath.Child("securityContext"))...)
	allErrs = append(allErrs, validateImagePullSecrets(spec.ImagePullSecrets, fldPath.Child("imagePullSecrets"))...)
	if len(spec.ServiceAccountName) > 0 {
		for _, msg := range ValidateServiceAccountName(spec.ServiceAccountName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("serviceAccountName"), spec.ServiceAccountName, msg))
		}
	}

	if len(spec.NodeName) > 0 {
		for _, msg := range ValidateNodeName(spec.NodeName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nodeName"), spec.NodeName, msg))
		}
	}

	if spec.ActiveDeadlineSeconds != nil {
		if *spec.ActiveDeadlineSeconds <= 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("activeDeadlineSeconds"), spec.ActiveDeadlineSeconds, "must be greater than 0"))
		}
	}

	if len(spec.Hostname) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(spec.Hostname, fldPath.Child("hostname"))...)
	}

	if len(spec.Subdomain) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(spec.Subdomain, fldPath.Child("subdomain"))...)
	}

	return allErrs
}

// ValidateNodeSelectorRequirement tests that the specified NodeSelectorRequirement fields has valid data
func ValidateNodeSelectorRequirement(rq api.NodeSelectorRequirement, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch rq.Operator {
	case api.NodeSelectorOpIn, api.NodeSelectorOpNotIn:
		if len(rq.Values) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Child("values"), "must be specified when `operator` is 'In' or 'NotIn'"))
		}
	case api.NodeSelectorOpExists, api.NodeSelectorOpDoesNotExist:
		if len(rq.Values) > 0 {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("values"), "may not be specified when `operator` is 'Exists' or 'DoesNotExist'"))
		}

	case api.NodeSelectorOpGt, api.NodeSelectorOpLt:
		if len(rq.Values) != 1 {
			allErrs = append(allErrs, field.Required(fldPath.Child("values"), "must be specified single value when `operator` is 'Lt' or 'Gt'"))
		}
	default:
		allErrs = append(allErrs, field.Invalid(fldPath.Child("operator"), rq.Operator, "not a valid selector operator"))
	}
	allErrs = append(allErrs, unversionedvalidation.ValidateLabelName(rq.Key, fldPath.Child("key"))...)
	return allErrs
}

// ValidateNodeSelectorTerm tests that the specified node selector term has valid data
func ValidateNodeSelectorTerm(term api.NodeSelectorTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(term.MatchExpressions) == 0 {
		return append(allErrs, field.Required(fldPath.Child("matchExpressions"), "must have at least one node selector requirement"))
	}
	for j, req := range term.MatchExpressions {
		allErrs = append(allErrs, ValidateNodeSelectorRequirement(req, fldPath.Child("matchExpressions").Index(j))...)
	}
	return allErrs
}

// ValidateNodeSelector tests that the specified nodeSelector fields has valid data
func ValidateNodeSelector(nodeSelector *api.NodeSelector, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	termFldPath := fldPath.Child("nodeSelectorTerms")
	if len(nodeSelector.NodeSelectorTerms) == 0 {
		return append(allErrs, field.Required(termFldPath, "must have at least one node selector term"))
	}

	for i, term := range nodeSelector.NodeSelectorTerms {
		allErrs = append(allErrs, ValidateNodeSelectorTerm(term, termFldPath.Index(i))...)
	}

	return allErrs
}

// ValidateAvoidPodsInNodeAnnotations tests that the serialized AvoidPods in Node.Annotations has valid data
func ValidateAvoidPodsInNodeAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	avoids, err := api.GetAvoidPodsFromNodeAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("AvoidPods"), api.PreferAvoidPodsAnnotationKey, err.Error()))
		return allErrs
	}

	if len(avoids.PreferAvoidPods) != 0 {
		for i, pa := range avoids.PreferAvoidPods {
			idxPath := fldPath.Child(api.PreferAvoidPodsAnnotationKey).Index(i)
			allErrs = append(allErrs, validatePreferAvoidPodsEntry(pa, idxPath)...)
		}
	}

	return allErrs
}

// validatePreferAvoidPodsEntry tests if given PreferAvoidPodsEntry has valid data.
func validatePreferAvoidPodsEntry(avoidPodEntry api.PreferAvoidPodsEntry, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	if avoidPodEntry.PodSignature.PodController == nil {
		allErrors = append(allErrors, field.Required(fldPath.Child("PodSignature"), ""))
	} else {
		if *(avoidPodEntry.PodSignature.PodController.Controller) != true {
			allErrors = append(allErrors,
				field.Invalid(fldPath.Child("PodSignature").Child("PodController").Child("Controller"),
					*(avoidPodEntry.PodSignature.PodController.Controller), "must point to a controller"))
		}
	}
	return allErrors
}

// ValidatePreferredSchedulingTerms tests that the specified SoftNodeAffinity fields has valid data
func ValidatePreferredSchedulingTerms(terms []api.PreferredSchedulingTerm, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i, term := range terms {
		if term.Weight <= 0 || term.Weight > 100 {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(i).Child("weight"), term.Weight, "must be in the range 1-100"))
		}

		allErrs = append(allErrs, ValidateNodeSelectorTerm(term.Preference, fldPath.Index(i).Child("preference"))...)
	}
	return allErrs
}

// validatePodAffinityTerm tests that the specified podAffinityTerm fields have valid data
func validatePodAffinityTerm(podAffinityTerm api.PodAffinityTerm, allowEmptyTopologyKey bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(podAffinityTerm.LabelSelector, fldPath.Child("matchExpressions"))...)
	for _, name := range podAffinityTerm.Namespaces {
		for _, msg := range ValidateNamespaceName(name, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("namespace"), name, msg))
		}
	}
	if !allowEmptyTopologyKey && len(podAffinityTerm.TopologyKey) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("topologyKey"), "can only be empty for PreferredDuringScheduling pod anti affinity"))
	}
	if len(podAffinityTerm.TopologyKey) != 0 {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelName(podAffinityTerm.TopologyKey, fldPath.Child("topologyKey"))...)
	}
	return allErrs
}

// validatePodAffinityTerms tests that the specified podAffinityTerms fields have valid data
func validatePodAffinityTerms(podAffinityTerms []api.PodAffinityTerm, allowEmptyTopologyKey bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, podAffinityTerm := range podAffinityTerms {
		allErrs = append(allErrs, validatePodAffinityTerm(podAffinityTerm, allowEmptyTopologyKey, fldPath.Index(i))...)
	}
	return allErrs
}

// validateWeightedPodAffinityTerms tests that the specified weightedPodAffinityTerms fields have valid data
func validateWeightedPodAffinityTerms(weightedPodAffinityTerms []api.WeightedPodAffinityTerm, allowEmptyTopologyKey bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for j, weightedTerm := range weightedPodAffinityTerms {
		if weightedTerm.Weight <= 0 || weightedTerm.Weight > 100 {
			allErrs = append(allErrs, field.Invalid(fldPath.Index(j).Child("weight"), weightedTerm.Weight, "must be in the range 1-100"))
		}
		allErrs = append(allErrs, validatePodAffinityTerm(weightedTerm.PodAffinityTerm, allowEmptyTopologyKey, fldPath.Index(j).Child("podAffinityTerm"))...)
	}
	return allErrs
}

// validatePodAntiAffinity tests that the specified podAntiAffinity fields have valid data
func validatePodAntiAffinity(podAntiAffinity *api.PodAntiAffinity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO:Uncomment below code once RequiredDuringSchedulingRequiredDuringExecution is implemented.
	// if podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution != nil {
	//	allErrs = append(allErrs, validatePodAffinityTerms(podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution, false,
	//		fldPath.Child("requiredDuringSchedulingRequiredDuringExecution"))...)
	//}
	if podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		// empty topologyKey is not allowed for hard pod anti-affinity
		allErrs = append(allErrs, validatePodAffinityTerms(podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution, false,
			fldPath.Child("requiredDuringSchedulingIgnoredDuringExecution"))...)
	}
	if podAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		// empty topologyKey is allowed for soft pod anti-affinity
		allErrs = append(allErrs, validateWeightedPodAffinityTerms(podAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution, true,
			fldPath.Child("preferredDuringSchedulingIgnoredDuringExecution"))...)
	}
	return allErrs
}

// validatePodAffinity tests that the specified podAffinity fields have valid data
func validatePodAffinity(podAffinity *api.PodAffinity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// TODO:Uncomment below code once RequiredDuringSchedulingRequiredDuringExecution is implemented.
	// if podAffinity.RequiredDuringSchedulingRequiredDuringExecution != nil {
	//	allErrs = append(allErrs, validatePodAffinityTerms(podAffinity.RequiredDuringSchedulingRequiredDuringExecution, false,
	//		fldPath.Child("requiredDuringSchedulingRequiredDuringExecution"))...)
	//}
	if podAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		// empty topologyKey is not allowed for hard pod affinity
		allErrs = append(allErrs, validatePodAffinityTerms(podAffinity.RequiredDuringSchedulingIgnoredDuringExecution, false,
			fldPath.Child("requiredDuringSchedulingIgnoredDuringExecution"))...)
	}
	if podAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		// empty topologyKey is not allowed for soft pod affinity
		allErrs = append(allErrs, validateWeightedPodAffinityTerms(podAffinity.PreferredDuringSchedulingIgnoredDuringExecution, false,
			fldPath.Child("preferredDuringSchedulingIgnoredDuringExecution"))...)
	}
	return allErrs
}

// ValidateAffinityInPodAnnotations tests that the serialized Affinity in Pod.Annotations has valid data
func ValidateAffinityInPodAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	affinity, err := api.GetAffinityFromPodAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, api.AffinityAnnotationKey, err.Error()))
		return allErrs
	}
	if affinity == nil {
		return allErrs
	}

	affinityFldPath := fldPath.Child(api.AffinityAnnotationKey)
	if affinity.NodeAffinity != nil {
		na := affinity.NodeAffinity
		naFldPath := affinityFldPath.Child("nodeAffinity")
		// TODO: Uncomment the next three lines once RequiredDuringSchedulingRequiredDuringExecution is implemented.
		// if na.RequiredDuringSchedulingRequiredDuringExecution != nil {
		//	allErrs = append(allErrs, ValidateNodeSelector(na.RequiredDuringSchedulingRequiredDuringExecution, naFldPath.Child("requiredDuringSchedulingRequiredDuringExecution"))...)
		// }

		if na.RequiredDuringSchedulingIgnoredDuringExecution != nil {
			allErrs = append(allErrs, ValidateNodeSelector(na.RequiredDuringSchedulingIgnoredDuringExecution, naFldPath.Child("requiredDuringSchedulingIgnoredDuringExecution"))...)
		}

		if len(na.PreferredDuringSchedulingIgnoredDuringExecution) > 0 {
			allErrs = append(allErrs, ValidatePreferredSchedulingTerms(na.PreferredDuringSchedulingIgnoredDuringExecution, naFldPath.Child("preferredDuringSchedulingIgnoredDuringExecution"))...)
		}
	}
	if affinity.PodAffinity != nil {
		allErrs = append(allErrs, validatePodAffinity(affinity.PodAffinity, affinityFldPath.Child("podAffinity"))...)
	}
	if affinity.PodAntiAffinity != nil {
		allErrs = append(allErrs, validatePodAntiAffinity(affinity.PodAntiAffinity, affinityFldPath.Child("podAntiAffinity"))...)
	}

	return allErrs
}

// ValidateTolerationsInPodAnnotations tests that the serialized tolerations in Pod.Annotations has valid data
func ValidateTolerationsInPodAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	tolerations, err := api.GetTolerationsFromPodAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, api.TolerationsAnnotationKey, err.Error()))
		return allErrs
	}
	if len(tolerations) > 0 {
		allErrs = append(allErrs, validateTolerations(tolerations, fldPath.Child(api.TolerationsAnnotationKey))...)
	}

	return allErrs
}

func validateSeccompProfile(p string, fldPath *field.Path) field.ErrorList {
	if p == "docker/default" {
		return nil
	}
	if p == "unconfined" {
		return nil
	}
	if strings.HasPrefix(p, "localhost/") {
		return validateLocalDescendingPath(strings.TrimPrefix(p, "localhost/"), fldPath)
	}
	return field.ErrorList{field.Invalid(fldPath, p, "must be a valid seccomp profile")}
}

func ValidateSeccompPodAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if p, exists := annotations[api.SeccompPodAnnotationKey]; exists {
		allErrs = append(allErrs, validateSeccompProfile(p, fldPath.Child(api.SeccompPodAnnotationKey))...)
	}
	for k, p := range annotations {
		if strings.HasPrefix(k, api.SeccompContainerAnnotationKeyPrefix) {
			allErrs = append(allErrs, validateSeccompProfile(p, fldPath.Child(k))...)
		}
	}

	return allErrs
}

// ValidatePodSecurityContext test that the specified PodSecurityContext has valid data.
func ValidatePodSecurityContext(securityContext *api.PodSecurityContext, spec *api.PodSpec, specPath, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if securityContext != nil {
		allErrs = append(allErrs, validateHostNetwork(securityContext.HostNetwork, spec.Containers, specPath.Child("containers"))...)
		if securityContext.FSGroup != nil {
			for _, msg := range validation.IsValidGroupId(*securityContext.FSGroup) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("fsGroup"), *(securityContext.FSGroup), msg))
			}
		}
		if securityContext.RunAsUser != nil {
			for _, msg := range validation.IsValidUserId(*securityContext.RunAsUser) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsUser"), *(securityContext.RunAsUser), msg))
			}
		}
		for g, gid := range securityContext.SupplementalGroups {
			for _, msg := range validation.IsValidGroupId(gid) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("supplementalGroups").Index(g), gid, msg))
			}
		}
	}

	return allErrs
}

func validateContainerUpdates(newContainers, oldContainers []api.Container, fldPath *field.Path) (allErrs field.ErrorList, stop bool) {
	allErrs = field.ErrorList{}
	if len(newContainers) != len(oldContainers) {
		//TODO: Pinpoint the specific container that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, field.Forbidden(fldPath, "pod updates may not add or remove containers"))
		return allErrs, true
	}

	// validate updated container images
	for i, ctr := range newContainers {
		if len(ctr.Image) == 0 {
			allErrs = append(allErrs, field.Required(fldPath.Index(i).Child("image"), ""))
		}
	}
	return allErrs, false
}

// ValidatePodUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodUpdate(newPod, oldPod *api.Pod) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMetaUpdate(&newPod.ObjectMeta, &oldPod.ObjectMeta, fldPath)
	allErrs = append(allErrs, ValidatePodSpecificAnnotations(newPod.ObjectMeta.Annotations, fldPath.Child("annotations"))...)
	specPath := field.NewPath("spec")

	// validate updateable fields:
	// 1.  containers[*].image
	// 2.  initContainers[*].image
	// 3.  spec.activeDeadlineSeconds

	containerErrs, stop := validateContainerUpdates(newPod.Spec.Containers, oldPod.Spec.Containers, specPath.Child("containers"))
	allErrs = append(allErrs, containerErrs...)
	if stop {
		return allErrs
	}
	containerErrs, stop = validateContainerUpdates(newPod.Spec.InitContainers, oldPod.Spec.InitContainers, specPath.Child("initContainers"))
	allErrs = append(allErrs, containerErrs...)
	if stop {
		return allErrs
	}

	// validate updated spec.activeDeadlineSeconds.  two types of updates are allowed:
	// 1.  from nil to a positive value
	// 2.  from a positive value to a lesser, non-negative value
	if newPod.Spec.ActiveDeadlineSeconds != nil {
		newActiveDeadlineSeconds := *newPod.Spec.ActiveDeadlineSeconds
		if newActiveDeadlineSeconds < 0 {
			allErrs = append(allErrs, field.Invalid(specPath.Child("activeDeadlineSeconds"), newActiveDeadlineSeconds, isNegativeErrorMsg))
			return allErrs
		}
		if oldPod.Spec.ActiveDeadlineSeconds != nil {
			oldActiveDeadlineSeconds := *oldPod.Spec.ActiveDeadlineSeconds
			if oldActiveDeadlineSeconds < newActiveDeadlineSeconds {
				allErrs = append(allErrs, field.Invalid(specPath.Child("activeDeadlineSeconds"), newActiveDeadlineSeconds, "must be less than or equal to previous value"))
				return allErrs
			}
		}
	} else if oldPod.Spec.ActiveDeadlineSeconds != nil {
		allErrs = append(allErrs, field.Invalid(specPath.Child("activeDeadlineSeconds"), newPod.Spec.ActiveDeadlineSeconds, "must not update from a positive integer to nil value"))
	}

	// handle updateable fields by munging those fields prior to deep equal comparison.
	mungedPod := *newPod
	// munge containers[*].image
	var newContainers []api.Container
	for ix, container := range mungedPod.Spec.Containers {
		container.Image = oldPod.Spec.Containers[ix].Image
		newContainers = append(newContainers, container)
	}
	mungedPod.Spec.Containers = newContainers
	// munge initContainers[*].image
	var newInitContainers []api.Container
	for ix, container := range mungedPod.Spec.InitContainers {
		container.Image = oldPod.Spec.InitContainers[ix].Image
		newInitContainers = append(newInitContainers, container)
	}
	mungedPod.Spec.InitContainers = newInitContainers
	// munge spec.activeDeadlineSeconds
	mungedPod.Spec.ActiveDeadlineSeconds = nil
	if oldPod.Spec.ActiveDeadlineSeconds != nil {
		activeDeadlineSeconds := *oldPod.Spec.ActiveDeadlineSeconds
		mungedPod.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	}
	if !api.Semantic.DeepEqual(mungedPod.Spec, oldPod.Spec) {
		//TODO: Pinpoint the specific field that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, field.Forbidden(specPath, "pod updates may not change fields other than `containers[*].image` or `spec.activeDeadlineSeconds`"))
	}

	return allErrs
}

// ValidatePodStatusUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodStatusUpdate(newPod, oldPod *api.Pod) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newPod.ObjectMeta, &oldPod.ObjectMeta, field.NewPath("metadata"))

	// TODO: allow change when bindings are properly decoupled from pods
	if newPod.Spec.NodeName != oldPod.Spec.NodeName {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("status", "nodeName"), "may not be changed directly"))
	}

	// For status update we ignore changes to pod spec.
	newPod.Spec = oldPod.Spec

	return allErrs
}

// ValidatePodBinding tests if required fields in the pod binding are legal.
func ValidatePodBinding(binding *api.Binding) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(binding.Target.Kind) != 0 && binding.Target.Kind != "Node" {
		// TODO: When validation becomes versioned, this gets more complicated.
		allErrs = append(allErrs, field.NotSupported(field.NewPath("target", "kind"), binding.Target.Kind, []string{"Node", "<empty>"}))
	}
	if len(binding.Target.Name) == 0 {
		// TODO: When validation becomes versioned, this gets more complicated.
		allErrs = append(allErrs, field.Required(field.NewPath("target", "name"), ""))
	}

	return allErrs
}

// ValidatePodTemplate tests if required fields in the pod template are set.
func ValidatePodTemplate(pod *api.PodTemplate) field.ErrorList {
	allErrs := ValidateObjectMeta(&pod.ObjectMeta, true, ValidatePodName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePodTemplateSpec(&pod.Template, field.NewPath("template"))...)
	return allErrs
}

// ValidatePodTemplateUpdate tests to see if the update is legal for an end user to make. newPod is updated with fields
// that cannot be changed.
func ValidatePodTemplateUpdate(newPod, oldPod *api.PodTemplate) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&oldPod.ObjectMeta, &newPod.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePodTemplateSpec(&newPod.Template, field.NewPath("template"))...)
	return allErrs
}

var supportedSessionAffinityType = sets.NewString(string(api.ServiceAffinityClientIP), string(api.ServiceAffinityNone))
var supportedServiceType = sets.NewString(string(api.ServiceTypeClusterIP), string(api.ServiceTypeNodePort),
	string(api.ServiceTypeLoadBalancer))

// ValidateService tests if required fields in the service are set.
func ValidateService(service *api.Service) field.ErrorList {
	allErrs := ValidateObjectMeta(&service.ObjectMeta, true, ValidateServiceName, field.NewPath("metadata"))

	specPath := field.NewPath("spec")
	if len(service.Spec.Ports) == 0 && service.Spec.ClusterIP != api.ClusterIPNone {
		allErrs = append(allErrs, field.Required(specPath.Child("ports"), ""))
	}
	if service.Spec.Type == api.ServiceTypeLoadBalancer {
		for ix := range service.Spec.Ports {
			port := &service.Spec.Ports[ix]
			// This is a workaround for broken cloud environments that
			// over-open firewalls.  Hopefully it can go away when more clouds
			// understand containers better.
			if port.Port == 10250 {
				portPath := specPath.Child("ports").Index(ix)
				allErrs = append(allErrs, field.Invalid(portPath, port.Port, "may not expose port 10250 externally since it is used by kubelet"))
			}
		}
	}

	isHeadlessService := service.Spec.ClusterIP == api.ClusterIPNone
	allPortNames := sets.String{}
	portsPath := specPath.Child("ports")
	for i := range service.Spec.Ports {
		portPath := portsPath.Index(i)
		allErrs = append(allErrs, validateServicePort(&service.Spec.Ports[i], len(service.Spec.Ports) > 1, isHeadlessService, &allPortNames, portPath)...)
	}

	if service.Spec.Selector != nil {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabels(service.Spec.Selector, specPath.Child("selector"))...)
	}

	if len(service.Spec.SessionAffinity) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("sessionAffinity"), ""))
	} else if !supportedSessionAffinityType.Has(string(service.Spec.SessionAffinity)) {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("sessionAffinity"), service.Spec.SessionAffinity, supportedSessionAffinityType.List()))
	}

	if api.IsServiceIPSet(service) {
		if ip := net.ParseIP(service.Spec.ClusterIP); ip == nil {
			allErrs = append(allErrs, field.Invalid(specPath.Child("clusterIP"), service.Spec.ClusterIP, "must be empty, 'None', or a valid IP address"))
		}
	}

	ipPath := specPath.Child("externalIPs")
	for i, ip := range service.Spec.ExternalIPs {
		idxPath := ipPath.Index(i)
		if msgs := validation.IsValidIP(ip); len(msgs) != 0 {
			for i := range msgs {
				allErrs = append(allErrs, field.Invalid(idxPath, ip, msgs[i]))
			}
		} else {
			allErrs = append(allErrs, validateNonSpecialIP(ip, idxPath)...)
		}
	}

	if len(service.Spec.Type) == 0 {
		allErrs = append(allErrs, field.Required(specPath.Child("type"), ""))
	} else if !supportedServiceType.Has(string(service.Spec.Type)) {
		allErrs = append(allErrs, field.NotSupported(specPath.Child("type"), service.Spec.Type, supportedServiceType.List()))
	}

	if service.Spec.Type == api.ServiceTypeLoadBalancer {
		portsPath := specPath.Child("ports")
		includeProtocols := sets.NewString()
		for i := range service.Spec.Ports {
			portPath := portsPath.Index(i)
			if !supportedPortProtocols.Has(string(service.Spec.Ports[i].Protocol)) {
				allErrs = append(allErrs, field.Invalid(portPath.Child("protocol"), service.Spec.Ports[i].Protocol, "cannot create an external load balancer with non-TCP/UDP ports"))
			} else {
				includeProtocols.Insert(string(service.Spec.Ports[i].Protocol))
			}
		}
		if includeProtocols.Len() > 1 {
			allErrs = append(allErrs, field.Invalid(portsPath, service.Spec.Ports, "cannot create an external load balancer with mix protocols"))
		}
	}

	if service.Spec.Type == api.ServiceTypeClusterIP {
		portsPath := specPath.Child("ports")
		for i := range service.Spec.Ports {
			portPath := portsPath.Index(i)
			if service.Spec.Ports[i].NodePort != 0 {
				allErrs = append(allErrs, field.Invalid(portPath.Child("nodePort"), service.Spec.Ports[i].NodePort, "may not be used when `type` is 'ClusterIP'"))
			}
		}
	}

	// Check for duplicate NodePorts, considering (protocol,port) pairs
	portsPath = specPath.Child("ports")
	nodePorts := make(map[api.ServicePort]bool)
	for i := range service.Spec.Ports {
		port := &service.Spec.Ports[i]
		if port.NodePort == 0 {
			continue
		}
		portPath := portsPath.Index(i)
		var key api.ServicePort
		key.Protocol = port.Protocol
		key.NodePort = port.NodePort
		_, found := nodePorts[key]
		if found {
			allErrs = append(allErrs, field.Duplicate(portPath.Child("nodePort"), port.NodePort))
		}
		nodePorts[key] = true
	}

	// Validate SourceRange field and annotation
	_, ok := service.Annotations[apiservice.AnnotationLoadBalancerSourceRangesKey]
	if len(service.Spec.LoadBalancerSourceRanges) > 0 || ok {
		var fieldPath *field.Path
		var val string
		if len(service.Spec.LoadBalancerSourceRanges) > 0 {
			fieldPath = specPath.Child("LoadBalancerSourceRanges")
			val = fmt.Sprintf("%v", service.Spec.LoadBalancerSourceRanges)
		} else {
			fieldPath = field.NewPath("metadata", "annotations").Key(apiservice.AnnotationLoadBalancerSourceRangesKey)
			val = service.Annotations[apiservice.AnnotationLoadBalancerSourceRangesKey]
		}
		if service.Spec.Type != api.ServiceTypeLoadBalancer {
			allErrs = append(allErrs, field.Invalid(fieldPath, "", "may only be used when `type` is 'LoadBalancer'"))
		}
		_, err := apiservice.GetLoadBalancerSourceRanges(service)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fieldPath, val, "must be a list of IP ranges. For example, 10.240.0.0/24,10.250.0.0/24 "))
		}
	}
	return allErrs
}

func validateServicePort(sp *api.ServicePort, requireName, isHeadlessService bool, allNames *sets.String, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if requireName && len(sp.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else if len(sp.Name) != 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(sp.Name, fldPath.Child("name"))...)
		if allNames.Has(sp.Name) {
			allErrs = append(allErrs, field.Duplicate(fldPath.Child("name"), sp.Name))
		} else {
			allNames.Insert(sp.Name)
		}
	}

	for _, msg := range validation.IsValidPortNum(int(sp.Port)) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("port"), sp.Port, msg))
	}

	if len(sp.Protocol) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("protocol"), ""))
	} else if !supportedPortProtocols.Has(string(sp.Protocol)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("protocol"), sp.Protocol, supportedPortProtocols.List()))
	}

	allErrs = append(allErrs, ValidatePortNumOrName(sp.TargetPort, fldPath.Child("targetPort"))...)

	// in the v1 API, targetPorts on headless services were tolerated.
	// once we have version-specific validation, we can reject this on newer API versions, but until then, we have to tolerate it for compatibility.
	//
	// if isHeadlessService {
	// 	if sp.TargetPort.Type == intstr.String || (sp.TargetPort.Type == intstr.Int && sp.Port != sp.TargetPort.IntValue()) {
	// 		allErrs = append(allErrs, field.Invalid(fldPath.Child("targetPort"), sp.TargetPort, "must be equal to the value of 'port' when clusterIP = None"))
	// 	}
	// }

	return allErrs
}

// ValidateServiceUpdate tests if required fields in the service are set during an update
func ValidateServiceUpdate(service, oldService *api.Service) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&service.ObjectMeta, &oldService.ObjectMeta, field.NewPath("metadata"))

	if api.IsServiceIPSet(oldService) {
		allErrs = append(allErrs, ValidateImmutableField(service.Spec.ClusterIP, oldService.Spec.ClusterIP, field.NewPath("spec", "clusterIP"))...)
	}

	allErrs = append(allErrs, ValidateService(service)...)
	return allErrs
}

// ValidateServiceStatusUpdate tests if required fields in the Service are set when updating status.
func ValidateServiceStatusUpdate(service, oldService *api.Service) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&service.ObjectMeta, &oldService.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLoadBalancerStatus(&service.Status.LoadBalancer, field.NewPath("status", "loadBalancer"))...)
	return allErrs
}

// ValidateReplicationController tests if required fields in the replication controller are set.
func ValidateReplicationController(controller *api.ReplicationController) field.ErrorList {
	allErrs := ValidateObjectMeta(&controller.ObjectMeta, true, ValidateReplicationControllerName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateReplicationControllerUpdate tests if required fields in the replication controller are set.
func ValidateReplicationControllerUpdate(controller, oldController *api.ReplicationController) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateReplicationControllerStatusUpdate tests if required fields in the replication controller are set.
func ValidateReplicationControllerStatusUpdate(controller, oldController *api.ReplicationController) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta, field.NewPath("metadata"))
	statusPath := field.NewPath("status")
	allErrs = append(allErrs, ValidateNonnegativeField(int64(controller.Status.Replicas), statusPath.Child("replicas"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(controller.Status.FullyLabeledReplicas), statusPath.Child("fullyLabeledReplicas"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(controller.Status.ObservedGeneration), statusPath.Child("observedGeneration"))...)
	return allErrs
}

// Validates that the given selector is non-empty.
func ValidateNonEmptySelector(selectorMap map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	selector := labels.Set(selectorMap).AsSelector()
	if selector.Empty() {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	}
	return allErrs
}

// Validates the given template and ensures that it is in accordance with the desired selector and replicas.
func ValidatePodTemplateSpecForRC(template *api.PodTemplateSpec, selectorMap map[string]string, replicas int32, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if template == nil {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else {
		selector := labels.Set(selectorMap).AsSelector()
		if !selector.Empty() {
			// Verify that the RC selector matches the labels in template.
			labels := labels.Set(template.Labels)
			if !selector.Matches(labels) {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("metadata", "labels"), template.Labels, "`selector` does not match template `labels`"))
			}
		}
		allErrs = append(allErrs, ValidatePodTemplateSpec(template, fldPath)...)
		if replicas > 1 {
			allErrs = append(allErrs, ValidateReadOnlyPersistentDisks(template.Spec.Volumes, fldPath.Child("spec", "volumes"))...)
		}
		// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
		if template.Spec.RestartPolicy != api.RestartPolicyAlways {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("spec", "restartPolicy"), template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
		}
	}
	return allErrs
}

// ValidateReplicationControllerSpec tests if required fields in the replication controller spec are set.
func ValidateReplicationControllerSpec(spec *api.ReplicationControllerSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNonEmptySelector(spec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, ValidateNonnegativeField(int64(spec.Replicas), fldPath.Child("replicas"))...)
	allErrs = append(allErrs, ValidatePodTemplateSpecForRC(spec.Template, spec.Selector, spec.Replicas, fldPath.Child("template"))...)
	return allErrs
}

// ValidatePodTemplateSpec validates the spec of a pod template
func ValidatePodTemplateSpec(spec *api.PodTemplateSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, unversionedvalidation.ValidateLabels(spec.Labels, fldPath.Child("labels"))...)
	allErrs = append(allErrs, ValidateAnnotations(spec.Annotations, fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidatePodSpecificAnnotations(spec.Annotations, fldPath.Child("annotations"))...)
	allErrs = append(allErrs, ValidatePodSpec(&spec.Spec, fldPath.Child("spec"))...)
	return allErrs
}

func ValidateReadOnlyPersistentDisks(volumes []api.Volume, fldPath *field.Path) field.ErrorList {
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

// validateTaints tests if given taints have valid data.
func validateTaints(taints []api.Taint, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	for i, currTaint := range taints {
		idxPath := fldPath.Index(i)
		// validate the taint key
		allErrors = append(allErrors, unversionedvalidation.ValidateLabelName(currTaint.Key, idxPath.Child("key"))...)
		// validate the taint value
		if errs := validation.IsValidLabelValue(currTaint.Value); len(errs) != 0 {
			allErrors = append(allErrors, field.Invalid(idxPath.Child("value"), currTaint.Value, strings.Join(errs, ";")))
		}
		// validate the taint effect
		allErrors = append(allErrors, validateTaintEffect(&currTaint.Effect, false, idxPath.Child("effect"))...)
	}
	return allErrors
}

// ValidateTaintsInNodeAnnotations tests that the serialized taints in Node.Annotations has valid data
func ValidateTaintsInNodeAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	taints, err := api.GetTaintsFromNodeAnnotations(annotations)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, api.TaintsAnnotationKey, err.Error()))
		return allErrs
	}
	if len(taints) > 0 {
		allErrs = append(allErrs, validateTaints(taints, fldPath.Child(api.TaintsAnnotationKey))...)
	}

	return allErrs
}

func ValidateNodeSpecificAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if annotations[api.PreferAvoidPodsAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateAvoidPodsInNodeAnnotations(annotations, fldPath)...)
	}
	if annotations[api.TaintsAnnotationKey] != "" {
		allErrs = append(allErrs, ValidateTaintsInNodeAnnotations(annotations, fldPath)...)
	}
	return allErrs
}

// ValidateNode tests if required fields in the node are set.
func ValidateNode(node *api.Node) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMeta(&node.ObjectMeta, false, ValidateNodeName, fldPath)
	allErrs = append(allErrs, ValidateNodeSpecificAnnotations(node.ObjectMeta.Annotations, fldPath.Child("annotations"))...)

	// Only validate spec. All status fields are optional and can be updated later.

	// external ID is required.
	if len(node.Spec.ExternalID) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "externalID"), ""))
	}

	// TODO(rjnagal): Ignore PodCIDR till its completely implemented.
	return allErrs
}

// ValidateNodeUpdate tests to make sure a node update can be applied.  Modifies oldNode.
func ValidateNodeUpdate(node, oldNode *api.Node) field.ErrorList {
	fldPath := field.NewPath("metadata")
	allErrs := ValidateObjectMetaUpdate(&node.ObjectMeta, &oldNode.ObjectMeta, fldPath)
	allErrs = append(allErrs, ValidateNodeSpecificAnnotations(node.ObjectMeta.Annotations, fldPath.Child("annotations"))...)

	// TODO: Enable the code once we have better api object.status update model. Currently,
	// anyone can update node status.
	// if !api.Semantic.DeepEqual(node.Status, api.NodeStatus{}) {
	// 	allErrs = append(allErrs, field.Invalid("status", node.Status, "must be empty"))
	// }

	// Validte no duplicate addresses in node status.
	addresses := make(map[api.NodeAddress]bool)
	for i, address := range node.Status.Addresses {
		if _, ok := addresses[address]; ok {
			allErrs = append(allErrs, field.Duplicate(field.NewPath("status", "addresses").Index(i), address))
		}
		addresses[address] = true
	}

	if len(oldNode.Spec.PodCIDR) == 0 {
		// Allow the controller manager to assign a CIDR to a node if it doesn't have one.
		oldNode.Spec.PodCIDR = node.Spec.PodCIDR
	} else {
		if oldNode.Spec.PodCIDR != node.Spec.PodCIDR {
			allErrs = append(allErrs, field.Forbidden(field.NewPath("spec", "podCIDR"), "node updates may not change podCIDR except from \"\" to valid"))
		}
	}
	// TODO: move reset function to its own location
	// Ignore metadata changes now that they have been tested
	oldNode.ObjectMeta = node.ObjectMeta
	// Allow users to update capacity
	oldNode.Status.Capacity = node.Status.Capacity
	// Allow users to unschedule node
	oldNode.Spec.Unschedulable = node.Spec.Unschedulable
	// Clear status
	oldNode.Status = node.Status

	// TODO: Add a 'real' error type for this error and provide print actual diffs.
	if !api.Semantic.DeepEqual(oldNode, node) {
		glog.V(4).Infof("Update failed validation %#v vs %#v", oldNode, node)
		allErrs = append(allErrs, field.Forbidden(field.NewPath(""), "node updates may only change labels or capacity"))
	}

	return allErrs
}

// Validate compute resource typename.
// Refer to docs/design/resources.md for more details.
func validateResourceName(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsQualifiedName(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	if len(allErrs) != 0 {
		return allErrs
	}

	if len(strings.Split(value, "/")) == 1 {
		if !api.IsStandardResourceName(value) {
			return append(allErrs, field.Invalid(fldPath, value, "must be a standard resource type or fully qualified"))
		}
	}

	return field.ErrorList{}
}

// Validate container resource name
// Refer to docs/design/resources.md for more details.
func validateContainerResourceName(value string, fldPath *field.Path) field.ErrorList {
	allErrs := validateResourceName(value, fldPath)
	if len(strings.Split(value, "/")) == 1 {
		if !api.IsStandardContainerResourceName(value) {
			return append(allErrs, field.Invalid(fldPath, value, "must be a standard resource for containers"))
		}
	}
	return field.ErrorList{}
}

// Validate resource names that can go in a resource quota
// Refer to docs/design/resources.md for more details.
func ValidateResourceQuotaResourceName(value string, fldPath *field.Path) field.ErrorList {
	allErrs := validateResourceName(value, fldPath)
	if len(strings.Split(value, "/")) == 1 {
		if !api.IsStandardQuotaResourceName(value) {
			return append(allErrs, field.Invalid(fldPath, value, isInvalidQuotaResource))
		}
	}
	return field.ErrorList{}
}

// Validate limit range types
func validateLimitRangeTypeName(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsQualifiedName(value) {
		allErrs = append(allErrs, field.Invalid(fldPath, value, msg))
	}
	if len(allErrs) != 0 {
		return allErrs
	}

	if len(strings.Split(value, "/")) == 1 {
		if !api.IsStandardLimitRangeType(value) {
			return append(allErrs, field.Invalid(fldPath, value, "must be a standard limit type or fully qualified"))
		}
	}

	return allErrs
}

// Validate limit range resource name
// limit types (other than Pod/Container) could contain storage not just cpu or memory
func validateLimitRangeResourceName(limitType api.LimitType, value string, fldPath *field.Path) field.ErrorList {
	switch limitType {
	case api.LimitTypePod, api.LimitTypeContainer:
		return validateContainerResourceName(value, fldPath)
	default:
		return validateResourceName(value, fldPath)
	}
}

// ValidateLimitRange tests if required fields in the LimitRange are set.
func ValidateLimitRange(limitRange *api.LimitRange) field.ErrorList {
	allErrs := ValidateObjectMeta(&limitRange.ObjectMeta, true, ValidateLimitRangeName, field.NewPath("metadata"))

	// ensure resource names are properly qualified per docs/design/resources.md
	limitTypeSet := map[api.LimitType]bool{}
	fldPath := field.NewPath("spec", "limits")
	for i := range limitRange.Spec.Limits {
		idxPath := fldPath.Index(i)
		limit := &limitRange.Spec.Limits[i]
		allErrs = append(allErrs, validateLimitRangeTypeName(string(limit.Type), idxPath.Child("type"))...)

		_, found := limitTypeSet[limit.Type]
		if found {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("type"), limit.Type))
		}
		limitTypeSet[limit.Type] = true

		keys := sets.String{}
		min := map[string]resource.Quantity{}
		max := map[string]resource.Quantity{}
		defaults := map[string]resource.Quantity{}
		defaultRequests := map[string]resource.Quantity{}
		maxLimitRequestRatios := map[string]resource.Quantity{}

		for k, q := range limit.Max {
			allErrs = append(allErrs, validateLimitRangeResourceName(limit.Type, string(k), idxPath.Child("max").Key(string(k)))...)
			keys.Insert(string(k))
			max[string(k)] = q
		}
		for k, q := range limit.Min {
			allErrs = append(allErrs, validateLimitRangeResourceName(limit.Type, string(k), idxPath.Child("min").Key(string(k)))...)
			keys.Insert(string(k))
			min[string(k)] = q
		}

		if limit.Type == api.LimitTypePod {
			if len(limit.Default) > 0 {
				allErrs = append(allErrs, field.Forbidden(idxPath.Child("default"), "may not be specified when `type` is 'Pod'"))
			}
			if len(limit.DefaultRequest) > 0 {
				allErrs = append(allErrs, field.Forbidden(idxPath.Child("defaultRequest"), "may not be specified when `type` is 'Pod'"))
			}
		} else {
			for k, q := range limit.Default {
				allErrs = append(allErrs, validateLimitRangeResourceName(limit.Type, string(k), idxPath.Child("default").Key(string(k)))...)
				keys.Insert(string(k))
				defaults[string(k)] = q
			}
			for k, q := range limit.DefaultRequest {
				allErrs = append(allErrs, validateLimitRangeResourceName(limit.Type, string(k), idxPath.Child("defaultRequest").Key(string(k)))...)
				keys.Insert(string(k))
				defaultRequests[string(k)] = q
			}
		}

		for k, q := range limit.MaxLimitRequestRatio {
			allErrs = append(allErrs, validateLimitRangeResourceName(limit.Type, string(k), idxPath.Child("maxLimitRequestRatio").Key(string(k)))...)
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
				allErrs = append(allErrs, field.Invalid(idxPath.Child("min").Key(string(k)), minQuantity, fmt.Sprintf("min value %s is greater than max value %s", minQuantity.String(), maxQuantity.String())))
			}

			if defaultRequestQuantityFound && minQuantityFound && minQuantity.Cmp(defaultRequestQuantity) > 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("defaultRequest").Key(string(k)), defaultRequestQuantity, fmt.Sprintf("min value %s is greater than default request value %s", minQuantity.String(), defaultRequestQuantity.String())))
			}

			if defaultRequestQuantityFound && maxQuantityFound && defaultRequestQuantity.Cmp(maxQuantity) > 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("defaultRequest").Key(string(k)), defaultRequestQuantity, fmt.Sprintf("default request value %s is greater than max value %s", defaultRequestQuantity.String(), maxQuantity.String())))
			}

			if defaultRequestQuantityFound && defaultQuantityFound && defaultRequestQuantity.Cmp(defaultQuantity) > 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("defaultRequest").Key(string(k)), defaultRequestQuantity, fmt.Sprintf("default request value %s is greater than default limit value %s", defaultRequestQuantity.String(), defaultQuantity.String())))
			}

			if defaultQuantityFound && minQuantityFound && minQuantity.Cmp(defaultQuantity) > 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("default").Key(string(k)), minQuantity, fmt.Sprintf("min value %s is greater than default value %s", minQuantity.String(), defaultQuantity.String())))
			}

			if defaultQuantityFound && maxQuantityFound && defaultQuantity.Cmp(maxQuantity) > 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("default").Key(string(k)), maxQuantity, fmt.Sprintf("default value %s is greater than max value %s", defaultQuantity.String(), maxQuantity.String())))
			}
			if maxRatioFound && maxRatio.Cmp(*resource.NewQuantity(1, resource.DecimalSI)) < 0 {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("maxLimitRequestRatio").Key(string(k)), maxRatio, fmt.Sprintf("ratio %s is less than 1", maxRatio.String())))
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
					allErrs = append(allErrs, field.Invalid(idxPath.Child("maxLimitRequestRatio").Key(string(k)), maxRatio, fmt.Sprintf("ratio %s is greater than max/min = %f", maxRatio.String(), maxRatioLimit)))
				}
			}
		}
	}

	return allErrs
}

// ValidateServiceAccount tests if required fields in the ServiceAccount are set.
func ValidateServiceAccount(serviceAccount *api.ServiceAccount) field.ErrorList {
	allErrs := ValidateObjectMeta(&serviceAccount.ObjectMeta, true, ValidateServiceAccountName, field.NewPath("metadata"))
	return allErrs
}

// ValidateServiceAccountUpdate tests if required fields in the ServiceAccount are set.
func ValidateServiceAccountUpdate(newServiceAccount, oldServiceAccount *api.ServiceAccount) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newServiceAccount.ObjectMeta, &oldServiceAccount.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateServiceAccount(newServiceAccount)...)
	return allErrs
}

// ValidateSecret tests if required fields in the Secret are set.
func ValidateSecret(secret *api.Secret) field.ErrorList {
	allErrs := ValidateObjectMeta(&secret.ObjectMeta, true, ValidateSecretName, field.NewPath("metadata"))

	dataPath := field.NewPath("data")
	totalSize := 0
	for key, value := range secret.Data {
		for _, msg := range validation.IsConfigMapKey(key) {
			allErrs = append(allErrs, field.Invalid(dataPath.Key(key), key, msg))
		}
		totalSize += len(value)
	}
	if totalSize > api.MaxSecretSize {
		allErrs = append(allErrs, field.TooLong(dataPath, "", api.MaxSecretSize))
	}

	switch secret.Type {
	case api.SecretTypeServiceAccountToken:
		// Only require Annotations[kubernetes.io/service-account.name]
		// Additional fields (like Annotations[kubernetes.io/service-account.uid] and Data[token]) might be contributed later by a controller loop
		if value := secret.Annotations[api.ServiceAccountNameKey]; len(value) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("metadata", "annotations").Key(api.ServiceAccountNameKey), ""))
		}
	case api.SecretTypeOpaque, "":
	// no-op
	case api.SecretTypeDockercfg:
		dockercfgBytes, exists := secret.Data[api.DockerConfigKey]
		if !exists {
			allErrs = append(allErrs, field.Required(dataPath.Key(api.DockerConfigKey), ""))
			break
		}

		// make sure that the content is well-formed json.
		if err := json.Unmarshal(dockercfgBytes, &map[string]interface{}{}); err != nil {
			allErrs = append(allErrs, field.Invalid(dataPath.Key(api.DockerConfigKey), "<secret contents redacted>", err.Error()))
		}
	case api.SecretTypeDockerConfigJson:
		dockerConfigJsonBytes, exists := secret.Data[api.DockerConfigJsonKey]
		if !exists {
			allErrs = append(allErrs, field.Required(dataPath.Key(api.DockerConfigJsonKey), ""))
			break
		}

		// make sure that the content is well-formed json.
		if err := json.Unmarshal(dockerConfigJsonBytes, &map[string]interface{}{}); err != nil {
			allErrs = append(allErrs, field.Invalid(dataPath.Key(api.DockerConfigJsonKey), "<secret contents redacted>", err.Error()))
		}
	case api.SecretTypeBasicAuth:
		_, usernameFieldExists := secret.Data[api.BasicAuthUsernameKey]
		_, passwordFieldExists := secret.Data[api.BasicAuthPasswordKey]

		// username or password might be empty, but the field must be present
		if !usernameFieldExists && !passwordFieldExists {
			allErrs = append(allErrs, field.Required(field.NewPath("data[%s]").Key(api.BasicAuthUsernameKey), ""))
			allErrs = append(allErrs, field.Required(field.NewPath("data[%s]").Key(api.BasicAuthPasswordKey), ""))
			break
		}
	case api.SecretTypeSSHAuth:
		if len(secret.Data[api.SSHAuthPrivateKey]) == 0 {
			allErrs = append(allErrs, field.Required(field.NewPath("data[%s]").Key(api.SSHAuthPrivateKey), ""))
			break
		}

	case api.SecretTypeTLS:
		if _, exists := secret.Data[api.TLSCertKey]; !exists {
			allErrs = append(allErrs, field.Required(dataPath.Key(api.TLSCertKey), ""))
		}
		if _, exists := secret.Data[api.TLSPrivateKeyKey]; !exists {
			allErrs = append(allErrs, field.Required(dataPath.Key(api.TLSPrivateKeyKey), ""))
		}
	// TODO: Verify that the key matches the cert.
	default:
		// no-op
	}

	return allErrs
}

// ValidateSecretUpdate tests if required fields in the Secret are set.
func ValidateSecretUpdate(newSecret, oldSecret *api.Secret) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newSecret.ObjectMeta, &oldSecret.ObjectMeta, field.NewPath("metadata"))

	if len(newSecret.Type) == 0 {
		newSecret.Type = oldSecret.Type
	}

	allErrs = append(allErrs, ValidateImmutableField(newSecret.Type, oldSecret.Type, field.NewPath("type"))...)

	allErrs = append(allErrs, ValidateSecret(newSecret)...)
	return allErrs
}

// ValidateConfigMapName can be used to check whether the given ConfigMap name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateConfigMapName = NameIsDNSSubdomain

// ValidateConfigMap tests whether required fields in the ConfigMap are set.
func ValidateConfigMap(cfg *api.ConfigMap) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&cfg.ObjectMeta, true, ValidateConfigMapName, field.NewPath("metadata"))...)

	totalSize := 0

	for key, value := range cfg.Data {
		for _, msg := range validation.IsConfigMapKey(key) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("data").Key(key), key, msg))
		}
		totalSize += len(value)
	}
	if totalSize > api.MaxSecretSize {
		allErrs = append(allErrs, field.TooLong(field.NewPath("data"), "", api.MaxSecretSize))
	}

	return allErrs
}

// ValidateConfigMapUpdate tests if required fields in the ConfigMap are set.
func ValidateConfigMapUpdate(newCfg, oldCfg *api.ConfigMap) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&newCfg.ObjectMeta, &oldCfg.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, ValidateConfigMap(newCfg)...)

	return allErrs
}

func validateBasicResource(quantity resource.Quantity, fldPath *field.Path) field.ErrorList {
	if quantity.Value() < 0 {
		return field.ErrorList{field.Invalid(fldPath, quantity.Value(), "must be a valid resource quantity")}
	}
	return field.ErrorList{}
}

// Validates resource requirement spec.
func ValidateResourceRequirements(requirements *api.ResourceRequirements, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	limPath := fldPath.Child("limits")
	reqPath := fldPath.Child("requests")
	for resourceName, quantity := range requirements.Limits {
		fldPath := limPath.Key(string(resourceName))
		// Validate resource name.
		allErrs = append(allErrs, validateContainerResourceName(string(resourceName), fldPath)...)
		if api.IsStandardResourceName(string(resourceName)) {
			allErrs = append(allErrs, validateBasicResource(quantity, fldPath.Key(string(resourceName)))...)
		}
		// Check that request <= limit.
		requestQuantity, exists := requirements.Requests[resourceName]
		if exists {
			// For GPUs, not only requests can't exceed limits, they also can't be lower, i.e. must be equal.
			if resourceName == api.ResourceNvidiaGPU && quantity.Cmp(requestQuantity) != 0 {
				allErrs = append(allErrs, field.Invalid(reqPath, requestQuantity.String(), fmt.Sprintf("must be equal to %s limit", api.ResourceNvidiaGPU)))
			} else if quantity.Cmp(requestQuantity) < 0 {
				allErrs = append(allErrs, field.Invalid(limPath, quantity.String(), fmt.Sprintf("must be greater than or equal to %s request", resourceName)))
			}
		}
	}
	for resourceName, quantity := range requirements.Requests {
		fldPath := reqPath.Key(string(resourceName))
		// Validate resource name.
		allErrs = append(allErrs, validateContainerResourceName(string(resourceName), fldPath)...)
		if api.IsStandardResourceName(string(resourceName)) {
			allErrs = append(allErrs, validateBasicResource(quantity, fldPath.Key(string(resourceName)))...)
		}
	}
	return allErrs
}

// validateResourceQuotaScopes ensures that each enumerated hard resource constraint is valid for set of scopes
func validateResourceQuotaScopes(resourceQuotaSpec *api.ResourceQuotaSpec, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(resourceQuotaSpec.Scopes) == 0 {
		return allErrs
	}
	hardLimits := sets.NewString()
	for k := range resourceQuotaSpec.Hard {
		hardLimits.Insert(string(k))
	}
	fldPath := fld.Child("scopes")
	scopeSet := sets.NewString()
	for _, scope := range resourceQuotaSpec.Scopes {
		if !api.IsStandardResourceQuotaScope(string(scope)) {
			allErrs = append(allErrs, field.Invalid(fldPath, resourceQuotaSpec.Scopes, "unsupported scope"))
		}
		for _, k := range hardLimits.List() {
			if api.IsStandardQuotaResourceName(k) && !api.IsResourceQuotaScopeValidForResource(scope, k) {
				allErrs = append(allErrs, field.Invalid(fldPath, resourceQuotaSpec.Scopes, "unsupported scope applied to resource"))
			}
		}
		scopeSet.Insert(string(scope))
	}
	invalidScopePairs := []sets.String{
		sets.NewString(string(api.ResourceQuotaScopeBestEffort), string(api.ResourceQuotaScopeNotBestEffort)),
		sets.NewString(string(api.ResourceQuotaScopeTerminating), string(api.ResourceQuotaScopeNotTerminating)),
	}
	for _, invalidScopePair := range invalidScopePairs {
		if scopeSet.HasAll(invalidScopePair.List()...) {
			allErrs = append(allErrs, field.Invalid(fldPath, resourceQuotaSpec.Scopes, "conflicting scopes"))
		}
	}
	return allErrs
}

// ValidateResourceQuota tests if required fields in the ResourceQuota are set.
func ValidateResourceQuota(resourceQuota *api.ResourceQuota) field.ErrorList {
	allErrs := ValidateObjectMeta(&resourceQuota.ObjectMeta, true, ValidateResourceQuotaName, field.NewPath("metadata"))

	allErrs = append(allErrs, ValidateResourceQuotaSpec(&resourceQuota.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateResourceQuotaStatus(&resourceQuota.Status, field.NewPath("status"))...)

	return allErrs
}

func ValidateResourceQuotaStatus(status *api.ResourceQuotaStatus, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	fldPath := fld.Child("hard")
	for k, v := range status.Hard {
		resPath := fldPath.Key(string(k))
		allErrs = append(allErrs, ValidateResourceQuotaResourceName(string(k), resPath)...)
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
	}
	fldPath = fld.Child("used")
	for k, v := range status.Used {
		resPath := fldPath.Key(string(k))
		allErrs = append(allErrs, ValidateResourceQuotaResourceName(string(k), resPath)...)
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
	}

	return allErrs
}

func ValidateResourceQuotaSpec(resourceQuotaSpec *api.ResourceQuotaSpec, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	fldPath := fld.Child("hard")
	for k, v := range resourceQuotaSpec.Hard {
		resPath := fldPath.Key(string(k))
		allErrs = append(allErrs, ValidateResourceQuotaResourceName(string(k), resPath)...)
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
	}
	allErrs = append(allErrs, validateResourceQuotaScopes(resourceQuotaSpec, fld)...)

	return allErrs
}

// ValidateResourceQuantityValue enforces that specified quantity is valid for specified resource
func ValidateResourceQuantityValue(resource string, value resource.Quantity, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNonnegativeQuantity(value, fldPath)...)
	if api.IsIntegerResourceName(resource) {
		if value.MilliValue()%int64(1000) != int64(0) {
			allErrs = append(allErrs, field.Invalid(fldPath, value, isNotIntegerErrorMsg))
		}
	}
	return allErrs
}

// ValidateResourceQuotaUpdate tests to see if the update is legal for an end user to make.
// newResourceQuota is updated with fields that cannot be changed.
func ValidateResourceQuotaUpdate(newResourceQuota, oldResourceQuota *api.ResourceQuota) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newResourceQuota.ObjectMeta, &oldResourceQuota.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateResourceQuotaSpec(&newResourceQuota.Spec, field.NewPath("spec"))...)

	// ensure scopes cannot change, and that resources are still valid for scope
	fldPath := field.NewPath("spec", "scopes")
	oldScopes := sets.NewString()
	newScopes := sets.NewString()
	for _, scope := range newResourceQuota.Spec.Scopes {
		newScopes.Insert(string(scope))
	}
	for _, scope := range oldResourceQuota.Spec.Scopes {
		oldScopes.Insert(string(scope))
	}
	if !oldScopes.Equal(newScopes) {
		allErrs = append(allErrs, field.Invalid(fldPath, newResourceQuota.Spec.Scopes, "field is immutable"))
	}

	newResourceQuota.Status = oldResourceQuota.Status
	return allErrs
}

// ValidateResourceQuotaStatusUpdate tests to see if the status update is legal for an end user to make.
// newResourceQuota is updated with fields that cannot be changed.
func ValidateResourceQuotaStatusUpdate(newResourceQuota, oldResourceQuota *api.ResourceQuota) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newResourceQuota.ObjectMeta, &oldResourceQuota.ObjectMeta, field.NewPath("metadata"))
	if len(newResourceQuota.ResourceVersion) == 0 {
		allErrs = append(allErrs, field.Required(field.NewPath("resourceVersion"), ""))
	}
	fldPath := field.NewPath("status", "hard")
	for k, v := range newResourceQuota.Status.Hard {
		resPath := fldPath.Key(string(k))
		allErrs = append(allErrs, ValidateResourceQuotaResourceName(string(k), resPath)...)
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
	}
	fldPath = field.NewPath("status", "used")
	for k, v := range newResourceQuota.Status.Used {
		resPath := fldPath.Key(string(k))
		allErrs = append(allErrs, ValidateResourceQuotaResourceName(string(k), resPath)...)
		allErrs = append(allErrs, ValidateResourceQuantityValue(string(k), v, resPath)...)
	}
	newResourceQuota.Spec = oldResourceQuota.Spec
	return allErrs
}

// ValidateNamespace tests if required fields are set.
func ValidateNamespace(namespace *api.Namespace) field.ErrorList {
	allErrs := ValidateObjectMeta(&namespace.ObjectMeta, false, ValidateNamespaceName, field.NewPath("metadata"))
	for i := range namespace.Spec.Finalizers {
		allErrs = append(allErrs, validateFinalizerName(string(namespace.Spec.Finalizers[i]), field.NewPath("spec", "finalizers"))...)
	}
	return allErrs
}

// Validate finalizer names
func validateFinalizerName(stringValue string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsQualifiedName(stringValue) {
		allErrs = append(allErrs, field.Invalid(fldPath, stringValue, msg))
	}
	if len(allErrs) != 0 {
		return allErrs
	}

	if len(strings.Split(stringValue, "/")) == 1 {
		if !api.IsStandardFinalizerName(stringValue) {
			return append(allErrs, field.Invalid(fldPath, stringValue, "name is neither a standard finalizer name nor is it fully qualified"))
		}
	}

	return field.ErrorList{}
}

// ValidateNamespaceUpdate tests to make sure a namespace update can be applied.
// newNamespace is updated with fields that cannot be changed
func ValidateNamespaceUpdate(newNamespace *api.Namespace, oldNamespace *api.Namespace) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newNamespace.ObjectMeta, &oldNamespace.ObjectMeta, field.NewPath("metadata"))
	newNamespace.Spec.Finalizers = oldNamespace.Spec.Finalizers
	newNamespace.Status = oldNamespace.Status
	return allErrs
}

// ValidateNamespaceStatusUpdate tests to see if the update is legal for an end user to make. newNamespace is updated with fields
// that cannot be changed.
func ValidateNamespaceStatusUpdate(newNamespace, oldNamespace *api.Namespace) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newNamespace.ObjectMeta, &oldNamespace.ObjectMeta, field.NewPath("metadata"))
	newNamespace.Spec = oldNamespace.Spec
	if newNamespace.DeletionTimestamp.IsZero() {
		if newNamespace.Status.Phase != api.NamespaceActive {
			allErrs = append(allErrs, field.Invalid(field.NewPath("status", "Phase"), newNamespace.Status.Phase, "may only be 'Active' if `deletionTimestamp` is empty"))
		}
	} else {
		if newNamespace.Status.Phase != api.NamespaceTerminating {
			allErrs = append(allErrs, field.Invalid(field.NewPath("status", "Phase"), newNamespace.Status.Phase, "may only be 'Terminating' if `deletionTimestamp` is not empty"))
		}
	}
	return allErrs
}

// ValidateNamespaceFinalizeUpdate tests to see if the update is legal for an end user to make.
// newNamespace is updated with fields that cannot be changed.
func ValidateNamespaceFinalizeUpdate(newNamespace, oldNamespace *api.Namespace) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newNamespace.ObjectMeta, &oldNamespace.ObjectMeta, field.NewPath("metadata"))

	fldPath := field.NewPath("spec", "finalizers")
	for i := range newNamespace.Spec.Finalizers {
		idxPath := fldPath.Index(i)
		allErrs = append(allErrs, validateFinalizerName(string(newNamespace.Spec.Finalizers[i]), idxPath)...)
	}
	newNamespace.Status = oldNamespace.Status
	return allErrs
}

// ValidateEndpoints tests if required fields are set.
func ValidateEndpoints(endpoints *api.Endpoints) field.ErrorList {
	allErrs := ValidateObjectMeta(&endpoints.ObjectMeta, true, ValidateEndpointsName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateEndpointsSpecificAnnotations(endpoints.Annotations, field.NewPath("annotations"))...)
	allErrs = append(allErrs, validateEndpointSubsets(endpoints.Subsets, field.NewPath("subsets"))...)
	return allErrs
}

func validateEndpointSubsets(subsets []api.EndpointSubset, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for i := range subsets {
		ss := &subsets[i]
		idxPath := fldPath.Index(i)

		if len(ss.Addresses) == 0 && len(ss.NotReadyAddresses) == 0 {
			//TODO: consider adding a RequiredOneOf() error for this and similar cases
			allErrs = append(allErrs, field.Required(idxPath, "must specify `addresses` or `notReadyAddresses`"))
		}
		if len(ss.Ports) == 0 {
			allErrs = append(allErrs, field.Required(idxPath.Child("ports"), ""))
		}
		for addr := range ss.Addresses {
			allErrs = append(allErrs, validateEndpointAddress(&ss.Addresses[addr], idxPath.Child("addresses").Index(addr))...)
		}
		for addr := range ss.NotReadyAddresses {
			allErrs = append(allErrs, validateEndpointAddress(&ss.NotReadyAddresses[addr], idxPath.Child("notReadyAddresses").Index(addr))...)
		}
		for port := range ss.Ports {
			allErrs = append(allErrs, validateEndpointPort(&ss.Ports[port], len(ss.Ports) > 1, idxPath.Child("ports").Index(port))...)
		}
	}

	return allErrs
}

func validateEndpointAddress(address *api.EndpointAddress, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsValidIP(address.IP) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("ip"), address.IP, msg))
	}
	if len(address.Hostname) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(address.Hostname, fldPath.Child("hostname"))...)
	}
	if len(allErrs) > 0 {
		return allErrs
	}
	allErrs = append(allErrs, validateNonSpecialIP(address.IP, fldPath.Child("ip"))...)
	return allErrs
}

func validateNonSpecialIP(ipAddress string, fldPath *field.Path) field.ErrorList {
	// We disallow some IPs as endpoints or external-ips.  Specifically,
	// unspecified and loopback addresses are nonsensical and link-local
	// addresses tend to be used for node-centric purposes (e.g. metadata
	// service).
	allErrs := field.ErrorList{}
	ip := net.ParseIP(ipAddress)
	if ip == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "must be a valid IP address"))
		return allErrs
	}
	if ip.IsUnspecified() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be unspecified (0.0.0.0)"))
	}
	if ip.IsLoopback() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be in the loopback range (127.0.0.0/8)"))
	}
	if ip.IsLinkLocalUnicast() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be in the link-local range (169.254.0.0/16)"))
	}
	if ip.IsLinkLocalMulticast() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be in the link-local multicast range (224.0.0.0/24)"))
	}
	return allErrs
}

func validateEndpointPort(port *api.EndpointPort, requireName bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if requireName && len(port.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else if len(port.Name) != 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(port.Name, fldPath.Child("name"))...)
	}
	for _, msg := range validation.IsValidPortNum(int(port.Port)) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("port"), port.Port, msg))
	}
	if len(port.Protocol) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("protocol"), ""))
	} else if !supportedPortProtocols.Has(string(port.Protocol)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("protocol"), port.Protocol, supportedPortProtocols.List()))
	}
	return allErrs
}

// ValidateEndpointsUpdate tests to make sure an endpoints update can be applied.
func ValidateEndpointsUpdate(newEndpoints, oldEndpoints *api.Endpoints) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newEndpoints.ObjectMeta, &oldEndpoints.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateEndpointSubsets(newEndpoints.Subsets, field.NewPath("subsets"))...)
	allErrs = append(allErrs, ValidateEndpointsSpecificAnnotations(newEndpoints.Annotations, field.NewPath("annotations"))...)
	return allErrs
}

// ValidateSecurityContext ensure the security context contains valid settings
func ValidateSecurityContext(sc *api.SecurityContext, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	//this should only be true for testing since SecurityContext is defaulted by the api
	if sc == nil {
		return allErrs
	}

	if sc.Privileged != nil {
		if *sc.Privileged && !capabilities.Get().AllowPrivileged {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("privileged"), "disallowed by policy"))
		}
	}

	if sc.RunAsUser != nil {
		if *sc.RunAsUser < 0 {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("runAsUser"), *sc.RunAsUser, isNegativeErrorMsg))
		}
	}
	return allErrs
}

func ValidatePodLogOptions(opts *api.PodLogOptions) field.ErrorList {
	allErrs := field.ErrorList{}
	if opts.TailLines != nil && *opts.TailLines < 0 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("tailLines"), *opts.TailLines, isNegativeErrorMsg))
	}
	if opts.LimitBytes != nil && *opts.LimitBytes < 1 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("limitBytes"), *opts.LimitBytes, "must be greater than 0"))
	}
	switch {
	case opts.SinceSeconds != nil && opts.SinceTime != nil:
		allErrs = append(allErrs, field.Forbidden(field.NewPath(""), "at most one of `sinceTime` or `sinceSeconds` may be specified"))
	case opts.SinceSeconds != nil:
		if *opts.SinceSeconds < 1 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("sinceSeconds"), *opts.SinceSeconds, "must be greater than 0"))
		}
	}
	return allErrs
}

// ValidateLoadBalancerStatus validates required fields on a LoadBalancerStatus
func ValidateLoadBalancerStatus(status *api.LoadBalancerStatus, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, ingress := range status.Ingress {
		idxPath := fldPath.Child("ingress").Index(i)
		if len(ingress.IP) > 0 {
			if isIP := (net.ParseIP(ingress.IP) != nil); !isIP {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("ip"), ingress.IP, "must be a valid IP address"))
			}
		}
		if len(ingress.Hostname) > 0 {
			for _, msg := range validation.IsDNS1123Subdomain(ingress.Hostname) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostname"), ingress.Hostname, msg))
			}
			if isIP := (net.ParseIP(ingress.Hostname) != nil); isIP {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostname"), ingress.Hostname, "must be a DNS name, not an IP address"))
			}
		}
	}
	return allErrs
}

// TODO: remove this after we EOL the annotation that carries it.
func isValidHostnamesMap(serializedPodHostNames string) bool {
	if len(serializedPodHostNames) == 0 {
		return false
	}
	podHostNames := map[string]endpoints.HostRecord{}
	err := json.Unmarshal([]byte(serializedPodHostNames), &podHostNames)
	if err != nil {
		return false
	}

	for ip, hostRecord := range podHostNames {
		if len(validation.IsDNS1123Label(hostRecord.HostName)) != 0 {
			return false
		}
		if net.ParseIP(ip) == nil {
			return false
		}
	}
	return true
}
