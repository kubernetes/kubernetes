/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"regexp"
	"strings"

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	dracel "k8s.io/dynamic-resource-allocation/cel"
	corevalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/resource"
)

// TODO: more validation, fix tests...

var (
	// validateResourceDriverName reuses the validation of a CSI driver because
	// the allowed values are exactly the same.
	validateDriverName = corevalidation.ValidateCSIDriverName
	validateDeviceName = corevalidation.ValidateDNS1123Subdomain
)

// ValidateResourceClaim validates a ResourceClaim.
func ValidateResourceClaim(resourceClaim *resource.ResourceClaim) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&resourceClaim.ObjectMeta, true, corevalidation.ValidateResourceClaimName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourceClaimSpec(&resourceClaim.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateResourceClaimSpec(spec *resource.ResourceClaimSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// for _, msg := range corevalidation.ValidateClassName(spec.ResourceClassName, false) {
	// 	allErrs = append(allErrs, field.Invalid(fldPath.Child("resourceClassName"), spec.ResourceClassName, msg))
	// }
	return allErrs
}

// ValidateResourceClaimUpdate tests if an update to ResourceClaim is valid.
func ValidateResourceClaimUpdate(resourceClaim, oldClaim *resource.ResourceClaim) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&resourceClaim.ObjectMeta, &oldClaim.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(resourceClaim.Spec, oldClaim.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateResourceClaim(resourceClaim)...)
	return allErrs
}

// ValidateClaimStatusUpdate tests if an update to the status of a ResourceClaim is valid.
func ValidateClaimStatusUpdate(resourceClaim, oldClaim *resource.ResourceClaim) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&resourceClaim.ObjectMeta, &oldClaim.ObjectMeta, field.NewPath("metadata"))
	fldPath := field.NewPath("status")

	allErrs = append(allErrs, validateAllocationResult(resourceClaim.Status.Allocation, fldPath.Child("allocation"))...)
	allErrs = append(allErrs, validateResourceClaimConsumers(resourceClaim.Status.ReservedFor, resource.ResourceClaimReservedForMaxSize, fldPath.Child("reservedFor"))...)

	// Now check for invariants that must be valid for a ResourceClaim.
	if len(resourceClaim.Status.ReservedFor) > 0 {
		if resourceClaim.Status.Allocation == nil {
			allErrs = append(allErrs, field.Forbidden(fldPath.Child("reservedFor"), "may not be specified when `allocated` is not set"))
		} else {
			// Items may be removed from ReservedFor while the claim is meant to be deallocated,
			// but not added.
			if resourceClaim.DeletionTimestamp != nil || resourceClaim.Status.DeallocationRequested {
				oldSet := sets.New(oldClaim.Status.ReservedFor...)
				newSet := sets.New(resourceClaim.Status.ReservedFor...)
				newItems := newSet.Difference(oldSet)
				if len(newItems) > 0 {
					allErrs = append(allErrs, field.Forbidden(fldPath.Child("reservedFor"), "new entries may not be added while `deallocationRequested` or `deletionTimestamp` are set"))
				}
			}
		}
	}

	// Updates to a populated resourceClaim.Status.Allocation are not allowed
	if oldClaim.Status.Allocation != nil && resourceClaim.Status.Allocation != nil {
		allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(resourceClaim.Status.Allocation, oldClaim.Status.Allocation, fldPath.Child("allocation"))...)
	}

	if !oldClaim.Status.DeallocationRequested &&
		resourceClaim.Status.DeallocationRequested &&
		len(resourceClaim.Status.ReservedFor) > 0 {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("deallocationRequested"), "deallocation cannot be requested while `reservedFor` is set"))
	}

	if resourceClaim.Status.Allocation == nil &&
		resourceClaim.Status.DeallocationRequested {
		// Either one or the other field was modified incorrectly.
		// For the sake of simplicity this only reports the invalid
		// end result.
		allErrs = append(allErrs, field.Forbidden(fldPath, "`allocation` must be set when `deallocationRequested` is set"))
	}

	// Once deallocation has been requested, that request cannot be removed
	// anymore because the deallocation may already have started. The field
	// can only get reset by the driver together with removing the
	// allocation.
	if oldClaim.Status.DeallocationRequested &&
		!resourceClaim.Status.DeallocationRequested &&
		resourceClaim.Status.Allocation != nil {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("deallocationRequested"), "may not be cleared when `allocation` is set"))
	}

	return allErrs
}

func validateAllocationResult(allocation *resource.AllocationResult, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if allocation != nil {
		// if len(allocation.ResourceHandles) > 0 {
		// 	allErrs = append(allErrs, validateResourceHandles(allocation.ResourceHandles, resource.AllocationResultResourceHandlesMaxSize, fldPath.Child("resourceHandles"))...)
		// }
		if allocation.NodeSelector != nil {
			allErrs = append(allErrs, corevalidation.ValidateNodeSelector(allocation.NodeSelector, fldPath.Child("nodeSelector"))...)
		}
	}
	return allErrs
}

func validateResourceClaimUserReference(ref resource.ResourceClaimConsumerReference, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if ref.Resource == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("resource"), ""))
	}
	if ref.Name == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	}
	if ref.UID == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("uid"), ""))
	}
	return allErrs
}

// validateSliceIsASet ensures that a slice contains no duplicates and does not exceed a certain maximum size.
func validateSliceIsASet[T comparable](slice []T, maxSize int, validateItem func(item T, fldPath *field.Path) field.ErrorList, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allItems := sets.New[T]()
	for i, item := range slice {
		idxPath := fldPath.Index(i)
		if allItems.Has(item) {
			allErrs = append(allErrs, field.Duplicate(idxPath, item))
		} else {
			allErrs = append(allErrs, validateItem(item, idxPath)...)
			allItems.Insert(item)
		}
	}
	if len(slice) > maxSize {
		// Dumping the entire field into the error message is likely to be too long,
		// in particular when it is already beyond the maximum size. Instead this
		// just shows the number of entries.
		allErrs = append(allErrs, field.TooLongMaxLength(fldPath, len(slice), maxSize))
	}
	return allErrs
}

// validateResourceClaimConsumers ensures that the slice contains no duplicate UIDs and does not exceed a certain maximum size.
func validateResourceClaimConsumers(consumers []resource.ResourceClaimConsumerReference, maxSize int, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allUIDs := sets.New[types.UID]()
	for i, consumer := range consumers {
		idxPath := fldPath.Index(i)
		if allUIDs.Has(consumer.UID) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("uid"), consumer.UID))
		} else {
			allErrs = append(allErrs, validateResourceClaimUserReference(consumer, idxPath)...)
			allUIDs.Insert(consumer.UID)
		}
	}
	if len(consumers) > maxSize {
		// Dumping the entire field into the error message is likely to be too long,
		// in particular when it is already beyond the maximum size. Instead this
		// just shows the number of entries.
		allErrs = append(allErrs, field.TooLongMaxLength(fldPath, len(consumers), maxSize))
	}
	return allErrs
}

// ValidateClass validates a DeviceClass.
func ValidateDeviceClass(class *resource.DeviceClass) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&class.ObjectMeta, false, corevalidation.ValidateClassName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateDeviceClassSpec(&class.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateDeviceClassSpec(spec *resource.DeviceClassSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if spec.SuitableNodes != nil {
		allErrs = append(allErrs, corevalidation.ValidateNodeSelector(spec.SuitableNodes, field.NewPath("suitableNodes"))...)
	}
	return allErrs
}

// ValidateClassUpdate tests if an update to DeviceClass is valid.
func ValidateDeviceClassUpdate(class, oldClass *resource.DeviceClass) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&class.ObjectMeta, &oldClass.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateDeviceClass(class)...)
	return allErrs
}

// ValidatePodSchedulingContext validates a PodSchedulingContext.
func ValidatePodSchedulingContexts(schedulingCtx *resource.PodSchedulingContext) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&schedulingCtx.ObjectMeta, true, corevalidation.ValidatePodName, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodSchedulingSpec(&schedulingCtx.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validatePodSchedulingSpec(spec *resource.PodSchedulingContextSpec, fldPath *field.Path) field.ErrorList {
	allErrs := validateSliceIsASet(spec.PotentialNodes, resource.PodSchedulingNodeListMaxSize, validateNodeName, fldPath.Child("potentialNodes"))
	return allErrs
}

// ValidatePodSchedulingContextUpdate tests if an update to PodSchedulingContext is valid.
func ValidatePodSchedulingContextUpdate(schedulingCtx, oldSchedulingCtx *resource.PodSchedulingContext) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&schedulingCtx.ObjectMeta, &oldSchedulingCtx.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidatePodSchedulingContexts(schedulingCtx)...)
	return allErrs
}

// ValidatePodSchedulingContextStatusUpdate tests if an update to the status of a PodSchedulingContext is valid.
func ValidatePodSchedulingContextStatusUpdate(schedulingCtx, oldSchedulingCtx *resource.PodSchedulingContext) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&schedulingCtx.ObjectMeta, &oldSchedulingCtx.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validatePodSchedulingStatus(&schedulingCtx.Status, field.NewPath("status"))...)
	return allErrs
}

func validatePodSchedulingStatus(status *resource.PodSchedulingContextStatus, fldPath *field.Path) field.ErrorList {
	return validatePodSchedulingClaims(status.ResourceClaims, fldPath.Child("claims"))
}

func validatePodSchedulingClaims(claimStatuses []resource.ResourceClaimSchedulingStatus, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	names := sets.NewString()
	for i, claimStatus := range claimStatuses {
		allErrs = append(allErrs, validatePodSchedulingClaim(claimStatus, fldPath.Index(i))...)
		if names.Has(claimStatus.Name) {
			allErrs = append(allErrs, field.Duplicate(fldPath.Index(i), claimStatus.Name))
		} else {
			names.Insert(claimStatus.Name)
		}
	}
	return allErrs
}

func validatePodSchedulingClaim(status resource.ResourceClaimSchedulingStatus, fldPath *field.Path) field.ErrorList {
	allErrs := validateSliceIsASet(status.UnsuitableNodes, resource.PodSchedulingNodeListMaxSize, validateNodeName, fldPath.Child("unsuitableNodes"))
	return allErrs
}

// ValidateClaimTemplace validates a ResourceClaimTemplate.
func ValidateResourceClaimTemplate(template *resource.ResourceClaimTemplate) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&template.ObjectMeta, true, corevalidation.ValidateResourceClaimTemplateName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourceClaimTemplateSpec(&template.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateResourceClaimTemplateSpec(spec *resource.ResourceClaimTemplateSpec, fldPath *field.Path) field.ErrorList {
	allErrs := corevalidation.ValidateTemplateObjectMeta(&spec.ObjectMeta, fldPath.Child("metadata"))
	allErrs = append(allErrs, validateResourceClaimSpec(&spec.Spec, fldPath.Child("spec"))...)
	return allErrs
}

// ValidateResourceClaimTemplateUpdate tests if an update to template is valid.
func ValidateResourceClaimTemplateUpdate(template, oldTemplate *resource.ResourceClaimTemplate) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&template.ObjectMeta, &oldTemplate.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(template.Spec, oldTemplate.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateResourceClaimTemplate(template)...)
	return allErrs
}

func validateNodeName(name string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	for _, msg := range corevalidation.ValidateNodeName(name, false) {
		allErrs = append(allErrs, field.Invalid(fldPath, name, msg))
	}
	return allErrs
}

// ValidateResourceSlice tests if a ResourceSlice object is valid.
func ValidateResourceSlice(slice *resource.ResourceSlice) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMeta(&slice.ObjectMeta, false, apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, validateResourceSliceSpec(&slice.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateResourceSliceSpec(spec *resource.ResourceSliceSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if spec.NodeName != "" {
		allErrs = append(allErrs, validateNodeName(spec.NodeName, field.NewPath("nodeName"))...)
	}
	allErrs = append(allErrs, validateDriverName(spec.Driver, field.NewPath("driverName"))...)
	return allErrs
}

// ValidateResourceSlice tests if a ResourceSlice update is valid.
func ValidateResourceSliceUpdate(resourceSlice, oldResourceSlice *resource.ResourceSlice) field.ErrorList {
	allErrs := corevalidation.ValidateObjectMetaUpdate(&resourceSlice.ObjectMeta, &oldResourceSlice.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateResourceSlice(resourceSlice)...)
	allErrs = append(allErrs, validateResourceSliceSpecUpdate(&resourceSlice.Spec, &oldResourceSlice.Spec, field.NewPath("spec"))...)
	return allErrs
}

func validateResourceSliceSpecUpdate(spec, oldSpec *resource.ResourceSliceSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(spec.NodeName, spec.NodeName, fldPath.Child("nodeName"))...)
	allErrs = append(allErrs, apimachineryvalidation.ValidateImmutableField(spec.Driver, spec.Driver, fldPath.Child("driverName"))...)
	return allErrs
}

func validateResourceRequests(requests []resource.DeviceRequest, fldPath *field.Path, opts Options) field.ErrorList {
	var allErrs field.ErrorList
	for i := range requests {
		idxPath := fldPath.Index(i)
		allErrs = append(allErrs, validateRequest(&requests[i], idxPath, opts)...)
	}
	if len(requests) == 0 {
		// We could allow this ("null claim"), but it also could be a user mistake, so we flag it as an error.
		allErrs = append(allErrs, field.Required(fldPath, "a claim must have at least one request"))
	}
	return allErrs
}

func validateRequest(request *resource.DeviceRequest, fldPath *field.Path, opts Options) field.ErrorList {
	allErrs := corevalidation.ValidateDNS1123Label(request.Name, fldPath.Child("name"))
	allErrs = append(allErrs, validateDeviceRequest(request.Device, fldPath.Child("device"), opts)...)
	return allErrs
}

func validateDeviceRequest(deviceRequest *resource.DeviceRequestDetail, fldPath *field.Path, opts Options) field.ErrorList {
	if deviceRequest == nil {
		return field.ErrorList{field.Required(fldPath, "")}
	}
	var allErrs field.ErrorList
	// TODO
	return allErrs
}

func validateSelectors(selectors []resource.DeviceSelector, fldPath *field.Path, opts Options) field.ErrorList {
	var allErrs field.ErrorList
	for i, selector := range selectors {
		idxPath := fldPath.Index(i)
		allErrs = append(allErrs, validateCELSelector(selector.CEL, idxPath.Child("cel"), opts)...)
	}
	return allErrs
}

func validateOpaqueConfiguration(parameters []resource.OpaqueDeviceConfiguration, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	driverNames := sets.New[string]()
	for i, parameters := range parameters {
		idxPath := fldPath.Index(i)
		driverName := parameters.Driver
		allErrs = append(allErrs, validateDriverName(driverName, idxPath.Child("driverName"))...)
		if driverNames.Has(driverName) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("driverName"), driverName))
		} else {
			driverNames.Insert(driverName)
		}
		// TODO: validate RawExtension as in https://github.com/kubernetes/kubernetes/pull/125549/
	}
	return allErrs
}

type Options struct {
	// Stored must be true if and only if validating CEL
	// expressions and fields that were already stored persistently. This makes
	// validation more permissive by enabling CEL definitions that are not
	// valid yet for new expressions and allowing unknown enums for fields.
	Stored bool
}

func validateDevices(devices []resource.Device, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	instanceNames := sets.New[string]()
	for i, device := range devices {
		idxPath := fldPath.Index(i)
		allErrs = append(allErrs, validateDeviceName(device.Name, idxPath.Child("name"))...)
		if instanceNames.Has(device.Name) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), device.Name))
		} else {
			instanceNames.Insert(device.Name)
		}
		allErrs = append(allErrs, validateAttributes(device.Attributes, idxPath.Child("attributes"))...)
	}
	return allErrs
}

var (
	numericIdentifier = `(0|[1-9]\d*)`

	preReleaseIdentifier = `(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)`

	buildIdentifier = `[0-9a-zA-Z-]+`

	semverRe = regexp.MustCompile(`^` +

		// dot-separated version segments (e.g. 1.2.3)
		numericIdentifier + `\.` + numericIdentifier + `\.` + numericIdentifier +

		// optional dot-separated prerelease segments (e.g. -alpha.PRERELEASE.1)
		`(-` + preReleaseIdentifier + `(\.` + preReleaseIdentifier + `)*)?` +

		// optional dot-separated build identifier segments (e.g. +build.id.20240305)
		`(\+` + buildIdentifier + `(\.` + buildIdentifier + `)*)?` +

		`$`)
)

func validateAttributes(attributes []resource.DeviceAttribute, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	attributeNames := sets.New[string]()
	for i, attribute := range attributes {
		idxPath := fldPath.Index(i)
		attributeName := attribute.Name
		allErrs = append(allErrs, validateAttributeName(attributeName, idxPath.Child("name"))...)
		if attributeNames.Has(attributeName) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), attributeName))
		} else {
			attributeNames.Insert(attributeName)
		}

		entries := sets.New[string]()
		if attribute.BoolValue != nil {
			entries.Insert("bool")
		}
		if attribute.IntValue != nil {
			entries.Insert("int")
		}
		if attribute.StringValue != nil {
			entries.Insert("string")
		}
		if attribute.VersionValue != nil {
			entries.Insert("version")
			if !semverRe.MatchString(*attribute.VersionValue) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("version"), *attribute.VersionValue, "must be a string compatible with semver.org spec 2.0.0"))
			}
		}

		switch len(entries) {
		case 0:
			allErrs = append(allErrs, field.Required(idxPath, "exactly one value must be set"))
		case 1:
			// Okay.
		default:
			allErrs = append(allErrs, field.Invalid(idxPath, sets.List(entries), "exactly one field must be set, not several"))
		}
	}
	return allErrs
}

func validateAttributeName(name string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if name == "" {
		allErrs = append(allErrs, field.Required(fldPath, "name required"))
		return allErrs
	}

	// Naming the two parts in a field path is tricky. Treating them as a child field
	// is not quite right, but close enough...
	parts := strings.Split(name, "/")
	switch len(parts) {
	case 1:
		allErrs = append(allErrs, validateCIdentifier(parts[0], fldPath.Child("identifier"))...)
	case 2:
		allErrs = append(allErrs, validateDriverName(parts[0], fldPath.Child("domain"))...)
		allErrs = append(allErrs, validateCIdentifier(parts[1], fldPath.Child("identifier"))...)

	}
	return allErrs
}

// cIdentifierFmt is the same as in https://github.com/google/cel-spec/blob/master/doc/langdef.md#overview for IDENT.
const cIdentifierFmt string = "[_a-zA-Z][_a-zA-Z0-9]*"
const cIdentifierErrMsg string = "a C identifier must consist of alphanumeric characters or '_', and must start with an alphabetic character or '_'"

var cIdentifierRegexp = regexp.MustCompile("^" + cIdentifierFmt + "$")

func validateCIdentifier(id string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	if len(id) > resource.DeviceMaxIDLength {
		allErrs = append(allErrs, field.TypeInvalid(fldPath, id, fmt.Sprintf("must not be longer than %d characters", resource.DeviceMaxIDLength)))
	}
	if !cIdentifierRegexp.MatchString(id) {
		allErrs = append(allErrs, field.TypeInvalid(fldPath, id, validation.RegexError(cIdentifierErrMsg, cIdentifierFmt, "myName", "_abc42")))
	}
	return allErrs
}

func validateCELSelector(celSelector *resource.CELDeviceSelector, fldPath *field.Path, opts Options) field.ErrorList {
	var allErrs field.ErrorList
	if celSelector == nil {
		allErrs = append(allErrs, field.Required(fldPath, "CEL selector required"))
		return allErrs
	}
	envType := environment.NewExpressions
	if opts.Stored {
		envType = environment.StoredExpressions
	}
	result := dracel.Compiler.CompileCELExpression(celSelector.Expression, envType)
	if result.Error != nil {
		allErrs = append(allErrs, convertCELErrorToValidationError(fldPath, celSelector.Expression, result.Error))
	}
	return allErrs
}

func convertCELErrorToValidationError(fldPath *field.Path, expression string, err error) *field.Error {
	if celErr, ok := err.(*cel.Error); ok {
		switch celErr.Type {
		case cel.ErrorTypeRequired:
			return field.Required(fldPath, celErr.Detail)
		case cel.ErrorTypeInvalid:
			return field.Invalid(fldPath, expression, celErr.Detail)
		case cel.ErrorTypeInternal:
			return field.InternalError(fldPath, celErr)
		}
	}
	return field.InternalError(fldPath, fmt.Errorf("unsupported error type: %w", err))
}
