/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	errs "k8s.io/kubernetes/pkg/util/fielderrors"
	"k8s.io/kubernetes/pkg/util/sets"
)

const isNegativeErrorMsg string = `must be non-negative`

// ValidateHorizontalPodAutoscaler can be used to check whether the given autoscaler name is valid.
// Prefix indicates this name will be used as part of generation, in which case trailing dashes are allowed.
func ValidateHorizontalPodAutoscalerName(name string, prefix bool) (bool, string) {
	// TODO: finally move it to pkg/api/validation and use nameIsDNSSubdomain function
	return apivalidation.ValidateReplicationControllerName(name, prefix)
}

func validateHorizontalPodAutoscalerSpec(autoscaler experimental.HorizontalPodAutoscalerSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if autoscaler.MinCount < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("minCount", autoscaler.MinCount, `must be non-negative`))
	}
	if autoscaler.MaxCount < autoscaler.MinCount {
		allErrs = append(allErrs, errs.NewFieldInvalid("maxCount", autoscaler.MaxCount, `must be bigger or equal to minCount`))
	}
	if autoscaler.ScaleRef == nil {
		allErrs = append(allErrs, errs.NewFieldRequired("scaleRef"))
	}
	resource := autoscaler.Target.Resource.String()
	if resource != string(api.ResourceMemory) && resource != string(api.ResourceCPU) {
		allErrs = append(allErrs, errs.NewFieldInvalid("target.resource", resource, "resource not supported by autoscaler"))
	}
	quantity := autoscaler.Target.Quantity.Value()
	if quantity < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("target.quantity", quantity, "must be non-negative"))
	}
	return allErrs
}

func ValidateHorizontalPodAutoscaler(autoscaler *experimental.HorizontalPodAutoscaler) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&autoscaler.ObjectMeta, true, ValidateHorizontalPodAutoscalerName).Prefix("metadata")...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(autoscaler.Spec)...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerUpdate(newAutoscler, oldAutoscaler *experimental.HorizontalPodAutoscaler) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&newAutoscler.ObjectMeta, &oldAutoscaler.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(newAutoscler.Spec)...)
	return allErrs
}

func ValidateThirdPartyResourceUpdate(old, update *experimental.ThirdPartyResource) errs.ValidationErrorList {
	return ValidateThirdPartyResource(update)
}

func ValidateThirdPartyResource(obj *experimental.ThirdPartyResource) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(obj.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", obj.Name, "name must be non-empty"))
	}
	versions := sets.String{}
	for ix := range obj.Versions {
		version := &obj.Versions[ix]
		if len(version.Name) == 0 {
			allErrs = append(allErrs, errs.NewFieldInvalid("name", version, "name can not be empty"))
		}
		if versions.Has(version.Name) {
			allErrs = append(allErrs, errs.NewFieldDuplicate("version", version))
		}
		versions.Insert(version.Name)
	}
	return allErrs
}

// ValidateDaemonSet tests if required fields in the DaemonSet are set.
func ValidateDaemonSet(controller *experimental.DaemonSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&controller.ObjectMeta, true, apivalidation.ValidateReplicationControllerName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDaemonSetSpec(&controller.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateDaemonSetUpdate tests if required fields in the DaemonSet are set.
func ValidateDaemonSetUpdate(oldController, controller *experimental.DaemonSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDaemonSetSpec(&controller.Spec).Prefix("spec")...)
	allErrs = append(allErrs, ValidateDaemonSetTemplateUpdate(oldController.Spec.Template, controller.Spec.Template).Prefix("spec.template")...)
	return allErrs
}

// ValidateDaemonSetTemplateUpdate tests that certain fields in the daemon set's pod template are not updated.
func ValidateDaemonSetTemplateUpdate(oldPodTemplate, podTemplate *api.PodTemplateSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	podSpec := podTemplate.Spec
	// podTemplate.Spec is not a pointer, so we can modify NodeSelector and NodeName directly.
	podSpec.NodeSelector = oldPodTemplate.Spec.NodeSelector
	podSpec.NodeName = oldPodTemplate.Spec.NodeName
	// In particular, we do not allow updates to container images at this point.
	if !api.Semantic.DeepEqual(oldPodTemplate.Spec, podSpec) {
		// TODO: Pinpoint the specific field that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, errs.NewFieldInvalid("spec", "content of spec is not printed out, please refer to the \"details\"", "may not update fields other than spec.nodeSelector"))
	}
	return allErrs
}

// ValidateDaemonSetSpec tests if required fields in the DaemonSetSpec are set.
func ValidateDaemonSetSpec(spec *experimental.DaemonSetSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	selector := labels.Set(spec.Selector).AsSelector()
	if selector.Empty() {
		allErrs = append(allErrs, errs.NewFieldRequired("selector"))
	}

	if spec.Template == nil {
		allErrs = append(allErrs, errs.NewFieldRequired("template"))
	} else {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, errs.NewFieldInvalid("template.metadata.labels", spec.Template.Labels, "selector does not match template"))
		}
		allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(spec.Template).Prefix("template")...)
		// Daemons typically run on more than one node, so mark Read-Write persistent disks as invalid.
		allErrs = append(allErrs, apivalidation.ValidateReadOnlyPersistentDisks(spec.Template.Spec.Volumes).Prefix("template.spec.volumes")...)
		// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
		if spec.Template.Spec.RestartPolicy != api.RestartPolicyAlways {
			allErrs = append(allErrs, errs.NewFieldValueNotSupported("template.spec.restartPolicy", spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
		}
	}
	return allErrs
}

// ValidateDaemonSetName can be used to check whether the given daemon set name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
func ValidateDaemonSetName(name string, prefix bool) (bool, string) {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// Validates that the given name can be used as a deployment name.
func ValidateDeploymentName(name string, prefix bool) (bool, string) {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

func ValidatePositiveIntOrPercent(intOrPercent util.IntOrString, fieldName string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if intOrPercent.Kind == util.IntstrString {
		if !util.IsValidPercent(intOrPercent.StrVal) {
			allErrs = append(allErrs, errs.NewFieldInvalid(fieldName, intOrPercent, "value should be int(5) or percentage(5%)"))
		}

	} else if intOrPercent.Kind == util.IntstrInt {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(intOrPercent.IntVal), fieldName)...)
	}
	return allErrs
}

func getPercentValue(intOrStringValue util.IntOrString) (int, bool) {
	if intOrStringValue.Kind != util.IntstrString || !util.IsValidPercent(intOrStringValue.StrVal) {
		return 0, false
	}
	value, _ := strconv.Atoi(intOrStringValue.StrVal[:len(intOrStringValue.StrVal)-1])
	return value, true
}

func getIntOrPercentValue(intOrStringValue util.IntOrString) int {
	value, isPercent := getPercentValue(intOrStringValue)
	if isPercent {
		return value
	}
	return intOrStringValue.IntVal
}

func IsNotMoreThan100Percent(intOrStringValue util.IntOrString, fieldName string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	value, isPercent := getPercentValue(intOrStringValue)
	if !isPercent || value <= 100 {
		return nil
	}
	allErrs = append(allErrs, errs.NewFieldInvalid(fieldName, intOrStringValue, "should not be more than 100%"))
	return allErrs
}

func ValidateRollingUpdateDeployment(rollingUpdate *experimental.RollingUpdateDeployment, fieldName string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidatePositiveIntOrPercent(rollingUpdate.MaxUnavailable, fieldName+"maxUnavailable")...)
	allErrs = append(allErrs, ValidatePositiveIntOrPercent(rollingUpdate.MaxSurge, fieldName+".maxSurge")...)
	if getIntOrPercentValue(rollingUpdate.MaxUnavailable) == 0 && getIntOrPercentValue(rollingUpdate.MaxSurge) == 0 {
		// Both MaxSurge and MaxUnavailable cannot be zero.
		allErrs = append(allErrs, errs.NewFieldInvalid(fieldName+".maxUnavailable", rollingUpdate.MaxUnavailable, "cannot be 0 when maxSurge is 0 as well"))
	}
	// Validate that MaxUnavailable is not more than 100%.
	allErrs = append(allErrs, IsNotMoreThan100Percent(rollingUpdate.MaxUnavailable, fieldName+".maxUnavailable")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(rollingUpdate.MinReadySeconds), fieldName+".minReadySeconds")...)
	return allErrs
}

func ValidateDeploymentStrategy(strategy *experimental.DeploymentStrategy, fieldName string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if strategy.RollingUpdate == nil {
		return allErrs
	}
	switch strategy.Type {
	case experimental.DeploymentRecreate:
		allErrs = append(allErrs, errs.NewFieldForbidden("rollingUpdate", "rollingUpdate should be nil when strategy type is "+experimental.DeploymentRecreate))
	case experimental.DeploymentRollingUpdate:
		allErrs = append(allErrs, ValidateRollingUpdateDeployment(strategy.RollingUpdate, "rollingUpdate")...)
	}
	return allErrs
}

// Validates given deployment spec.
func ValidateDeploymentSpec(spec *experimental.DeploymentSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonEmptySelector(spec.Selector, "selector")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(spec.Replicas), "replicas")...)
	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpecForRC(spec.Template, spec.Selector, spec.Replicas, "template")...)
	allErrs = append(allErrs, ValidateDeploymentStrategy(&spec.Strategy, "strategy")...)
	allErrs = append(allErrs, apivalidation.ValidateLabelName(spec.UniqueLabelKey, "uniqueLabel")...)
	return allErrs
}

func ValidateDeploymentUpdate(old, update *experimental.Deployment) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDeploymentSpec(&update.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateDeployment(obj *experimental.Deployment) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&obj.ObjectMeta, true, ValidateDeploymentName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDeploymentSpec(&obj.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateThirdPartyResourceDataUpdate(old, update *experimental.ThirdPartyResourceData) errs.ValidationErrorList {
	return ValidateThirdPartyResourceData(update)
}

func ValidateThirdPartyResourceData(obj *experimental.ThirdPartyResourceData) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(obj.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", obj.Name, "name must be non-empty"))
	}
	return allErrs
}

func ValidateJob(job *experimental.Job) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	// Jobs and rcs have the same name validation
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateJobSpec(spec *experimental.JobSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if spec.Parallelism != nil && *spec.Parallelism < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("parallelism", spec.Parallelism, isNegativeErrorMsg))
	}
	if spec.Completions != nil && *spec.Completions < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("completions", spec.Completions, isNegativeErrorMsg))
	}

	selector := labels.Set(spec.Selector).AsSelector()
	if selector.Empty() {
		allErrs = append(allErrs, errs.NewFieldRequired("selector"))
	}

	if spec.Template == nil {
		allErrs = append(allErrs, errs.NewFieldRequired("template"))
	} else {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, errs.NewFieldInvalid("template.labels", spec.Template.Labels, "selector does not match template"))
		}
		allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(spec.Template).Prefix("template")...)
		if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure &&
			spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
			allErrs = append(allErrs, errs.NewFieldValueNotSupported("template.spec.restartPolicy",
				spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}))
		}
	}
	return allErrs
}

func ValidateJobUpdate(oldJob, job *experimental.Job) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec).Prefix("spec")...)
	return allErrs
}
