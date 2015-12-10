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
	"net"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation"
)

// ValidateHorizontalPodAutoscaler can be used to check whether the given autoscaler name is valid.
// Prefix indicates this name will be used as part of generation, in which case trailing dashes are allowed.
func ValidateHorizontalPodAutoscalerName(name string, prefix bool) (bool, string) {
	// TODO: finally move it to pkg/api/validation and use nameIsDNSSubdomain function
	return apivalidation.ValidateReplicationControllerName(name, prefix)
}

func validateHorizontalPodAutoscalerSpec(autoscaler extensions.HorizontalPodAutoscalerSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if autoscaler.MinReplicas != nil && *autoscaler.MinReplicas < 1 {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("minReplicas"), autoscaler.MinReplicas, `must be greater than or equal to 1`))
	}
	if autoscaler.MaxReplicas < 1 {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("maxReplicas"), autoscaler.MaxReplicas, `must be greater than or equal to 1`))
	}
	if autoscaler.MinReplicas != nil && autoscaler.MaxReplicas < *autoscaler.MinReplicas {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("maxReplicas"), autoscaler.MaxReplicas, `must be greater than or equal to minReplicas`))
	}
	if autoscaler.CPUUtilization != nil && autoscaler.CPUUtilization.TargetPercentage < 1 {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("cpuUtilization", "targetPercentage"), autoscaler.CPUUtilization.TargetPercentage, `must be greater than or equal to 1`))
	}
	if refErrs := ValidateSubresourceReference(autoscaler.ScaleRef, fldPath.Child("scaleRef")); len(refErrs) > 0 {
		allErrs = append(allErrs, refErrs...)
	} else if autoscaler.ScaleRef.Subresource != "scale" {
		allErrs = append(allErrs, validation.NewNotSupportedError(fldPath.Child("scaleRef", "subresource"), autoscaler.ScaleRef.Subresource, []string{"scale"}))
	}
	return allErrs
}

func ValidateSubresourceReference(ref extensions.SubresourceReference, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(ref.Kind) == 0 {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("kind")))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Kind); !ok {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("kind"), ref.Kind, msg))
	}

	if len(ref.Name) == 0 {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("name")))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Name); !ok {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("name"), ref.Name, msg))
	}

	if len(ref.Subresource) == 0 {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("subresource")))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Subresource); !ok {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("subresource"), ref.Subresource, msg))
	}
	return allErrs
}

func ValidateHorizontalPodAutoscaler(autoscaler *extensions.HorizontalPodAutoscaler) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&autoscaler.ObjectMeta, true, ValidateHorizontalPodAutoscalerName, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(autoscaler.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerUpdate(newAutoscaler, oldAutoscaler *extensions.HorizontalPodAutoscaler) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&newAutoscaler.ObjectMeta, &oldAutoscaler.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(newAutoscaler.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerStatusUpdate(controller, oldController *extensions.HorizontalPodAutoscaler) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta, validation.NewFieldPath("metadata"))...)

	status := controller.Status
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.CurrentReplicas), validation.NewFieldPath("status", "currentReplicas"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.DesiredReplicas), validation.NewFieldPath("status", "desiredReplicasa"))...)
	return allErrs
}

func ValidateThirdPartyResourceUpdate(update, old *extensions.ThirdPartyResource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateThirdPartyResource(update)...)
	return allErrs
}

func ValidateThirdPartyResourceName(name string, prefix bool) (bool, string) {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

func ValidateThirdPartyResource(obj *extensions.ThirdPartyResource) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&obj.ObjectMeta, true, ValidateThirdPartyResourceName, validation.NewFieldPath("metadata"))...)

	versions := sets.String{}
	for ix := range obj.Versions {
		version := &obj.Versions[ix]
		if len(version.Name) == 0 {
			allErrs = append(allErrs, validation.NewInvalidError(validation.NewFieldPath("versions").Index(ix).Child("name"), version, "can not be empty"))
		}
		if versions.Has(version.Name) {
			allErrs = append(allErrs, validation.NewDuplicateError(validation.NewFieldPath("versions").Index(ix).Child("name"), version))
		}
		versions.Insert(version.Name)
	}
	return allErrs
}

// ValidateDaemonSet tests if required fields in the DaemonSet are set.
func ValidateDaemonSet(controller *extensions.DaemonSet) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&controller.ObjectMeta, true, ValidateDaemonSetName, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateDaemonSetSpec(&controller.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

// ValidateDaemonSetUpdate tests if required fields in the DaemonSet are set.
func ValidateDaemonSetUpdate(controller, oldController *extensions.DaemonSet) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateDaemonSetSpec(&controller.Spec, validation.NewFieldPath("spec"))...)
	allErrs = append(allErrs, ValidateDaemonSetTemplateUpdate(controller.Spec.Template, oldController.Spec.Template, validation.NewFieldPath("spec", "template"))...)
	return allErrs
}

// validateDaemonSetStatus validates a DaemonSetStatus
func validateDaemonSetStatus(status *extensions.DaemonSetStatus, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.CurrentNumberScheduled), fldPath.Child("currentNumberScheduled"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.NumberMisscheduled), fldPath.Child("numberMisscheduled"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.DesiredNumberScheduled), fldPath.Child("desiredNumberScheduled"))...)
	return allErrs
}

// ValidateDaemonSetStatus validates tests if required fields in the DaemonSet Status section
func ValidateDaemonSetStatusUpdate(controller, oldController *extensions.DaemonSet) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, validateDaemonSetStatus(&controller.Status, validation.NewFieldPath("status"))...)
	return allErrs
}

// ValidateDaemonSetTemplateUpdate tests that certain fields in the daemon set's pod template are not updated.
func ValidateDaemonSetTemplateUpdate(podTemplate, oldPodTemplate *api.PodTemplateSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	podSpec := podTemplate.Spec
	// podTemplate.Spec is not a pointer, so we can modify NodeSelector and NodeName directly.
	podSpec.NodeSelector = oldPodTemplate.Spec.NodeSelector
	podSpec.NodeName = oldPodTemplate.Spec.NodeName
	// In particular, we do not allow updates to container images at this point.
	if !api.Semantic.DeepEqual(oldPodTemplate.Spec, podSpec) {
		// TODO: Pinpoint the specific field that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("spec"), "content of spec is not printed out, please refer to the \"details\"", "may not update fields other than spec.nodeSelector"))
	}
	return allErrs
}

// ValidateDaemonSetSpec tests if required fields in the DaemonSetSpec are set.
func ValidateDaemonSetSpec(spec *extensions.DaemonSetSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}

	allErrs = append(allErrs, ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)

	if spec.Template == nil {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("template")))
		return allErrs
	}

	selector, err := extensions.LabelSelectorAsSelector(spec.Selector)
	if err == nil && !selector.Matches(labels.Set(spec.Template.Labels)) {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("template", "metadata", "labels"), spec.Template.Labels, "selector does not match template"))
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(spec.Template, fldPath.Child("template"))...)
	// Daemons typically run on more than one node, so mark Read-Write persistent disks as invalid.
	allErrs = append(allErrs, apivalidation.ValidateReadOnlyPersistentDisks(spec.Template.Spec.Volumes, fldPath.Child("template", "spec", "volumes"))...)
	// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
	if spec.Template.Spec.RestartPolicy != api.RestartPolicyAlways {
		allErrs = append(allErrs, validation.NewNotSupportedError(fldPath.Child("template", "spec", "restartPolicy"), spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
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

func ValidatePositiveIntOrPercent(intOrPercent intstr.IntOrString, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if intOrPercent.Type == intstr.String {
		if !validation.IsValidPercent(intOrPercent.StrVal) {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath, intOrPercent, "value should be int(5) or percentage(5%)"))
		}
	} else if intOrPercent.Type == intstr.Int {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(intOrPercent.IntValue()), fldPath)...)
	}
	return allErrs
}

func getPercentValue(intOrStringValue intstr.IntOrString) (int, bool) {
	if intOrStringValue.Type != intstr.String || !validation.IsValidPercent(intOrStringValue.StrVal) {
		return 0, false
	}
	value, _ := strconv.Atoi(intOrStringValue.StrVal[:len(intOrStringValue.StrVal)-1])
	return value, true
}

func getIntOrPercentValue(intOrStringValue intstr.IntOrString) int {
	value, isPercent := getPercentValue(intOrStringValue)
	if isPercent {
		return value
	}
	return intOrStringValue.IntValue()
}

func IsNotMoreThan100Percent(intOrStringValue intstr.IntOrString, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	value, isPercent := getPercentValue(intOrStringValue)
	if !isPercent || value <= 100 {
		return nil
	}
	allErrs = append(allErrs, validation.NewInvalidError(fldPath, intOrStringValue, "should not be more than 100%"))
	return allErrs
}

func ValidateRollingUpdateDeployment(rollingUpdate *extensions.RollingUpdateDeployment, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidatePositiveIntOrPercent(rollingUpdate.MaxUnavailable, fldPath.Child("maxUnavailable"))...)
	allErrs = append(allErrs, ValidatePositiveIntOrPercent(rollingUpdate.MaxSurge, fldPath.Child("maxSurge"))...)
	if getIntOrPercentValue(rollingUpdate.MaxUnavailable) == 0 && getIntOrPercentValue(rollingUpdate.MaxSurge) == 0 {
		// Both MaxSurge and MaxUnavailable cannot be zero.
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("maxUnavailable"), rollingUpdate.MaxUnavailable, "cannot be 0 when maxSurge is 0 as well"))
	}
	// Validate that MaxUnavailable is not more than 100%.
	allErrs = append(allErrs, IsNotMoreThan100Percent(rollingUpdate.MaxUnavailable, fldPath.Child("maxUnavailable"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(rollingUpdate.MinReadySeconds), fldPath.Child("minReadySeconds"))...)
	return allErrs
}

func ValidateDeploymentStrategy(strategy *extensions.DeploymentStrategy, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if strategy.RollingUpdate == nil {
		return allErrs
	}
	switch strategy.Type {
	case extensions.RecreateDeploymentStrategyType:
		allErrs = append(allErrs, validation.NewForbiddenError(fldPath.Child("rollingUpdate"), "should be nil when strategy type is "+extensions.RecreateDeploymentStrategyType))
	case extensions.RollingUpdateDeploymentStrategyType:
		allErrs = append(allErrs, ValidateRollingUpdateDeployment(strategy.RollingUpdate, fldPath.Child("rollingUpdate"))...)
	}
	return allErrs
}

// Validates given deployment spec.
func ValidateDeploymentSpec(spec *extensions.DeploymentSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonEmptySelector(spec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(spec.Replicas), fldPath.Child("replicas"))...)
	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpecForRC(&spec.Template, spec.Selector, spec.Replicas, fldPath.Child("template"))...)
	allErrs = append(allErrs, ValidateDeploymentStrategy(&spec.Strategy, fldPath.Child("strategy"))...)
	// empty string is a valid UniqueLabelKey
	if len(spec.UniqueLabelKey) > 0 {
		allErrs = append(allErrs, apivalidation.ValidateLabelName(spec.UniqueLabelKey, fldPath.Child("uniqueLabel"))...)
	}
	return allErrs
}

func ValidateDeploymentUpdate(update, old *extensions.Deployment) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateDeploymentSpec(&update.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateDeployment(obj *extensions.Deployment) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&obj.ObjectMeta, true, ValidateDeploymentName, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateDeploymentSpec(&obj.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateThirdPartyResourceDataUpdate(update, old *extensions.ThirdPartyResourceData) validation.ErrorList {
	return ValidateThirdPartyResourceData(update)
}

func ValidateThirdPartyResourceData(obj *extensions.ThirdPartyResourceData) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(obj.Name) == 0 {
		allErrs = append(allErrs, validation.NewInvalidError(validation.NewFieldPath("name"), obj.Name, "must be non-empty"))
	}
	return allErrs
}

func ValidateJob(job *extensions.Job) validation.ErrorList {
	allErrs := validation.ErrorList{}
	// Jobs and rcs have the same name validation
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateJobSpec(spec *extensions.JobSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}

	if spec.Parallelism != nil {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(*spec.Parallelism), fldPath.Child("parallelism"))...)
	}
	if spec.Completions != nil {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(*spec.Completions), fldPath.Child("completions"))...)
	}
	if spec.Selector == nil {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("selector")))
	} else {
		allErrs = append(allErrs, ValidateLabelSelector(spec.Selector, fldPath.Child("selector"))...)
	}

	if selector, err := extensions.LabelSelectorAsSelector(spec.Selector); err == nil {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("template", "metadata", "labels"), spec.Template.Labels, "selector does not match template"))
		}
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(&spec.Template, fldPath.Child("template"))...)
	if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure &&
		spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, validation.NewNotSupportedError(fldPath.Child("template", "spec", "restartPolicy"),
			spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}))
	}
	return allErrs
}

func ValidateJobStatus(status *extensions.JobStatus, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.Active), fldPath.Child("active"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.Succeeded), fldPath.Child("succeeded"))...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.Failed), fldPath.Child("failed"))...)
	return allErrs
}

func ValidateJobUpdate(job, oldJob *extensions.Job) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateJobSpecUpdate(job.Spec, oldJob.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateJobUpdateStatus(job, oldJob *extensions.Job) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateJobStatusUpdate(job.Status, oldJob.Status)...)
	return allErrs
}

func ValidateJobSpecUpdate(spec, oldSpec extensions.JobSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateJobSpec(&spec, fldPath)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, fldPath.Child("completions"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Selector, oldSpec.Selector, fldPath.Child("selector"))...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Template, oldSpec.Template, fldPath.Child("template"))...)
	return allErrs
}

func ValidateJobStatusUpdate(status, oldStatus extensions.JobStatus) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, ValidateJobStatus(&status, validation.NewFieldPath("status"))...)
	return allErrs
}

// ValidateIngress tests if required fields in the Ingress are set.
func ValidateIngress(ingress *extensions.Ingress) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&ingress.ObjectMeta, true, ValidateIngressName, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

// ValidateIngressName validates that the given name can be used as an Ingress name.
func ValidateIngressName(name string, prefix bool) (bool, string) {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidateIngressSpec tests if required fields in the IngressSpec are set.
func ValidateIngressSpec(spec *extensions.IngressSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	// TODO: Is a default backend mandatory?
	if spec.Backend != nil {
		allErrs = append(allErrs, validateIngressBackend(spec.Backend, fldPath.Child("backend"))...)
	} else if len(spec.Rules) == 0 {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("rules"), spec.Rules, "Either a default backend or a set of host rules are required for ingress."))
	}
	if len(spec.Rules) > 0 {
		allErrs = append(allErrs, validateIngressRules(spec.Rules, fldPath.Child("rules"))...)
	}
	return allErrs
}

// ValidateIngressUpdate tests if required fields in the Ingress are set.
func ValidateIngressUpdate(ingress, oldIngress *extensions.Ingress) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

// ValidateIngressStatusUpdate tests if required fields in the Ingress are set when updating status.
func ValidateIngressStatusUpdate(ingress, oldIngress *extensions.Ingress) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta, validation.NewFieldPath("metadata"))...)
	allErrs = append(allErrs, apivalidation.ValidateLoadBalancerStatus(&ingress.Status.LoadBalancer, validation.NewFieldPath("status", "loadBalancer"))...)
	return allErrs
}

func validateIngressRules(IngressRules []extensions.IngressRule, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(IngressRules) == 0 {
		return append(allErrs, validation.NewRequiredError(fldPath))
	}
	for i, ih := range IngressRules {
		if len(ih.Host) > 0 {
			// TODO: Ports and ips are allowed in the host part of a url
			// according to RFC 3986, consider allowing them.
			if valid, errMsg := apivalidation.NameIsDNSSubdomain(ih.Host, false); !valid {
				allErrs = append(allErrs, validation.NewInvalidError(fldPath.Index(i).Child("host"), ih.Host, errMsg))
			}
			if isIP := (net.ParseIP(ih.Host) != nil); isIP {
				allErrs = append(allErrs, validation.NewInvalidError(fldPath.Index(i).Child("host"), ih.Host, "Host must be a DNS name, not ip address"))
			}
		}
		allErrs = append(allErrs, validateIngressRuleValue(&ih.IngressRuleValue, fldPath.Index(0))...)
	}
	return allErrs
}

func validateIngressRuleValue(ingressRule *extensions.IngressRuleValue, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if ingressRule.HTTP != nil {
		allErrs = append(allErrs, validateHTTPIngressRuleValue(ingressRule.HTTP, fldPath.Child("http"))...)
	}
	return allErrs
}

func validateHTTPIngressRuleValue(httpIngressRuleValue *extensions.HTTPIngressRuleValue, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if len(httpIngressRuleValue.Paths) == 0 {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("paths")))
	}
	for i, rule := range httpIngressRuleValue.Paths {
		if len(rule.Path) > 0 {
			if !strings.HasPrefix(rule.Path, "/") {
				allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("paths").Index(i).Child("path"), rule.Path, "must begin with /"))
			}
			// TODO: More draconian path regex validation.
			// Path must be a valid regex. This is the basic requirement.
			// In addition to this any characters not allowed in a path per
			// RFC 3986 section-3.3 cannot appear as a literal in the regex.
			// Consider the example: http://host/valid?#bar, everything after
			// the last '/' is a valid regex that matches valid#bar, which
			// isn't a valid path, because the path terminates at the first ?
			// or #. A more sophisticated form of validation would detect that
			// the user is confusing url regexes with path regexes.
			_, err := regexp.CompilePOSIX(rule.Path)
			if err != nil {
				allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("paths").Index(i).Child("path"), rule.Path, "must be a valid regex."))
			}
		}
		allErrs = append(allErrs, validateIngressBackend(&rule.Backend, fldPath.Child("backend"))...)
	}
	return allErrs
}

// validateIngressBackend tests if a given backend is valid.
func validateIngressBackend(backend *extensions.IngressBackend, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}

	// All backends must reference a single local service by name, and a single service port by name or number.
	if len(backend.ServiceName) == 0 {
		return append(allErrs, validation.NewRequiredError(fldPath.Child("serviceName")))
	} else if ok, errMsg := apivalidation.ValidateServiceName(backend.ServiceName, false); !ok {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("serviceName"), backend.ServiceName, errMsg))
	}
	if backend.ServicePort.Type == intstr.String {
		if !validation.IsDNS1123Label(backend.ServicePort.StrVal) {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("servicePort"), backend.ServicePort.StrVal, apivalidation.DNS1123LabelErrorMsg))
		}
		if !validation.IsValidPortName(backend.ServicePort.StrVal) {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("servicePort"), backend.ServicePort.StrVal, apivalidation.PortNameErrorMsg))
		}
	} else if !validation.IsValidPortNum(backend.ServicePort.IntValue()) {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("servicePort"), backend.ServicePort, apivalidation.PortRangeErrorMsg))
	}
	return allErrs
}

func validateClusterAutoscalerSpec(spec extensions.ClusterAutoscalerSpec, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if spec.MinNodes < 0 {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("minNodes"), spec.MinNodes, `must be non-negative`))
	}
	if spec.MaxNodes < spec.MinNodes {
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("maxNodes"), spec.MaxNodes, `must be greater than or equal to minNodes`))
	}
	if len(spec.TargetUtilization) == 0 {
		allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("targetUtilization")))
	}
	for _, target := range spec.TargetUtilization {
		if len(target.Resource) == 0 {
			allErrs = append(allErrs, validation.NewRequiredError(fldPath.Child("targetUtilization", "resource")))
		}
		if target.Value <= 0 {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("targetUtilization", "value"), target.Value, "must be greater than 0"))
		}
		if target.Value > 1 {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("targetUtilization", "value"), target.Value, "must be less or equal 1"))
		}
	}
	return allErrs
}

func ValidateClusterAutoscaler(autoscaler *extensions.ClusterAutoscaler) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if autoscaler.Name != "ClusterAutoscaler" {
		allErrs = append(allErrs, validation.NewInvalidError(validation.NewFieldPath("metadata", "name"), autoscaler.Name, `name must be ClusterAutoscaler`))
	}
	if autoscaler.Namespace != api.NamespaceDefault {
		allErrs = append(allErrs, validation.NewInvalidError(validation.NewFieldPath("metadata", "namespace"), autoscaler.Namespace, `namespace must be default`))
	}
	allErrs = append(allErrs, validateClusterAutoscalerSpec(autoscaler.Spec, validation.NewFieldPath("spec"))...)
	return allErrs
}

func ValidateLabelSelector(ps *extensions.LabelSelector, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	if ps == nil {
		return allErrs
	}
	allErrs = append(allErrs, apivalidation.ValidateLabels(ps.MatchLabels, fldPath.Child("matchLabels"))...)
	for i, expr := range ps.MatchExpressions {
		allErrs = append(allErrs, ValidateLabelSelectorRequirement(expr, fldPath.Child("matchExpressions").Index(i))...)
	}
	return allErrs
}

func ValidateLabelSelectorRequirement(sr extensions.LabelSelectorRequirement, fldPath *validation.FieldPath) validation.ErrorList {
	allErrs := validation.ErrorList{}
	switch sr.Operator {
	case extensions.LabelSelectorOpIn, extensions.LabelSelectorOpNotIn:
		if len(sr.Values) == 0 {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("values"), sr.Values, "must be non-empty when operator is In or NotIn"))
		}
	case extensions.LabelSelectorOpExists, extensions.LabelSelectorOpDoesNotExist:
		if len(sr.Values) > 0 {
			allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("values"), sr.Values, "must be empty when operator is Exists or DoesNotExist"))
		}
	default:
		allErrs = append(allErrs, validation.NewInvalidError(fldPath.Child("operator"), sr.Operator, "not a valid pod selector operator"))
	}
	allErrs = append(allErrs, apivalidation.ValidateLabelName(sr.Key, fldPath.Child("key"))...)
	return allErrs
}

func ValidateScale(scale *extensions.Scale) validation.ErrorList {
	allErrs := validation.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&scale.ObjectMeta, true, apivalidation.NameIsDNSSubdomain, validation.NewFieldPath("metadata"))...)

	if scale.Spec.Replicas < 0 {
		allErrs = append(allErrs, validation.NewInvalidError(validation.NewFieldPath("spec", "replicas"), scale.Spec.Replicas, "must be non-negative"))
	}

	return allErrs
}
