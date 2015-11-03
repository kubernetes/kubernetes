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
	"fmt"
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

func validateHorizontalPodAutoscalerSpec(autoscaler extensions.HorizontalPodAutoscalerSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if autoscaler.MinReplicas != nil && *autoscaler.MinReplicas < 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("minReplicas", autoscaler.MinReplicas, `must be bigger or equal to 1`))
	}
	if autoscaler.MaxReplicas < 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("maxReplicas", autoscaler.MaxReplicas, `must be bigger or equal to 1`))
	}
	if autoscaler.MinReplicas != nil && autoscaler.MaxReplicas < *autoscaler.MinReplicas {
		allErrs = append(allErrs, validation.NewFieldInvalid("maxReplicas", autoscaler.MaxReplicas, `must be bigger or equal to minReplicas`))
	}
	if autoscaler.CPUUtilization != nil && autoscaler.CPUUtilization.TargetPercentage < 1 {
		allErrs = append(allErrs, validation.NewFieldInvalid("cpuUtilization.targetPercentage", autoscaler.CPUUtilization.TargetPercentage, `must be bigger or equal to 1`))
	}
	if refErrs := ValidateSubresourceReference(autoscaler.ScaleRef); len(refErrs) > 0 {
		allErrs = append(allErrs, refErrs.Prefix("scaleRef")...)
	} else if autoscaler.ScaleRef.Subresource != "scale" {
		allErrs = append(allErrs, validation.NewFieldValueNotSupported("scaleRef.subresource", autoscaler.ScaleRef.Subresource, []string{"scale"}))
	}
	return allErrs
}

func ValidateSubresourceReference(ref extensions.SubresourceReference) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if len(ref.Kind) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("kind"))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Kind); !ok {
		allErrs = append(allErrs, validation.NewFieldInvalid("kind", ref.Kind, msg))
	}

	if len(ref.Name) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("name"))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Name); !ok {
		allErrs = append(allErrs, validation.NewFieldInvalid("name", ref.Name, msg))
	}

	if len(ref.Subresource) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("subresource"))
	} else if ok, msg := apivalidation.IsValidPathSegmentName(ref.Subresource); !ok {
		allErrs = append(allErrs, validation.NewFieldInvalid("subresource", ref.Subresource, msg))
	}
	return allErrs
}

func ValidateHorizontalPodAutoscaler(autoscaler *extensions.HorizontalPodAutoscaler) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&autoscaler.ObjectMeta, true, ValidateHorizontalPodAutoscalerName).Prefix("metadata")...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(autoscaler.Spec)...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerUpdate(newAutoscler, oldAutoscaler *extensions.HorizontalPodAutoscaler) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&newAutoscler.ObjectMeta, &oldAutoscaler.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, validateHorizontalPodAutoscalerSpec(newAutoscler.Spec)...)
	return allErrs
}

func ValidateHorizontalPodAutoscalerStatusUpdate(controller, oldController *extensions.HorizontalPodAutoscaler) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta).Prefix("metadata")...)

	status := controller.Status
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.CurrentReplicas), "currentReplicas")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.DesiredReplicas), "desiredReplicas")...)
	return allErrs
}

func ValidateThirdPartyResourceUpdate(update, old *extensions.ThirdPartyResource) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateThirdPartyResource(update)...)
	return allErrs
}

func ValidateThirdPartyResourceName(name string, prefix bool) (bool, string) {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

func ValidateThirdPartyResource(obj *extensions.ThirdPartyResource) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&obj.ObjectMeta, true, ValidateThirdPartyResourceName).Prefix("metadata")...)

	versions := sets.String{}
	for ix := range obj.Versions {
		version := &obj.Versions[ix]
		if len(version.Name) == 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("name", version, "name can not be empty"))
		}
		if versions.Has(version.Name) {
			allErrs = append(allErrs, validation.NewFieldDuplicate("version", version))
		}
		versions.Insert(version.Name)
	}
	return allErrs
}

// ValidateDaemonSet tests if required fields in the DaemonSet are set.
func ValidateDaemonSet(controller *extensions.DaemonSet) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&controller.ObjectMeta, true, ValidateDaemonSetName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDaemonSetSpec(&controller.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateDaemonSetUpdate tests if required fields in the DaemonSet are set.
func ValidateDaemonSetUpdate(controller, oldController *extensions.DaemonSet) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDaemonSetSpec(&controller.Spec).Prefix("spec")...)
	allErrs = append(allErrs, ValidateDaemonSetTemplateUpdate(controller.Spec.Template, oldController.Spec.Template).Prefix("spec.template")...)
	return allErrs
}

// validateDaemonSetStatus validates a DaemonSetStatus
func validateDaemonSetStatus(status *extensions.DaemonSetStatus) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.CurrentNumberScheduled), "currentNumberScheduled")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.NumberMisscheduled), "numberMisscheduled")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.DesiredNumberScheduled), "desiredNumberScheduled")...)
	return allErrs
}

// ValidateDaemonSetStatus validates tests if required fields in the DaemonSet Status section
func ValidateDaemonSetStatusUpdate(controller, oldController *extensions.DaemonSet) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&controller.ObjectMeta, &oldController.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, validateDaemonSetStatus(&controller.Status)...)
	return allErrs
}

// ValidateDaemonSetTemplateUpdate tests that certain fields in the daemon set's pod template are not updated.
func ValidateDaemonSetTemplateUpdate(podTemplate, oldPodTemplate *api.PodTemplateSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	podSpec := podTemplate.Spec
	// podTemplate.Spec is not a pointer, so we can modify NodeSelector and NodeName directly.
	podSpec.NodeSelector = oldPodTemplate.Spec.NodeSelector
	podSpec.NodeName = oldPodTemplate.Spec.NodeName
	// In particular, we do not allow updates to container images at this point.
	if !api.Semantic.DeepEqual(oldPodTemplate.Spec, podSpec) {
		// TODO: Pinpoint the specific field that causes the invalid error after we have strategic merge diff
		allErrs = append(allErrs, validation.NewFieldInvalid("spec", "content of spec is not printed out, please refer to the \"details\"", "may not update fields other than spec.nodeSelector"))
	}
	return allErrs
}

// ValidateDaemonSetSpec tests if required fields in the DaemonSetSpec are set.
func ValidateDaemonSetSpec(spec *extensions.DaemonSetSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}

	allErrs = append(allErrs, ValidatePodSelector(spec.Selector)...)

	if spec.Template == nil {
		allErrs = append(allErrs, validation.NewFieldRequired("template"))
		return allErrs
	}

	selector, err := extensions.PodSelectorAsSelector(spec.Selector)
	if err == nil && !selector.Matches(labels.Set(spec.Template.Labels)) {
		allErrs = append(allErrs, validation.NewFieldInvalid("template.metadata.labels", spec.Template.Labels, "selector does not match template"))
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(spec.Template).Prefix("template")...)
	// Daemons typically run on more than one node, so mark Read-Write persistent disks as invalid.
	allErrs = append(allErrs, apivalidation.ValidateReadOnlyPersistentDisks(spec.Template.Spec.Volumes).Prefix("template.spec.volumes")...)
	// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
	if spec.Template.Spec.RestartPolicy != api.RestartPolicyAlways {
		allErrs = append(allErrs, validation.NewFieldValueNotSupported("template.spec.restartPolicy", spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyAlways)}))
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

func ValidatePositiveIntOrPercent(intOrPercent intstr.IntOrString, fieldName string) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if intOrPercent.Type == intstr.String {
		if !validation.IsValidPercent(intOrPercent.StrVal) {
			allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, intOrPercent, "value should be int(5) or percentage(5%)"))
		}
	} else if intOrPercent.Type == intstr.Int {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(intOrPercent.IntVal), fieldName)...)
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
	return intOrStringValue.IntVal
}

func IsNotMoreThan100Percent(intOrStringValue intstr.IntOrString, fieldName string) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	value, isPercent := getPercentValue(intOrStringValue)
	if !isPercent || value <= 100 {
		return nil
	}
	allErrs = append(allErrs, validation.NewFieldInvalid(fieldName, intOrStringValue, "should not be more than 100%"))
	return allErrs
}

func ValidateRollingUpdateDeployment(rollingUpdate *extensions.RollingUpdateDeployment, fieldName string) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, ValidatePositiveIntOrPercent(rollingUpdate.MaxUnavailable, fieldName+"maxUnavailable")...)
	allErrs = append(allErrs, ValidatePositiveIntOrPercent(rollingUpdate.MaxSurge, fieldName+".maxSurge")...)
	if getIntOrPercentValue(rollingUpdate.MaxUnavailable) == 0 && getIntOrPercentValue(rollingUpdate.MaxSurge) == 0 {
		// Both MaxSurge and MaxUnavailable cannot be zero.
		allErrs = append(allErrs, validation.NewFieldInvalid(fieldName+".maxUnavailable", rollingUpdate.MaxUnavailable, "cannot be 0 when maxSurge is 0 as well"))
	}
	// Validate that MaxUnavailable is not more than 100%.
	allErrs = append(allErrs, IsNotMoreThan100Percent(rollingUpdate.MaxUnavailable, fieldName+".maxUnavailable")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(rollingUpdate.MinReadySeconds), fieldName+".minReadySeconds")...)
	return allErrs
}

func ValidateDeploymentStrategy(strategy *extensions.DeploymentStrategy, fieldName string) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if strategy.RollingUpdate == nil {
		return allErrs
	}
	switch strategy.Type {
	case extensions.RecreateDeploymentStrategyType:
		allErrs = append(allErrs, validation.NewFieldForbidden("rollingUpdate", "rollingUpdate should be nil when strategy type is "+extensions.RecreateDeploymentStrategyType))
	case extensions.RollingUpdateDeploymentStrategyType:
		allErrs = append(allErrs, ValidateRollingUpdateDeployment(strategy.RollingUpdate, "rollingUpdate")...)
	}
	return allErrs
}

// Validates given deployment spec.
func ValidateDeploymentSpec(spec *extensions.DeploymentSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonEmptySelector(spec.Selector, "selector")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(spec.Replicas), "replicas")...)
	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpecForRC(&spec.Template, spec.Selector, spec.Replicas, "template")...)
	allErrs = append(allErrs, ValidateDeploymentStrategy(&spec.Strategy, "strategy")...)
	allErrs = append(allErrs, apivalidation.ValidateLabelName(spec.UniqueLabelKey, "uniqueLabel")...)
	return allErrs
}

func ValidateDeploymentUpdate(update, old *extensions.Deployment) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&update.ObjectMeta, &old.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDeploymentSpec(&update.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateDeployment(obj *extensions.Deployment) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&obj.ObjectMeta, true, ValidateDeploymentName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateDeploymentSpec(&obj.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateThirdPartyResourceDataUpdate(update, old *extensions.ThirdPartyResourceData) validation.ValidationErrorList {
	return ValidateThirdPartyResourceData(update)
}

func ValidateThirdPartyResourceData(obj *extensions.ThirdPartyResourceData) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if len(obj.Name) == 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("name", obj.Name, "name must be non-empty"))
	}
	return allErrs
}

func ValidateJob(job *extensions.Job) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	// Jobs and rcs have the same name validation
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&job.ObjectMeta, true, apivalidation.ValidateReplicationControllerName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateJobSpec(&job.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateJobSpec(spec *extensions.JobSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}

	if spec.Parallelism != nil {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(*spec.Parallelism), "parallelism")...)
	}
	if spec.Completions != nil {
		allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(*spec.Completions), "completions")...)
	}
	if spec.Selector == nil {
		allErrs = append(allErrs, validation.NewFieldRequired("selector"))
	} else {
		allErrs = append(allErrs, ValidatePodSelector(spec.Selector).Prefix("selector")...)
	}

	if selector, err := extensions.PodSelectorAsSelector(spec.Selector); err == nil {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, validation.NewFieldInvalid("template.metadata.labels", spec.Template.Labels, "selector does not match template"))
		}
	}

	allErrs = append(allErrs, apivalidation.ValidatePodTemplateSpec(&spec.Template).Prefix("template")...)
	if spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure &&
		spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
		allErrs = append(allErrs, validation.NewFieldValueNotSupported("template.spec.restartPolicy",
			spec.Template.Spec.RestartPolicy, []string{string(api.RestartPolicyOnFailure), string(api.RestartPolicyNever)}))
	}
	return allErrs
}

func ValidateJobStatus(status *extensions.JobStatus) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.Active), "active")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.Succeeded), "succeeded")...)
	allErrs = append(allErrs, apivalidation.ValidatePositiveField(int64(status.Failed), "failed")...)
	return allErrs
}

func ValidateJobUpdate(job, oldJob *extensions.Job) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateJobSpecUpdate(job.Spec, oldJob.Spec).Prefix("spec")...)
	return allErrs
}

func ValidateJobUpdateStatus(job, oldJob *extensions.Job) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&oldJob.ObjectMeta, &job.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateJobStatusUpdate(job.Status, oldJob.Status).Prefix("status")...)
	return allErrs
}

func ValidateJobSpecUpdate(spec, oldSpec extensions.JobSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, ValidateJobSpec(&spec)...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Completions, oldSpec.Completions, "completions")...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Selector, oldSpec.Selector, "selector")...)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(spec.Template, oldSpec.Template, "template")...)
	return allErrs
}

func ValidateJobStatusUpdate(status, oldStatus extensions.JobStatus) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, ValidateJobStatus(&status)...)
	return allErrs
}

// ValidateIngress tests if required fields in the Ingress are set.
func ValidateIngress(ingress *extensions.Ingress) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&ingress.ObjectMeta, true, ValidateIngressName).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateIngressName validates that the given name can be used as an Ingress name.
func ValidateIngressName(name string, prefix bool) (bool, string) {
	return apivalidation.NameIsDNSSubdomain(name, prefix)
}

// ValidateIngressSpec tests if required fields in the IngressSpec are set.
func ValidateIngressSpec(spec *extensions.IngressSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	// TODO: Is a default backend mandatory?
	if spec.Backend != nil {
		allErrs = append(allErrs, validateIngressBackend(spec.Backend).Prefix("backend")...)
	} else if len(spec.Rules) == 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("rules", spec.Rules, "Either a default backend or a set of host rules are required for ingress."))
	}
	if len(spec.Rules) > 0 {
		allErrs = append(allErrs, validateIngressRules(spec.Rules).Prefix("rules")...)
	}
	return allErrs
}

// ValidateIngressUpdate tests if required fields in the Ingress are set.
func ValidateIngressUpdate(ingress, oldIngress *extensions.Ingress) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateIngressSpec(&ingress.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateIngressStatusUpdate tests if required fields in the Ingress are set when updating status.
func ValidateIngressStatusUpdate(ingress, oldIngress *extensions.Ingress) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMetaUpdate(&ingress.ObjectMeta, &oldIngress.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, apivalidation.ValidateLoadBalancerStatus(&ingress.Status.LoadBalancer).Prefix("status.loadBalancer")...)
	return allErrs
}

func validateIngressRules(IngressRules []extensions.IngressRule) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if len(IngressRules) == 0 {
		return append(allErrs, validation.NewFieldRequired("IngressRules"))
	}
	for _, ih := range IngressRules {
		if len(ih.Host) > 0 {
			// TODO: Ports and ips are allowed in the host part of a url
			// according to RFC 3986, consider allowing them.
			if valid, errMsg := apivalidation.NameIsDNSSubdomain(ih.Host, false); !valid {
				allErrs = append(allErrs, validation.NewFieldInvalid("host", ih.Host, errMsg))
			}
			if isIP := (net.ParseIP(ih.Host) != nil); isIP {
				allErrs = append(allErrs, validation.NewFieldInvalid("host", ih.Host, "Host must be a DNS name, not ip address"))
			}
		}
		allErrs = append(allErrs, validateIngressRuleValue(&ih.IngressRuleValue).Prefix("ingressRule")...)
	}
	return allErrs
}

func validateIngressRuleValue(ingressRule *extensions.IngressRuleValue) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if ingressRule.HTTP != nil {
		allErrs = append(allErrs, validateHTTPIngressRuleValue(ingressRule.HTTP).Prefix("http")...)
	}
	return allErrs
}

func validateHTTPIngressRuleValue(httpIngressRuleValue *extensions.HTTPIngressRuleValue) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if len(httpIngressRuleValue.Paths) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("paths"))
	}
	for _, rule := range httpIngressRuleValue.Paths {
		if len(rule.Path) > 0 {
			if !strings.HasPrefix(rule.Path, "/") {
				allErrs = append(allErrs, validation.NewFieldInvalid("path", rule.Path, "path must begin with /"))
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
				allErrs = append(allErrs, validation.NewFieldInvalid("path", rule.Path, "httpIngressRuleValue.path must be a valid regex."))
			}
		}
		allErrs = append(allErrs, validateIngressBackend(&rule.Backend).Prefix("backend")...)
	}
	return allErrs
}

// validateIngressBackend tests if a given backend is valid.
func validateIngressBackend(backend *extensions.IngressBackend) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}

	// All backends must reference a single local service by name, and a single service port by name or number.
	if len(backend.ServiceName) == 0 {
		return append(allErrs, validation.NewFieldRequired("serviceName"))
	} else if ok, errMsg := apivalidation.ValidateServiceName(backend.ServiceName, false); !ok {
		allErrs = append(allErrs, validation.NewFieldInvalid("serviceName", backend.ServiceName, errMsg))
	}
	if backend.ServicePort.Type == intstr.String {
		if !validation.IsDNS1123Label(backend.ServicePort.StrVal) {
			allErrs = append(allErrs, validation.NewFieldInvalid("servicePort", backend.ServicePort.StrVal, apivalidation.DNS1123LabelErrorMsg))
		}
		if !validation.IsValidPortName(backend.ServicePort.StrVal) {
			allErrs = append(allErrs, validation.NewFieldInvalid("servicePort", backend.ServicePort.StrVal, apivalidation.PortNameErrorMsg))
		}
	} else if !validation.IsValidPortNum(backend.ServicePort.IntVal) {
		allErrs = append(allErrs, validation.NewFieldInvalid("servicePort", backend.ServicePort, apivalidation.PortRangeErrorMsg))
	}
	return allErrs
}

func validateClusterAutoscalerSpec(spec extensions.ClusterAutoscalerSpec) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if spec.MinNodes < 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("minNodes", spec.MinNodes, `must be non-negative`))
	}
	if spec.MaxNodes < spec.MinNodes {
		allErrs = append(allErrs, validation.NewFieldInvalid("maxNodes", spec.MaxNodes, `must be bigger or equal to minNodes`))
	}
	if len(spec.TargetUtilization) == 0 {
		allErrs = append(allErrs, validation.NewFieldRequired("targetUtilization"))
	}
	for _, target := range spec.TargetUtilization {
		if len(target.Resource) == 0 {
			allErrs = append(allErrs, validation.NewFieldRequired("targetUtilization.resource"))
		}
		if target.Value <= 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("targetUtilization.value", target.Value, "must be greater than 0"))
		}
		if target.Value > 1 {
			allErrs = append(allErrs, validation.NewFieldInvalid("targetUtilization.value", target.Value, "must be less or equal 1"))
		}
	}
	return allErrs
}

func ValidateClusterAutoscaler(autoscaler *extensions.ClusterAutoscaler) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if autoscaler.Name != "ClusterAutoscaler" {
		allErrs = append(allErrs, validation.NewFieldInvalid("name", autoscaler.Name, `name must be ClusterAutoscaler`))
	}
	if autoscaler.Namespace != api.NamespaceDefault {
		allErrs = append(allErrs, validation.NewFieldInvalid("namespace", autoscaler.Namespace, `namespace must be default`))
	}
	allErrs = append(allErrs, validateClusterAutoscalerSpec(autoscaler.Spec)...)
	return allErrs
}

func ValidatePodSelector(ps *extensions.PodSelector) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	if ps == nil {
		return allErrs
	}
	allErrs = append(allErrs, apivalidation.ValidateLabels(ps.MatchLabels, "matchLabels")...)
	for i, expr := range ps.MatchExpressions {
		allErrs = append(allErrs, ValidatePodSelectorRequirement(expr).Prefix(fmt.Sprintf("matchExpressions.[%v]", i))...)
	}
	return allErrs
}

func ValidatePodSelectorRequirement(sr extensions.PodSelectorRequirement) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	switch sr.Operator {
	case extensions.PodSelectorOpIn, extensions.PodSelectorOpNotIn:
		if len(sr.Values) == 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("values", sr.Values, "must be non-empty when operator is In or NotIn"))
		}
	case extensions.PodSelectorOpExists, extensions.PodSelectorOpDoesNotExist:
		if len(sr.Values) > 0 {
			allErrs = append(allErrs, validation.NewFieldInvalid("values", sr.Values, "must be empty when operator is Exists or DoesNotExist"))
		}
	default:
		allErrs = append(allErrs, validation.NewFieldInvalid("operator", sr.Operator, "not a valid pod selector operator"))
	}
	allErrs = append(allErrs, apivalidation.ValidateLabelName(sr.Key, "key")...)
	return allErrs
}

func ValidateScale(scale *extensions.Scale) validation.ValidationErrorList {
	allErrs := validation.ValidationErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&scale.ObjectMeta, true, apivalidation.NameIsDNSSubdomain).Prefix("metadata")...)

	if scale.Spec.Replicas < 0 {
		allErrs = append(allErrs, validation.NewFieldInvalid("spec.replicas", scale.Spec.Replicas, "must be non-negative"))
	}

	return allErrs
}
