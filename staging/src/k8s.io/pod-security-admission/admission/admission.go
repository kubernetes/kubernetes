/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"time"

	"k8s.io/klog/v2"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	admissionapi "k8s.io/pod-security-admission/admission/api"
	"k8s.io/pod-security-admission/admission/api/validation"
	"k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/metrics"
	"k8s.io/pod-security-admission/policy"
)

const (
	namespaceMaxPodsToCheck  = 3000
	namespacePodCheckTimeout = 1 * time.Second
)

// Admission implements the core admission logic for the Pod Security Admission controller.
// The admission logic can be
type Admission struct {
	Configuration *admissionapi.PodSecurityConfiguration

	// Getting policy checks per level/version
	Evaluator policy.Evaluator

	// Metrics
	Metrics metrics.EvaluationRecorder

	// Arbitrary object --> PodSpec
	PodSpecExtractor PodSpecExtractor

	// API connections
	NamespaceGetter NamespaceGetter
	PodLister       PodLister

	defaultPolicy api.Policy
}

type NamespaceGetter interface {
	GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error)
}

type PodLister interface {
	ListPods(ctx context.Context, namespace string) ([]*corev1.Pod, error)
}

// PodSpecExtractor extracts a PodSpec from pod-controller resources that embed a PodSpec.
// This interface can be extended to enforce policy on CRDs for custom pod-controllers.
type PodSpecExtractor interface {
	// HasPodSpec returns true if the given resource type MAY contain an extractable PodSpec.
	HasPodSpec(schema.GroupResource) bool
	// ExtractPodSpec returns a pod spec and metadata to evaluate from the object.
	// An error returned here does not block admission of the pod-spec-containing object and is not returned to the user.
	// If the object has no pod spec, return `nil, nil, nil`.
	ExtractPodSpec(runtime.Object) (*metav1.ObjectMeta, *corev1.PodSpec, error)
}

var defaultPodSpecResources = map[schema.GroupResource]bool{
	corev1.Resource("pods"):                   true,
	corev1.Resource("replicationcontrollers"): true,
	corev1.Resource("podtemplates"):           true,
	appsv1.Resource("replicasets"):            true,
	appsv1.Resource("deployments"):            true,
	appsv1.Resource("statefulsets"):           true,
	appsv1.Resource("daemonsets"):             true,
	batchv1.Resource("jobs"):                  true,
	batchv1.Resource("cronjobs"):              true,
}

type DefaultPodSpecExtractor struct{}

func (DefaultPodSpecExtractor) HasPodSpec(gr schema.GroupResource) bool {
	return defaultPodSpecResources[gr]
}

func (DefaultPodSpecExtractor) ExtractPodSpec(obj runtime.Object) (*metav1.ObjectMeta, *corev1.PodSpec, error) {
	switch o := obj.(type) {
	case *corev1.Pod:
		return &o.ObjectMeta, &o.Spec, nil
	case *corev1.PodTemplate:
		return extractPodSpecFromTemplate(&o.Template)
	case *corev1.ReplicationController:
		return extractPodSpecFromTemplate(o.Spec.Template)
	case *appsv1.ReplicaSet:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *appsv1.Deployment:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *appsv1.DaemonSet:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *appsv1.StatefulSet:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *batchv1.Job:
		return extractPodSpecFromTemplate(&o.Spec.Template)
	case *batchv1.CronJob:
		return extractPodSpecFromTemplate(&o.Spec.JobTemplate.Spec.Template)
	default:
		return nil, nil, fmt.Errorf("unexpected object type: %s", obj.GetObjectKind().GroupVersionKind().String())
	}
}

func (DefaultPodSpecExtractor) PodSpecResources() []schema.GroupResource {
	retval := make([]schema.GroupResource, 0, len(defaultPodSpecResources))
	for r := range defaultPodSpecResources {
		retval = append(retval, r)
	}
	return retval
}

func extractPodSpecFromTemplate(template *corev1.PodTemplateSpec) (*metav1.ObjectMeta, *corev1.PodSpec, error) {
	if template == nil {
		return nil, nil, nil
	}
	return &template.ObjectMeta, &template.Spec, nil
}

// CompleteConfiguration() sets up default or derived configuration.
func (a *Admission) CompleteConfiguration() error {
	if a.Configuration != nil {
		if p, err := admissionapi.ToPolicy(a.Configuration.Defaults); err != nil {
			return err
		} else {
			a.defaultPolicy = p
		}
	}

	if a.PodSpecExtractor == nil {
		a.PodSpecExtractor = &DefaultPodSpecExtractor{}
	}

	return nil
}

// ValidateConfiguration() ensures all required fields are set with valid values.
func (a *Admission) ValidateConfiguration() error {
	if a.Configuration == nil {
		return fmt.Errorf("configuration required")
	} else if errs := validation.ValidatePodSecurityConfiguration(a.Configuration); len(errs) > 0 {
		return errs.ToAggregate()
	} else {
		if p, err := admissionapi.ToPolicy(a.Configuration.Defaults); err != nil {
			return err
		} else if !reflect.DeepEqual(p, a.defaultPolicy) {
			return fmt.Errorf("default policy does not match; CompleteConfiguration() was not called before ValidateConfiguration()")
		}
	}
	// TODO: check metrics is non-nil?
	if a.PodSpecExtractor == nil {
		return fmt.Errorf("PodSpecExtractor required")
	}
	if a.Evaluator == nil {
		return fmt.Errorf("Evaluator required")
	}
	if a.NamespaceGetter == nil {
		return fmt.Errorf("NamespaceGetter required")
	}
	if a.PodLister == nil {
		return fmt.Errorf("PodLister required")
	}
	return nil
}

var (
	namespacesResource = corev1.Resource("namespaces")
	podsResource       = corev1.Resource("pods")
)

// Validate admits an API request.
// The objects in admission attributes are expected to be external v1 objects that we care about.
// The returned response may be shared and must not be mutated.
func (a *Admission) Validate(ctx context.Context, attrs Attributes) *admissionv1.AdmissionResponse {
	var response *admissionv1.AdmissionResponse
	switch attrs.GetResource().GroupResource() {
	case namespacesResource:
		response = a.ValidateNamespace(ctx, attrs)
	case podsResource:
		response = a.ValidatePod(ctx, attrs)
	default:
		response = a.ValidatePodController(ctx, attrs)
	}

	// TODO: record metrics.

	return response
}

// ValidateNamespace evaluates a namespace create or update request to ensure the pod security labels are valid,
// and checks existing pods in the namespace for violations of the new policy when updating the enforce level on a namespace.
// The returned response may be shared between evaluations and must not be mutated.
func (a *Admission) ValidateNamespace(ctx context.Context, attrs Attributes) *admissionv1.AdmissionResponse {
	// short-circuit on subresources
	if attrs.GetSubresource() != "" {
		return sharedAllowedResponse()
	}
	obj, err := attrs.GetObject()
	if err != nil {
		klog.ErrorS(err, "failed to get object")
		return internalErrorResponse("failed to get object")
	}
	namespace, ok := obj.(*corev1.Namespace)
	if !ok {
		klog.InfoS("failed to assert namespace type", "type", reflect.TypeOf(obj))
		return badRequestResponse("failed to decode namespace")
	}

	newPolicy, newErr := a.PolicyToEvaluate(namespace.Labels)

	switch attrs.GetOperation() {
	case admissionv1.Create:
		// require valid labels on create
		if newErr != nil {
			return invalidResponse(newErr.Error())
		}
		return sharedAllowedResponse()

	case admissionv1.Update:
		// if update, check if policy labels changed
		oldObj, err := attrs.GetOldObject()
		if err != nil {
			klog.ErrorS(err, "failed to decode old object")
			return badRequestResponse("failed to decode old object")
		}
		oldNamespace, ok := oldObj.(*corev1.Namespace)
		if !ok {
			klog.InfoS("failed to assert old namespace type", "type", reflect.TypeOf(oldObj))
			return badRequestResponse("failed to decode old namespace")
		}
		oldPolicy, oldErr := a.PolicyToEvaluate(oldNamespace.Labels)

		// require valid labels on update if they have changed
		if newErr != nil && (oldErr == nil || newErr.Error() != oldErr.Error()) {
			return invalidResponse(newErr.Error())
		}

		// Skip dry-running pods:
		// * if the enforce policy is unchanged
		// * if the new enforce policy is privileged
		// * if the new enforce is the same version and level was relaxed
		// * for exempt namespaces
		if newPolicy.Enforce == oldPolicy.Enforce {
			return sharedAllowedResponse()
		}
		if newPolicy.Enforce.Level == api.LevelPrivileged {
			return sharedAllowedResponse()
		}
		if newPolicy.Enforce.Version == oldPolicy.Enforce.Version &&
			api.CompareLevels(newPolicy.Enforce.Level, oldPolicy.Enforce.Level) < 1 {
			return sharedAllowedResponse()
		}
		if a.exemptNamespace(attrs.GetNamespace()) {
			return sharedAllowedResponse()
		}
		response := allowedResponse()
		response.Warnings = a.EvaluatePodsInNamespace(ctx, namespace.Name, newPolicy.Enforce)
		return response

	default:
		return sharedAllowedResponse()
	}
}

// ignoredPodSubresources is a set of ignored Pod subresources.
// Any other subresource is expected to be a *v1.Pod type and is evaluated.
// This ensures a version skewed webhook fails safe and denies an unknown pod subresource that allows modifying the pod spec.
var ignoredPodSubresources = map[string]bool{
	"exec":        true,
	"attach":      true,
	"binding":     true,
	"eviction":    true,
	"log":         true,
	"portforward": true,
	"proxy":       true,
	"status":      true,
}

// ValidatePod evaluates a pod create or update request against the effective policy for the namespace.
// The returned response may be shared between evaluations and must not be mutated.
func (a *Admission) ValidatePod(ctx context.Context, attrs Attributes) *admissionv1.AdmissionResponse {
	// short-circuit on ignored subresources
	if ignoredPodSubresources[attrs.GetSubresource()] {
		return sharedAllowedResponse()
	}
	// short-circuit on exempt namespaces and users
	if a.exemptNamespace(attrs.GetNamespace()) || a.exemptUser(attrs.GetUserName()) {
		return sharedAllowedResponse()
	}

	// short-circuit on privileged enforce+audit+warn namespaces
	namespace, err := a.NamespaceGetter.GetNamespace(ctx, attrs.GetNamespace())
	if err != nil {
		klog.ErrorS(err, "failed to fetch pod namespace", "namespace", attrs.GetNamespace())
		return internalErrorResponse(fmt.Sprintf("failed to lookup namespace %s", attrs.GetNamespace()))
	}
	nsPolicy, nsPolicyErr := a.PolicyToEvaluate(namespace.Labels)
	if nsPolicyErr == nil && nsPolicy.Enforce.Level == api.LevelPrivileged && nsPolicy.Warn.Level == api.LevelPrivileged && nsPolicy.Audit.Level == api.LevelPrivileged {
		return sharedAllowedResponse()
	}

	obj, err := attrs.GetObject()
	if err != nil {
		klog.ErrorS(err, "failed to decode object")
		return badRequestResponse("failed to decode object")
	}
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		klog.InfoS("failed to assert pod type", "type", reflect.TypeOf(obj))
		return badRequestResponse("failed to decode pod")
	}
	if attrs.GetOperation() == admissionv1.Update {
		oldObj, err := attrs.GetOldObject()
		if err != nil {
			klog.ErrorS(err, "failed to decode old object")
			return badRequestResponse("failed to decode old object")
		}
		oldPod, ok := oldObj.(*corev1.Pod)
		if !ok {
			klog.InfoS("failed to assert old pod type", "type", reflect.TypeOf(oldObj))
			return badRequestResponse("failed to decode old pod")
		}
		if !isSignificantPodUpdate(pod, oldPod) {
			// Nothing we care about changed, so always allow the update.
			return sharedAllowedResponse()
		}
	}
	return a.EvaluatePod(ctx, nsPolicy, nsPolicyErr, &pod.ObjectMeta, &pod.Spec, attrs, true)
}

// ValidatePodController evaluates a pod controller create or update request against the effective policy for the namespace.
// The returned response may be shared between evaluations and must not be mutated.
func (a *Admission) ValidatePodController(ctx context.Context, attrs Attributes) *admissionv1.AdmissionResponse {
	// short-circuit on subresources
	if attrs.GetSubresource() != "" {
		return sharedAllowedResponse()
	}
	// short-circuit on exempt namespaces and users
	if a.exemptNamespace(attrs.GetNamespace()) || a.exemptUser(attrs.GetUserName()) {
		return sharedAllowedResponse()
	}

	// short-circuit on privileged audit+warn namespaces
	namespace, err := a.NamespaceGetter.GetNamespace(ctx, attrs.GetNamespace())
	if err != nil {
		klog.ErrorS(err, "failed to fetch pod namespace", "namespace", attrs.GetNamespace())
		return internalErrorResponse(fmt.Sprintf("failed to lookup namespace %s", attrs.GetNamespace()))
	}
	nsPolicy, nsPolicyErr := a.PolicyToEvaluate(namespace.Labels)
	if nsPolicyErr == nil && nsPolicy.Warn.Level == api.LevelPrivileged && nsPolicy.Audit.Level == api.LevelPrivileged {
		return sharedAllowedResponse()
	}

	obj, err := attrs.GetObject()
	if err != nil {
		klog.ErrorS(err, "failed to decode object")
		return badRequestResponse("failed to decode object")
	}
	podMetadata, podSpec, err := a.PodSpecExtractor.ExtractPodSpec(obj)
	if err != nil {
		klog.ErrorS(err, "failed to extract pod spec")
		return badRequestResponse("failed to extract pod template")
	}
	if podMetadata == nil && podSpec == nil {
		// if a controller with an optional pod spec does not contain a pod spec, skip validation
		return sharedAllowedResponse()
	}
	return a.EvaluatePod(ctx, nsPolicy, nsPolicyErr, podMetadata, podSpec, attrs, false)
}

// EvaluatePod evaluates the given policy against the given pod(-like) object.
// The enforce policy is only checked if enforce=true.
// The returned response may be shared between evaluations and must not be mutated.
func (a *Admission) EvaluatePod(ctx context.Context, nsPolicy api.Policy, nsPolicyErr error, podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, attrs Attributes, enforce bool) *admissionv1.AdmissionResponse {
	// short-circuit on exempt runtimeclass
	if a.exemptRuntimeClass(podSpec.RuntimeClassName) {
		return sharedAllowedResponse()
	}

	auditAnnotations := map[string]string{}
	if nsPolicyErr != nil {
		klog.V(2).InfoS("failed to parse PodSecurity namespace labels", "err", nsPolicyErr)
		auditAnnotations["error"] = fmt.Sprintf("Failed to parse policy: %v", nsPolicyErr)
	}

	if klog.V(5).Enabled() {
		klog.InfoS("Pod Security evaluation", "policy", fmt.Sprintf("%v", nsPolicy), "op", attrs.GetOperation(), "resource", attrs.GetResource(), "namespace", attrs.GetNamespace(), "name", attrs.GetName())
	}

	response := allowedResponse()
	if enforce {
		if result := policy.AggregateCheckResults(a.Evaluator.EvaluatePod(nsPolicy.Enforce, podMetadata, podSpec)); !result.Allowed {
			response = forbiddenResponse(result.ForbiddenDetail())
		}
	}

	// TODO: reuse previous evaluation if audit level+version is the same as enforce level+version
	if result := policy.AggregateCheckResults(a.Evaluator.EvaluatePod(nsPolicy.Audit, podMetadata, podSpec)); !result.Allowed {
		auditAnnotations["audit"] = result.ForbiddenDetail()
	}

	// avoid adding warnings to a request we're already going to reject with an error
	if response.Allowed {
		// TODO: reuse previous evaluation if warn level+version is the same as audit or enforce level+version
		if result := policy.AggregateCheckResults(a.Evaluator.EvaluatePod(nsPolicy.Warn, podMetadata, podSpec)); !result.Allowed {
			// TODO: Craft a better user-facing warning message
			response.Warnings = append(response.Warnings, fmt.Sprintf(
				"would violate %q version of %q PodSecurity profile: %s",
				nsPolicy.Warn.Version.String(),
				nsPolicy.Warn.Level,
				result.ForbiddenDetail(),
			))
		}
	}

	response.AuditAnnotations = auditAnnotations
	return response
}

func (a *Admission) EvaluatePodsInNamespace(ctx context.Context, namespace string, enforce api.LevelVersion) []string {
	timeout := namespacePodCheckTimeout
	if deadline, ok := ctx.Deadline(); ok {
		timeRemaining := time.Duration(0.9 * float64(time.Until(deadline))) // Leave a little time to respond.
		if timeout > timeRemaining {
			timeout = timeRemaining
		}
	}
	deadline := time.Now().Add(timeout)
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	pods, err := a.PodLister.ListPods(ctx, namespace)
	if err != nil {
		klog.ErrorS(err, "Failed to list pods", "namespace", namespace)
		return []string{"Failed to list pods"}
	}

	var warnings []string
	if len(pods) > namespaceMaxPodsToCheck {
		warnings = append(warnings, fmt.Sprintf("Large namespace: only checking the first %d of %d pods", namespaceMaxPodsToCheck, len(pods)))
		pods = pods[0:namespaceMaxPodsToCheck]
	}

	for i, pod := range pods {
		// short-circuit on exempt runtimeclass
		if a.exemptRuntimeClass(pod.Spec.RuntimeClassName) {
			continue
		}
		r := policy.AggregateCheckResults(a.Evaluator.EvaluatePod(enforce, &pod.ObjectMeta, &pod.Spec))
		if !r.Allowed {
			// TODO: consider aggregating results (e.g. multiple pods failed for the same reasons)
			warnings = append(warnings, fmt.Sprintf("%s: %s", pod.Name, r.ForbiddenReason()))
		}
		if time.Now().After(deadline) {
			return append(warnings, fmt.Sprintf("Timeout reached after checking %d pods", i+1))
		}
	}

	return warnings
}

func (a *Admission) PolicyToEvaluate(labels map[string]string) (api.Policy, error) {
	return api.PolicyToEvaluate(labels, a.defaultPolicy)
}

var _sharedAllowedResponse = allowedResponse()

func sharedAllowedResponse() *admissionv1.AdmissionResponse {
	return _sharedAllowedResponse
}

// allowedResponse is the response used when the admission decision is allow.
func allowedResponse() *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{Allowed: true}
}

func failureResponse(msg string, reason metav1.StatusReason, code int32) *admissionv1.AdmissionResponse {
	return &admissionv1.AdmissionResponse{
		Allowed: false,
		Result: &metav1.Status{
			Status:  metav1.StatusFailure,
			Reason:  reason,
			Message: msg,
			Code:    code,
		},
	}
}

// forbiddenResponse is the response used when the admission decision is deny for policy violations.
func forbiddenResponse(msg string) *admissionv1.AdmissionResponse {
	return failureResponse(msg, metav1.StatusReasonForbidden, http.StatusForbidden)
}

// invalidResponse is the response used for namespace requests when namespace labels are invalid.
func invalidResponse(msg string) *admissionv1.AdmissionResponse {
	return failureResponse(msg, metav1.StatusReasonInvalid, 422)
}

// badRequestResponse is the response used when a request cannot be processed.
func badRequestResponse(msg string) *admissionv1.AdmissionResponse {
	return failureResponse(msg, metav1.StatusReasonBadRequest, http.StatusBadRequest)
}

// internalErrorResponse is the response used for unexpected errors
func internalErrorResponse(msg string) *admissionv1.AdmissionResponse {
	return failureResponse(msg, metav1.StatusReasonInternalError, http.StatusInternalServerError)
}

// isSignificantPodUpdate determines whether a pod update should trigger a policy evaluation.
// Relevant mutable pod fields as of 1.21 are image and seccomp annotations:
// * https://github.com/kubernetes/kubernetes/blob/release-1.21/pkg/apis/core/validation/validation.go#L3947-L3949
func isSignificantPodUpdate(pod, oldPod *corev1.Pod) bool {
	if pod.Annotations[corev1.SeccompPodAnnotationKey] != oldPod.Annotations[corev1.SeccompPodAnnotationKey] {
		return true
	}
	if len(pod.Spec.Containers) != len(oldPod.Spec.Containers) {
		return true
	}
	if len(pod.Spec.InitContainers) != len(oldPod.Spec.InitContainers) {
		return true
	}
	for i := 0; i < len(pod.Spec.Containers); i++ {
		if isSignificantContainerUpdate(&pod.Spec.Containers[i], &oldPod.Spec.Containers[i], pod.Annotations, oldPod.Annotations) {
			return true
		}
	}
	for i := 0; i < len(pod.Spec.InitContainers); i++ {
		if isSignificantContainerUpdate(&pod.Spec.InitContainers[i], &oldPod.Spec.InitContainers[i], pod.Annotations, oldPod.Annotations) {
			return true
		}
	}
	for _, c := range pod.Spec.EphemeralContainers {
		var oldC *corev1.Container
		for i, oc := range oldPod.Spec.EphemeralContainers {
			if oc.Name == c.Name {
				oldC = (*corev1.Container)(&oldPod.Spec.EphemeralContainers[i].EphemeralContainerCommon)
				break
			}
		}
		if oldC == nil {
			return true // EphemeralContainer added
		}
		if isSignificantContainerUpdate((*corev1.Container)(&c.EphemeralContainerCommon), oldC, pod.Annotations, oldPod.Annotations) {
			return true
		}
	}
	return false
}

// isSignificantContainerUpdate determines whether a container update should trigger a policy evaluation.
func isSignificantContainerUpdate(container, oldContainer *corev1.Container, annotations, oldAnnotations map[string]string) bool {
	if container.Image != oldContainer.Image {
		return true
	}
	seccompKey := corev1.SeccompContainerAnnotationKeyPrefix + container.Name
	return annotations[seccompKey] != oldAnnotations[seccompKey]
}

func (a *Admission) exemptNamespace(namespace string) bool {
	if len(namespace) == 0 {
		return false
	}
	// TODO: consider optimizing to O(1) lookup
	return containsString(namespace, a.Configuration.Exemptions.Namespaces)
}
func (a *Admission) exemptUser(username string) bool {
	if len(username) == 0 {
		return false
	}
	// TODO: consider optimizing to O(1) lookup
	return containsString(username, a.Configuration.Exemptions.Usernames)
}
func (a *Admission) exemptRuntimeClass(runtimeClass *string) bool {
	if runtimeClass == nil || len(*runtimeClass) == 0 {
		return false
	}
	// TODO: consider optimizing to O(1) lookup
	return containsString(*runtimeClass, a.Configuration.Exemptions.RuntimeClasses)
}
func containsString(needle string, haystack []string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
