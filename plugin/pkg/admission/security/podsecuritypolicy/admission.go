/*
Copyright 2016 The Kubernetes Authors.

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

package podsecuritypolicy

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"

	"k8s.io/klog/v2"

	policyv1beta1 "k8s.io/api/policy/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	policylisters "k8s.io/client-go/listers/policy/v1beta1"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	rbacregistry "k8s.io/kubernetes/pkg/registry/rbac"
	psp "k8s.io/kubernetes/pkg/security/podsecuritypolicy"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

// PluginName is a string with the name of the plugin
const PluginName = "PodSecurityPolicy"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := newPlugin(psp.NewSimpleStrategyFactory(), true)
		return plugin, nil
	})
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
	strategyFactory  psp.StrategyFactory
	failOnNoPolicies bool
	authz            authorizer.Authorizer
	lister           policylisters.PodSecurityPolicyLister
}

// SetAuthorizer sets the authorizer.
func (p *Plugin) SetAuthorizer(authz authorizer.Authorizer) {
	p.authz = authz
}

// ValidateInitialization ensures an authorizer is set.
func (p *Plugin) ValidateInitialization() error {
	if p.authz == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	if p.lister == nil {
		return fmt.Errorf("%s requires a lister", PluginName)
	}
	return nil
}

var _ admission.MutationInterface = &Plugin{}
var _ admission.ValidationInterface = &Plugin{}
var _ genericadmissioninit.WantsAuthorizer = &Plugin{}
var _ genericadmissioninit.WantsExternalKubeInformerFactory = &Plugin{}
var auditKeyPrefix = strings.ToLower(PluginName) + "." + policy.GroupName + ".k8s.io"

// newPlugin creates a new PSP admission plugin.
func newPlugin(strategyFactory psp.StrategyFactory, failOnNoPolicies bool) *Plugin {
	return &Plugin{
		Handler:          admission.NewHandler(admission.Create, admission.Update),
		strategyFactory:  strategyFactory,
		failOnNoPolicies: failOnNoPolicies,
	}
}

// SetExternalKubeInformerFactory registers an informer
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	podSecurityPolicyInformer := f.Policy().V1beta1().PodSecurityPolicies()
	p.lister = podSecurityPolicyInformer.Lister()
	p.SetReadyFunc(podSecurityPolicyInformer.Informer().HasSynced)
}

// Admit determines if the pod should be admitted based on the requested security context
// and the available PSPs.
//
// 1.  Find available PSPs.
// 2.  Create the providers, includes setting pre-allocated values if necessary.
// 3.  Try to generate and validate a PSP with providers.  If we find one then admit the pod
//     with the validated PSP.  If we don't find any reject the pod and give all errors from the
//     failed attempts.
func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if ignore, err := shouldIgnore(a); err != nil {
		return err
	} else if ignore {
		return nil
	}

	// only mutate if this is a CREATE request. On updates we only validate.
	if a.GetOperation() != admission.Create {
		return nil
	}

	pod := a.GetObject().(*api.Pod)

	// compute the context. Mutation is allowed. ValidatedPSPAnnotation is not taken into account.
	allowedPod, pspName, validationErrs, err := p.computeSecurityContext(ctx, a, pod, true, "")
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("PodSecurityPolicy: %w", err))
	}
	if allowedPod != nil {
		*pod = *allowedPod
		// annotate and accept the pod
		klog.V(4).Infof("pod %s (generate: %s) in namespace %s validated against provider %s", pod.Name, pod.GenerateName, a.GetNamespace(), pspName)
		if pod.ObjectMeta.Annotations == nil {
			pod.ObjectMeta.Annotations = map[string]string{}
		}
		pod.ObjectMeta.Annotations[psputil.ValidatedPSPAnnotation] = pspName
		key := auditKeyPrefix + "/" + "admit-policy"
		if err := a.AddAnnotation(key, pspName); err != nil {
			klog.Warningf("failed to set admission audit annotation %s to %s: %v", key, pspName, err)
		}
		return nil
	}

	// we didn't validate against any provider, reject the pod and give the errors for each attempt
	klog.V(4).Infof("unable to admit pod %s (generate: %s) in namespace %s against any pod security policy: %v", pod.Name, pod.GenerateName, a.GetNamespace(), validationErrs)
	return admission.NewForbidden(a, fmt.Errorf("PodSecurityPolicy: unable to admit pod: %v", validationErrs))
}

// Validate verifies attributes against the PodSecurityPolicy
func (p *Plugin) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if ignore, err := shouldIgnore(a); err != nil {
		return err
	} else if ignore {
		return nil
	}

	pod := a.GetObject().(*api.Pod)

	// compute the context. Mutation is not allowed. ValidatedPSPAnnotation is used as a hint to gain same speed-up.
	allowedPod, pspName, validationErrs, err := p.computeSecurityContext(ctx, a, pod, false, pod.ObjectMeta.Annotations[psputil.ValidatedPSPAnnotation])
	if err != nil {
		return admission.NewForbidden(a, fmt.Errorf("PodSecurityPolicy: %w", err))
	}
	if apiequality.Semantic.DeepEqual(pod, allowedPod) {
		key := auditKeyPrefix + "/" + "validate-policy"
		if err := a.AddAnnotation(key, pspName); err != nil {
			klog.Warningf("failed to set admission audit annotation %s to %s: %v", key, pspName, err)
		}
		return nil
	}

	// we didn't validate against any provider, reject the pod and give the errors for each attempt
	klog.V(4).Infof("unable to validate pod %s (generate: %s) in namespace %s against any pod security policy: %v", pod.Name, pod.GenerateName, a.GetNamespace(), validationErrs)
	return admission.NewForbidden(a, fmt.Errorf("PodSecurityPolicy: unable to validate pod: %v", validationErrs))
}

func shouldIgnore(a admission.Attributes) (bool, error) {
	if a.GetResource().GroupResource() != api.Resource("pods") {
		return true, nil
	}
	if len(a.GetSubresource()) != 0 {
		return true, nil
	}

	// if we can't convert then fail closed since we've already checked that this is supposed to be a pod object.
	// this shouldn't normally happen during admission but could happen if an integrator passes a versioned
	// pod object rather than an internal object.
	if _, ok := a.GetObject().(*api.Pod); !ok {
		return false, admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
	}

	// if this is an update, see if we are only updating the ownerRef/finalizers.  Garbage collection does this
	// and we should allow it in general, since you had the power to update and the power to delete.
	// The worst that happens is that you delete something, but you aren't controlling the privileged object itself
	if a.GetOperation() == admission.Update && rbacregistry.IsOnlyMutatingGCFields(a.GetObject(), a.GetOldObject(), apiequality.Semantic) {
		return true, nil
	}

	return false, nil
}

// computeSecurityContext derives a valid security context while trying to avoid any changes to the given pod. I.e.
// if there is a matching policy with the same security context as given, it will be reused. If there is no
// matching policy the returned pod will be nil and the pspName empty. validatedPSPHint is the validated psp name
// saved in kubernetes.io/psp annotation. This psp is usually the one we are looking for.
func (p *Plugin) computeSecurityContext(ctx context.Context, a admission.Attributes, pod *api.Pod, specMutationAllowed bool, validatedPSPHint string) (*api.Pod, string, field.ErrorList, error) {
	// get all constraints that are usable by the user
	klog.V(4).Infof("getting pod security policies for pod %s (generate: %s)", pod.Name, pod.GenerateName)
	var saInfo user.Info
	if len(pod.Spec.ServiceAccountName) > 0 {
		saInfo = serviceaccount.UserInfo(a.GetNamespace(), pod.Spec.ServiceAccountName, "")
	}

	policies, err := p.lister.List(labels.Everything())
	if err != nil {
		return nil, "", nil, err
	}

	// if we have no policies and want to succeed then return.  Otherwise we'll end up with no
	// providers and fail with "unable to validate against any pod security policy" below.
	if len(policies) == 0 && !p.failOnNoPolicies {
		return pod, "", nil, nil
	}

	// sort policies by name to make order deterministic
	// If mutation is not allowed and validatedPSPHint is provided, check the validated policy first.
	sort.SliceStable(policies, func(i, j int) bool {
		if !specMutationAllowed {
			if policies[i].Name == validatedPSPHint {
				return true
			}
			if policies[j].Name == validatedPSPHint {
				return false
			}
		}
		return strings.Compare(policies[i].Name, policies[j].Name) < 0
	})

	providers, errs := p.createProvidersFromPolicies(policies, pod.Namespace)
	for _, err := range errs {
		klog.V(4).Infof("provider creation error: %v", err)
	}

	if len(providers) == 0 {
		return nil, "", nil, fmt.Errorf("no providers available to validate pod request")
	}

	var (
		allowedMutatedPod   *api.Pod
		allowingMutatingPSP string
		// Map of PSP name to associated validation errors.
		validationErrs = map[string]field.ErrorList{}
	)

	for _, provider := range providers {
		podCopy := pod.DeepCopy()

		if errs := assignSecurityContext(provider, podCopy); len(errs) > 0 {
			validationErrs[provider.GetPSPName()] = errs
			continue
		}

		// the entire pod validated
		mutated := !apiequality.Semantic.DeepEqual(pod, podCopy)
		if mutated && !specMutationAllowed {
			continue
		}

		if !isAuthorizedForPolicy(ctx, a.GetUserInfo(), saInfo, a.GetNamespace(), provider.GetPSPName(), p.authz) {
			continue
		}

		switch {
		case !mutated:
			// if it validated without mutating anything, use this result
			return podCopy, provider.GetPSPName(), nil, nil

		case specMutationAllowed && allowedMutatedPod == nil:
			// if mutation is allowed and this is the first PSP to allow the pod, remember it,
			// but continue to see if another PSP allows without mutating
			allowedMutatedPod = podCopy
			allowingMutatingPSP = provider.GetPSPName()
		}
	}

	if allowedMutatedPod != nil {
		return allowedMutatedPod, allowingMutatingPSP, nil, nil
	}

	// Pod is rejected. Filter the validation errors to only include errors from authorized PSPs.
	aggregate := field.ErrorList{}
	for psp, errs := range validationErrs {
		if isAuthorizedForPolicy(ctx, a.GetUserInfo(), saInfo, a.GetNamespace(), psp, p.authz) {
			aggregate = append(aggregate, errs...)
		}
	}
	return nil, "", aggregate, nil
}

// assignSecurityContext creates a security context for each container in the pod
// and validates that the sc falls within the psp constraints.  All containers must validate against
// the same psp or is not considered valid.
func assignSecurityContext(provider psp.Provider, pod *api.Pod) field.ErrorList {
	errs := field.ErrorList{}

	if err := provider.MutatePod(pod); err != nil {
		// TODO(tallclair): MutatePod should return a field.ErrorList
		errs = append(errs, field.Invalid(field.NewPath(""), pod, err.Error()))
	}

	errs = append(errs, provider.ValidatePod(pod)...)

	return errs
}

// createProvidersFromPolicies creates providers from the constraints supplied.
func (p *Plugin) createProvidersFromPolicies(psps []*policyv1beta1.PodSecurityPolicy, namespace string) ([]psp.Provider, []error) {
	var (
		// collected providers
		providers []psp.Provider
		// collected errors to return
		errs []error
	)

	for _, constraint := range psps {
		provider, err := psp.NewSimpleProvider(constraint, namespace, p.strategyFactory)
		if err != nil {
			errs = append(errs, fmt.Errorf("error creating provider for PSP %s: %v", constraint.Name, err))
			continue
		}
		providers = append(providers, provider)
	}
	return providers, errs
}

func isAuthorizedForPolicy(ctx context.Context, user, sa user.Info, namespace, policyName string, authz authorizer.Authorizer) bool {
	// Check the service account first, as that is the more common use case.
	return authorizedForPolicy(ctx, sa, namespace, policyName, authz) ||
		authorizedForPolicy(ctx, user, namespace, policyName, authz)
}

// authorizedForPolicy returns true if info is authorized to perform the "use" verb on the policy resource.
// TODO: check against only the policy group when PSP will be completely moved out of the extensions
func authorizedForPolicy(ctx context.Context, info user.Info, namespace string, policyName string, authz authorizer.Authorizer) bool {
	// Check against extensions API group for backward compatibility
	return authorizedForPolicyInAPIGroup(ctx, info, namespace, policyName, policy.GroupName, authz) ||
		authorizedForPolicyInAPIGroup(ctx, info, namespace, policyName, extensions.GroupName, authz)
}

// authorizedForPolicyInAPIGroup returns true if info is authorized to perform the "use" verb on the policy resource in the specified API group.
func authorizedForPolicyInAPIGroup(ctx context.Context, info user.Info, namespace, policyName, apiGroupName string, authz authorizer.Authorizer) bool {
	if info == nil {
		return false
	}
	attr := buildAttributes(info, namespace, policyName, apiGroupName)
	decision, reason, err := authz.Authorize(ctx, attr)
	if err != nil {
		klog.V(5).Infof("cannot authorize for policy: %v,%v", reason, err)
	}
	return (decision == authorizer.DecisionAllow)
}

// buildAttributes builds an attributes record for a SAR based on the user info and policy.
func buildAttributes(info user.Info, namespace, policyName, apiGroupName string) authorizer.Attributes {
	// check against the namespace that the pod is being created in to allow per-namespace PSP grants.
	attr := authorizer.AttributesRecord{
		User:            info,
		Verb:            "use",
		Namespace:       namespace,
		Name:            policyName,
		APIGroup:        apiGroupName,
		APIVersion:      "*",
		Resource:        "podsecuritypolicies",
		ResourceRequest: true,
	}
	return attr
}
