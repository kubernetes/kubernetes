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
	"fmt"
	"io"
	"strings"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	extensionslisters "k8s.io/kubernetes/pkg/client/listers/extensions/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	psp "k8s.io/kubernetes/pkg/security/podsecuritypolicy"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/interfaces"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

const (
	PluginName = "PodSecurityPolicy"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		plugin := NewPlugin(psp.NewSimpleStrategyFactory(), getMatchingPolicies, true)
		return plugin, nil
	})
}

// PSPMatchFn allows plugging in how PSPs are matched against user information.
type PSPMatchFn func(lister extensionslisters.PodSecurityPolicyLister, user user.Info, sa user.Info, authz authorizer.Authorizer, namespace string) ([]*extensions.PodSecurityPolicy, error)

// podSecurityPolicyPlugin holds state for and implements the admission plugin.
type podSecurityPolicyPlugin struct {
	*admission.Handler
	strategyFactory  psp.StrategyFactory
	pspMatcher       PSPMatchFn
	failOnNoPolicies bool
	authz            authorizer.Authorizer
	lister           extensionslisters.PodSecurityPolicyLister
}

// SetAuthorizer sets the authorizer.
func (plugin *podSecurityPolicyPlugin) SetAuthorizer(authz authorizer.Authorizer) {
	plugin.authz = authz
}

// Validate ensures an authorizer is set.
func (plugin *podSecurityPolicyPlugin) Validate() error {
	if plugin.authz == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	if plugin.lister == nil {
		return fmt.Errorf("%s requires a lister", PluginName)
	}
	return nil
}

var _ admission.Interface = &podSecurityPolicyPlugin{}
var _ kubeapiserveradmission.WantsAuthorizer = &podSecurityPolicyPlugin{}
var _ kubeapiserveradmission.WantsInternalKubeInformerFactory = &podSecurityPolicyPlugin{}

// NewPlugin creates a new PSP admission plugin.
func NewPlugin(strategyFactory psp.StrategyFactory, pspMatcher PSPMatchFn, failOnNoPolicies bool) *podSecurityPolicyPlugin {
	return &podSecurityPolicyPlugin{
		Handler:          admission.NewHandler(admission.Create, admission.Update),
		strategyFactory:  strategyFactory,
		pspMatcher:       pspMatcher,
		failOnNoPolicies: failOnNoPolicies,
	}
}

func (a *podSecurityPolicyPlugin) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	podSecurityPolicyInformer := f.Extensions().InternalVersion().PodSecurityPolicies()
	a.lister = podSecurityPolicyInformer.Lister()
	a.SetReadyFunc(podSecurityPolicyInformer.Informer().HasSynced)
}

// Admit determines if the pod should be admitted based on the requested security context
// and the available PSPs.
//
// 1.  Find available PSPs.
// 2.  Create the providers, includes setting pre-allocated values if necessary.
// 3.  Try to generate and validate a PSP with providers.  If we find one then admit the pod
//     with the validated PSP.  If we don't find any reject the pod and give all errors from the
//     failed attempts.
func (c *podSecurityPolicyPlugin) Admit(a admission.Attributes) error {
	if a.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	if len(a.GetSubresource()) != 0 {
		return nil
	}

	pod, ok := a.GetObject().(*api.Pod)
	// if we can't convert then we don't handle this object so just return
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
	}

	// get all constraints that are usable by the user
	glog.V(4).Infof("getting pod security policies for pod %s (generate: %s)", pod.Name, pod.GenerateName)
	var saInfo user.Info
	if len(pod.Spec.ServiceAccountName) > 0 {
		saInfo = serviceaccount.UserInfo(a.GetNamespace(), pod.Spec.ServiceAccountName, "")
	}

	matchedPolicies, err := c.pspMatcher(c.lister, a.GetUserInfo(), saInfo, c.authz, a.GetNamespace())
	if err != nil {
		return admission.NewForbidden(a, err)
	}

	// if we have no policies and want to succeed then return.  Otherwise we'll end up with no
	// providers and fail with "unable to validate against any pod security policy" below.
	if len(matchedPolicies) == 0 && !c.failOnNoPolicies {
		return nil
	}

	providers, errs := c.createProvidersFromPolicies(matchedPolicies, pod.Namespace)
	logProviders(pod, providers, errs)

	if len(providers) == 0 {
		return admission.NewForbidden(a, fmt.Errorf("no pod security policies available"))
	}

	results := make([]*interfaces.ValidationResult, len(providers))
	for i, provider := range providers {
		result, err := provider.ValidatePod(pod)
		if err != nil {
			glog.Errorf("error evaluating pod %s (generate: %s) by provider %s: %v", pod.Name, pod.GenerateName, provider.GetPSPName(), err)
			continue
		}
		if result.Result != interfaces.Allowed {
			results[i] = result
			continue
		}
		glog.V(4).Infof("pod %s (generate: %s) validated against provider %s", pod.Name, pod.GenerateName, provider.GetPSPName())
		if pod.Annotations == nil {
			pod.Annotations = map[string]string{}
		}
		pod.Annotations[psputil.ValidatedPSPAnnotation] = provider.GetPSPName()
		return nil
	}

	// See if a provider just required defaulting
	for i, result := range results {
		if result.Result != interfaces.RequiresDefaulting {
			continue
		}

		provider := providers[i]
		if err := provider.DefaultPod(pod); err != nil {
			return admission.NewForbidden(a, err)
		}

		if result, err := provider.ValidatePod(pod); err != nil {
			return admission.NewForbidden(a, fmt.Errorf("provider %s did not allow the pod: %v", provider.GetPSPName(), err))
		} else if result.Result != interfaces.Allowed {
			return admission.NewForbidden(a, fmt.Errorf("provider %s did not allow the pod: %v", provider.GetPSPName(), result.Errors))
		}

		glog.V(4).Infof("pod %s (generate: %s) defaulted and validated against provider %s", pod.Name, pod.GenerateName, provider.GetPSPName())
		if pod.Annotations == nil {
			pod.Annotations = map[string]string{}
		}
		pod.Annotations[psputil.ValidatedPSPAnnotation] = provider.GetPSPName()
		return nil
	}

	validationErrs := []error{}
	for i, result := range results {
		if result.Result != interfaces.Forbidden {
			continue
		}
		provider := providers[i]
		validationErrs = append(validationErrs, fmt.Errorf("%s: %v", provider.GetPSPName(), result.Errors))
	}
	// we didn't validate against any provider, reject the pod and give the errors for each attempt
	glog.V(4).Infof("unable to validate pod %s (generate: %s) against any pod security policy: %v", pod.Name, pod.GenerateName, validationErrs)
	return admission.NewForbidden(a, fmt.Errorf("unable to validate against any pod security policy: %v", validationErrs))
}

// createProvidersFromPolicies creates providers from the constraints supplied.
func (c *podSecurityPolicyPlugin) createProvidersFromPolicies(psps []*extensions.PodSecurityPolicy, namespace string) ([]psp.Provider, []error) {
	var (
		// collected providers
		providers []psp.Provider
		// collected errors to return
		errs []error
	)

	for _, constraint := range psps {
		provider, err := psp.NewSimpleProvider(constraint, namespace, c.strategyFactory)
		if err != nil {
			errs = append(errs, fmt.Errorf("error creating provider for PSP %s: %v", constraint.Name, err))
			continue
		}
		providers = append(providers, provider)
	}
	return providers, errs
}

// getMatchingPolicies returns policies from the lister.  For now this returns everything
// in the future it can filter based on UserInfo and permissions.
//
// TODO: this will likely need optimization since the initial implementation will
// always query for authorization.  Needs scale testing and possibly checking against
// a cache.
func getMatchingPolicies(lister extensionslisters.PodSecurityPolicyLister, user user.Info, sa user.Info, authz authorizer.Authorizer, namespace string) ([]*extensions.PodSecurityPolicy, error) {
	matchedPolicies := make([]*extensions.PodSecurityPolicy, 0)

	list, err := lister.List(labels.Everything())
	if err != nil {
		return nil, err
	}

	for _, constraint := range list {
		if authorizedForPolicy(user, namespace, constraint, authz) || authorizedForPolicy(sa, namespace, constraint, authz) {
			matchedPolicies = append(matchedPolicies, constraint)
		}
	}

	return matchedPolicies, nil
}

// authorizedForPolicy returns true if info is authorized to perform the "use" verb on the policy resource.
func authorizedForPolicy(info user.Info, namespace string, policy *extensions.PodSecurityPolicy, authz authorizer.Authorizer) bool {
	if info == nil {
		return false
	}
	attr := buildAttributes(info, namespace, policy)
	allowed, reason, err := authz.Authorize(attr)
	if err != nil {
		glog.V(5).Infof("cannot authorize for policy: %v,%v", reason, err)
	}
	return allowed
}

// buildAttributes builds an attributes record for a SAR based on the user info and policy.
func buildAttributes(info user.Info, namespace string, policy *extensions.PodSecurityPolicy) authorizer.Attributes {
	// check against the namespace that the pod is being created in to allow per-namespace PSP grants.
	attr := authorizer.AttributesRecord{
		User:            info,
		Verb:            "use",
		Namespace:       namespace,
		Name:            policy.Name,
		APIGroup:        extensions.GroupName,
		Resource:        "podsecuritypolicies",
		ResourceRequest: true,
	}
	return attr
}

// logProviders logs what providers were found for the pod as well as any errors that were encountered
// while creating providers.
func logProviders(pod *api.Pod, providers []psp.Provider, providerCreationErrs []error) {
	for _, err := range providerCreationErrs {
		glog.V(4).Infof("provider creation error: %v", err)
	}

	if len(providers) == 0 {
		glog.V(4).Infof("unable to validate pod %s (generate: %s) against any provider.", pod.Name, pod.GenerateName)
		return
	}

	names := make([]string, len(providers))
	for i, p := range providers {
		names[i] = p.GetPSPName()
	}
	glog.V(4).Infof("validating pod %s (generate: %s) against providers: %s", pod.Name, pod.GenerateName, strings.Join(names, ","))
}
