/*
Copyright 2020 The Kubernetes Authors.

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

package gkeautopilot

import (
	"context"
	"fmt"
	"io/ioutil"
	"sync/atomic"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	cache "k8s.io/client-go/tools/cache"
	"sigs.k8s.io/yaml"
)

const (
	// impersonationVerb is the verb used in an impersonation request
	impersonationVerb string = "impersonate"
	// TODO(b/172375975): To keep the plugin surface small for now, the config
	// file name is hardcoded. This should eventually piped up to be read from
	// the kube-apiserver command line
	configFile string = "/etc/srv/kubernetes/gkeautopilot-authz-config.yaml"
)

// Reason messages for Authorize
const (
	authReasonNoOpinion                         string = ""
	authReasonDeniedImpersonation               string = "GKEAutopilot authz: user impersonation is not allowed"
	authReasonDeniedPolicyEnforcementNotEnabled string = "GKEAutopilot authz: the request was sent before policy enforcement is enabled"
	authReasonDeniedVerbManagedNamespace        string = "GKEAutopilot authz: the namespace %q is managed and the request's verb %q is denied"
	authReasonDeniedResourceManagedNamespace    string = "GKEAutopilot authz: the namespace %q is managed and the request's resource %q is denied"
	authReasonDeniedVerbClusterScopedResource   string = "GKEAutopilot authz: the verb %q is denied for cluster scoped resources"
	authReasonDeniedClusterScopedResource       string = "GKEAutopilot authz: cluster scoped resource %q is managed and access is denied"
	authReasonDeniedSubresourceManagedNamespace string = "GKEAutopilot authz: the namespace %q is managed and the request's subresource %q is denied"
	authReasonDeniedVerbManagedResource         string = "GKEAutopilot authz: the resource %q is managed and the request's verb %q is not allowed"
)

// atomicFlag is a boolean flag with atomic operations
type atomicFlag uint32

func (a *atomicFlag) set(flag bool) {
	if flag {
		atomic.StoreUint32((*uint32)(a), 1)
	} else {
		atomic.StoreUint32((*uint32)(a), 0)
	}
}
func (a *atomicFlag) isSet() bool {
	return atomic.LoadUint32((*uint32)(a)) == 1
}

// Authorizer implementens Authorize for GKEAutopilot
type Authorizer struct {
	webhookInformer cache.SharedIndexInformer
	// a boolean flag which is set when the policy enforcer is enabled
	policyEnforcerEnabledFlag atomicFlag
	config                    *config
	configHelper              *configHelper
}

// New returns a new instance of Authorizer
func New(webhookInformer cache.SharedIndexInformer) (*Authorizer, error) {
	if webhookInformer == nil {
		return nil, fmt.Errorf("the passed webhookInformer cannot be nil")
	}

	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to load gkeautopilot authorizer configuration: %w", err)
	}

	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("failed to validate the loaded gkeautopilot authorizer configuration: %w", err)
	}

	authorizer := &Authorizer{
		webhookInformer: webhookInformer,
		config:          config,
		configHelper:    buildConfigHelper(config),
	}

	go authorizer.setupInformer()

	return authorizer, nil
}

// sets up the informer for the policy enforcer validating webhook
func (a *Authorizer) setupInformer() {
	ctx, cancel := context.WithCancel(context.TODO())
	defer cancel()

	setFlagIfPolicyEnforcerObject := func(obj interface{}, flagToSet bool) {
		webhook, ok := obj.(metav1.Object)
		if ok && webhook.GetName() == a.config.PolicyEnforcerWebhookName {
			a.policyEnforcerEnabledFlag.set(flagToSet)
		}
	}

	a.webhookInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			setFlagIfPolicyEnforcerObject(obj, true)
		},
		DeleteFunc: func(obj interface{}) {
			// if obj is a DeletedFinalStateUnknown, unwrap to get the
			// underlying obj
			if deletedObj, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				obj = deletedObj.Obj
			}
			setFlagIfPolicyEnforcerObject(obj, false)
		},
	})
	a.webhookInformer.Run(ctx.Done())
}

// loadConfig reads the configuration and returns a config
// object
func loadConfig() (*config, error) {
	configYaml, err := ioutil.ReadFile(configFile)
	if err != nil {
		return nil, err
	}

	config := &config{}
	return config, yaml.Unmarshal(configYaml, config)
}

// Authorize implements the interface for Authorizer type
func (a *Authorizer) Authorize(ctx context.Context, request authorizer.Attributes) (authorizer.Decision, string, error) {

	if a.isIdentityIgnored(request) {
		return authorizer.DecisionNoOpinion, authReasonNoOpinion, nil
	}

	if a.isUserImpersonating(request) {
		return authorizer.DecisionDeny, authReasonDeniedImpersonation, nil
	}

	// if this is not a resource request return no opinion
	if !request.IsResourceRequest() {
		return authorizer.DecisionNoOpinion, authReasonNoOpinion, nil
	}

	if !a.policyEnforcerEnabledFlag.isSet() {
		return authorizer.DecisionDeny, authReasonDeniedPolicyEnforcementNotEnabled, nil
	}

	if denied, reason := a.isRequestForNamespaceDenied(request); denied {
		return authorizer.DecisionDeny, reason, nil
	}

	if denied, reason := a.isRequestVerbForResourceDenied(request); denied {
		return authorizer.DecisionDeny, reason, nil
	}

	return authorizer.DecisionNoOpinion, authReasonNoOpinion, nil
}

// RulesFor implements the interface for Authorizer type
func (a *Authorizer) RulesFor(user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	return nil, nil, false, nil
}

// isIdentityIgnored returns true if the request is from an ignored
// identity. The authorized will not have an opinion on these requests
func (a *Authorizer) isIdentityIgnored(request authorizer.Attributes) bool {
	username := request.GetUser().GetName()
	if a.configHelper.ignoredUsersSet.Has(username) {
		return true
	}

	groups := request.GetUser().GetGroups()
	for _, group := range groups {
		if a.configHelper.ignoredGroupsSet.Has(group) {
			return true
		}
	}

	return false
}

// isRequestVerbForNamespaceDenied returns true if:
// - the namespace is managed AND the verb is not allowed
//
// Note: Cluster-scoped resources are treated as resources in the "" (empty)
// namespace
func (a *Authorizer) isRequestForNamespaceDenied(request authorizer.Attributes) (bool, string) {
	reqNamespace := request.GetNamespace()

	if ds, ok := a.configHelper.managedNamespacesMap[reqNamespace]; ok {
		reqResource := request.GetResource()
		reqResSubres := resSubToString(reqResource, request.GetSubresource())

		if ds.ignoredResourceSubresource.Has(resSubToString(reqResource, "")) {
			return false, authReasonNoOpinion
		}

		if ds.ignoredResourceSubresource.Has(reqResSubres) {
			return false, authReasonNoOpinion
		}

		reqVerb := request.GetVerb()
		if ds.deniedVerbs.Has(reqVerb) {
			reason := fmt.Sprintf(authReasonDeniedVerbManagedNamespace, reqNamespace, reqVerb)
			if reqNamespace == "" {
				reason = fmt.Sprintf(authReasonDeniedVerbClusterScopedResource, reqVerb)
			}

			return true, reason
		}

		if ds.deniedResourceSubresource.Has(resSubToString(reqResource, "")) {
			reason := fmt.Sprintf(authReasonDeniedResourceManagedNamespace, reqNamespace, reqResource)
			if reqNamespace == "" {
				reason = fmt.Sprintf(authReasonDeniedClusterScopedResource, resSubToString(reqResource, ""))
			}

			return true, reason
		}

		if ds.deniedResourceSubresource.Has(reqResSubres) {
			reason := fmt.Sprintf(authReasonDeniedSubresourceManagedNamespace, reqNamespace, reqResSubres)
			if reqNamespace == "" {
				reason = fmt.Sprintf(authReasonDeniedClusterScopedResource, reqResSubres)
			}

			return true, reason
		}
	}

	return false, authReasonNoOpinion
}

// isRequestVerbForResourceDenied returns true if:
//   when request is not for a subresource
//    - the resource is managed AND the verb is not allowed
//   when request is for a subresource
//    - the subresource is managed AND the verb is not allowed
//    - the subresource is not managed AND the parent resource is managed
func (a *Authorizer) isRequestVerbForResourceDenied(request authorizer.Attributes) (bool, string) {
	// explore the resource tree from the root. If there is no path to resource
	// node, return immediately
	root := a.configHelper.managedResourcesTree
	agNode := root.children[request.GetAPIGroup()]
	if agNode == nil {
		return false, authReasonNoOpinion
	}

	nsNode := agNode.children[request.GetNamespace()]
	if nsNode == nil {
		return false, authReasonNoOpinion
	}

	resource := request.GetResource()
	resNode := nsNode.children[resource]
	if resNode == nil {
		return false, authReasonNoOpinion
	}

	resName := request.GetName()
	nameNode := resNode.children[resName]
	if nameNode == nil {
		return false, authReasonNoOpinion
	}

	subresource := request.GetSubresource()
	reqVerb := request.GetVerb()
	if subresource == "" {
		if !nameNode.allowedVerbs.Has(reqVerb) {
			reason := fmt.Sprintf(authReasonDeniedVerbManagedResource, resource+"/"+resName, reqVerb)
			return true, reason
		}

		return false, authReasonNoOpinion
	}

	subresNode := nameNode.children[subresource]
	if subresNode == nil || !subresNode.allowedVerbs.Has(reqVerb) {
		reason := fmt.Sprintf(authReasonDeniedVerbManagedResource, resource+"/"+resName+"/"+subresource, reqVerb)
		return true, reason
	}

	return false, authReasonNoOpinion
}

// isUserImpersonating determines whether this is an impersonation request
func (a *Authorizer) isUserImpersonating(request authorizer.Attributes) bool {
	return request.GetVerb() == impersonationVerb
}
