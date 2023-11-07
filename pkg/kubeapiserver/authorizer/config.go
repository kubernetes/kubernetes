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

package authorizer

import (
	"errors"
	"fmt"
	"strings"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	authzconfig "k8s.io/apiserver/pkg/apis/apiserver"
	"k8s.io/apiserver/pkg/apis/apiserver/load"
	"k8s.io/apiserver/pkg/apis/apiserver/validation"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/authorizerfactory"
	"k8s.io/apiserver/pkg/authorization/union"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook"
	versionedinformers "k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/node"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

// Config contains the data on how to authorize a request to the Kube API Server
type Config struct {
	// Options for ModeABAC

	// Path to an ABAC policy file.
	PolicyFile string

	// Options for ModeWebhook

	// WebhookRetryBackoff specifies the backoff parameters for the authorization webhook retry logic.
	// This allows us to configure the sleep time at each iteration and the maximum number of retries allowed
	// before we fail the webhook call in order to limit the fan out that ensues when the system is degraded.
	WebhookRetryBackoff *wait.Backoff

	VersionedInformerFactory versionedinformers.SharedInformerFactory

	// Optional field, custom dial function used to connect to webhook
	CustomDial utilnet.DialFunc

	// AuthorizationConfiguration stores the configuration for the Authorizer chain
	// It will deprecate most of the above flags when GA
	AuthorizationConfiguration *authzconfig.AuthorizationConfiguration
}

// New returns the right sort of union of multiple authorizer.Authorizer objects
// based on the authorizationMode or an error.
func (config Config) New() (authorizer.Authorizer, authorizer.RuleResolver, error) {
	if len(config.AuthorizationConfiguration.Authorizers) == 0 {
		return nil, nil, fmt.Errorf("at least one authorization mode must be passed")
	}

	r := &reloadableAuthorizerResolver{
		initialConfig: config,
	}

	// Build and store authorizers which will persist across reloads
	for _, configuredAuthorizer := range config.AuthorizationConfiguration.Authorizers {
		// Keep cases in sync with constant list in k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes/modes.go.
		switch configuredAuthorizer.Type {
		case authzconfig.AuthorizerType(modes.ModeNode):
			node.RegisterMetrics()
			graph := node.NewGraph()
			node.AddGraphEventHandlers(
				graph,
				config.VersionedInformerFactory.Core().V1().Nodes(),
				config.VersionedInformerFactory.Core().V1().Pods(),
				config.VersionedInformerFactory.Core().V1().PersistentVolumes(),
				config.VersionedInformerFactory.Storage().V1().VolumeAttachments(),
			)
			r.nodeAuthorizer = node.NewAuthorizer(graph, nodeidentifier.NewDefaultNodeIdentifier(), bootstrappolicy.NodeRules())

		case authzconfig.AuthorizerType(modes.ModeABAC):
			var err error
			r.abacAuthorizer, err = abac.NewFromFile(config.PolicyFile)
			if err != nil {
				return nil, nil, err
			}
		case authzconfig.AuthorizerType(modes.ModeRBAC):
			r.rbacAuthorizer = rbac.New(
				&rbac.RoleGetter{Lister: config.VersionedInformerFactory.Rbac().V1().Roles().Lister()},
				&rbac.RoleBindingLister{Lister: config.VersionedInformerFactory.Rbac().V1().RoleBindings().Lister()},
				&rbac.ClusterRoleGetter{Lister: config.VersionedInformerFactory.Rbac().V1().ClusterRoles().Lister()},
				&rbac.ClusterRoleBindingLister{Lister: config.VersionedInformerFactory.Rbac().V1().ClusterRoleBindings().Lister()},
			)
		}
	}

	// Construct the authorizers / ruleResolvers for the given configuration
	authorizer, ruleResolver, err := r.newForConfig(r.initialConfig.AuthorizationConfiguration)
	if err != nil {
		return nil, nil, err
	}
	r.current.Store(&authorizerResolver{
		authorizer:   authorizer,
		ruleResolver: ruleResolver,
	})

	return r, r, nil
}

// newForConfig constructs
func (r *reloadableAuthorizerResolver) newForConfig(authzConfig *authzconfig.AuthorizationConfiguration) (authorizer.Authorizer, authorizer.RuleResolver, error) {
	if len(authzConfig.Authorizers) == 0 {
		return nil, nil, fmt.Errorf("at least one authorization mode must be passed")
	}

	var (
		authorizers   []authorizer.Authorizer
		ruleResolvers []authorizer.RuleResolver
	)

	// Add SystemPrivilegedGroup as an authorizing group
	superuserAuthorizer := authorizerfactory.NewPrivilegedGroups(user.SystemPrivilegedGroup)
	authorizers = append(authorizers, superuserAuthorizer)

	for _, configuredAuthorizer := range authzConfig.Authorizers {
		// Keep cases in sync with constant list in k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes/modes.go.
		switch configuredAuthorizer.Type {
		case authzconfig.AuthorizerType(modes.ModeNode):
			if r.nodeAuthorizer == nil {
				return nil, nil, fmt.Errorf("nil nodeAuthorizer")
			}
			authorizers = append(authorizers, r.nodeAuthorizer)
			ruleResolvers = append(ruleResolvers, r.nodeAuthorizer)
		case authzconfig.AuthorizerType(modes.ModeAlwaysAllow):
			alwaysAllowAuthorizer := authorizerfactory.NewAlwaysAllowAuthorizer()
			authorizers = append(authorizers, alwaysAllowAuthorizer)
			ruleResolvers = append(ruleResolvers, alwaysAllowAuthorizer)
		case authzconfig.AuthorizerType(modes.ModeAlwaysDeny):
			alwaysDenyAuthorizer := authorizerfactory.NewAlwaysDenyAuthorizer()
			authorizers = append(authorizers, alwaysDenyAuthorizer)
			ruleResolvers = append(ruleResolvers, alwaysDenyAuthorizer)
		case authzconfig.AuthorizerType(modes.ModeABAC):
			if r.abacAuthorizer == nil {
				return nil, nil, fmt.Errorf("nil abacAuthorizer")
			}
			authorizers = append(authorizers, r.abacAuthorizer)
			ruleResolvers = append(ruleResolvers, r.abacAuthorizer)
		case authzconfig.AuthorizerType(modes.ModeWebhook):
			if r.initialConfig.WebhookRetryBackoff == nil {
				return nil, nil, errors.New("retry backoff parameters for authorization webhook has not been specified")
			}
			clientConfig, err := webhookutil.LoadKubeconfig(*configuredAuthorizer.Webhook.ConnectionInfo.KubeConfigFile, r.initialConfig.CustomDial)
			if err != nil {
				return nil, nil, err
			}
			var decisionOnError authorizer.Decision
			switch configuredAuthorizer.Webhook.FailurePolicy {
			case authzconfig.FailurePolicyNoOpinion:
				decisionOnError = authorizer.DecisionNoOpinion
			case authzconfig.FailurePolicyDeny:
				decisionOnError = authorizer.DecisionDeny
			default:
				return nil, nil, fmt.Errorf("unknown failurePolicy %q", configuredAuthorizer.Webhook.FailurePolicy)
			}
			webhookAuthorizer, err := webhook.New(clientConfig,
				configuredAuthorizer.Webhook.SubjectAccessReviewVersion,
				configuredAuthorizer.Webhook.AuthorizedTTL.Duration,
				configuredAuthorizer.Webhook.UnauthorizedTTL.Duration,
				*r.initialConfig.WebhookRetryBackoff,
				decisionOnError,
				configuredAuthorizer.Webhook.MatchConditions,
			)
			if err != nil {
				return nil, nil, err
			}
			authorizers = append(authorizers, webhookAuthorizer)
			ruleResolvers = append(ruleResolvers, webhookAuthorizer)
		case authzconfig.AuthorizerType(modes.ModeRBAC):
			if r.rbacAuthorizer == nil {
				return nil, nil, fmt.Errorf("nil rbacAuthorizer")
			}
			authorizers = append(authorizers, r.rbacAuthorizer)
			ruleResolvers = append(ruleResolvers, r.rbacAuthorizer)
		default:
			return nil, nil, fmt.Errorf("unknown authorization mode %s specified", configuredAuthorizer.Type)
		}
	}

	return union.New(authorizers...), union.NewRuleResolvers(ruleResolvers...), nil
}

// RepeatableAuthorizerTypes is the list of Authorizer that can be repeated in the Authorization Config
var repeatableAuthorizerTypes = []string{modes.ModeWebhook}

// GetNameForAuthorizerMode returns the name to be set for the mode in AuthorizationConfiguration
// For now, lower cases the mode name
func GetNameForAuthorizerMode(mode string) string {
	return strings.ToLower(mode)
}

func LoadAndValidateFile(configFile string, requireNonWebhookTypes sets.Set[authzconfig.AuthorizerType]) (*authzconfig.AuthorizationConfiguration, error) {
	// load the file and check for errors
	authorizationConfiguration, err := load.LoadFromFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load AuthorizationConfiguration from file: %v", err)
	}

	// validate the file and return any error
	if errors := validation.ValidateAuthorizationConfiguration(nil, authorizationConfiguration,
		sets.NewString(modes.AuthorizationModeChoices...),
		sets.NewString(repeatableAuthorizerTypes...),
	); len(errors) != 0 {
		return nil, fmt.Errorf(errors.ToAggregate().Error())
	}

	// test to check if the authorizer names passed conform to the authorizers for type!=Webhook
	// this test is only for kube-apiserver and hence checked here
	// it preserves compatibility with o.buildAuthorizationConfiguration
	var allErrors []error
	seenModes := sets.New[authzconfig.AuthorizerType]()
	for _, authorizer := range authorizationConfiguration.Authorizers {
		if string(authorizer.Type) == modes.ModeWebhook {
			continue
		}
		seenModes.Insert(authorizer.Type)

		expectedName := GetNameForAuthorizerMode(string(authorizer.Type))
		if expectedName != authorizer.Name {
			allErrors = append(allErrors, fmt.Errorf("expected name %s for authorizer %s instead of %s", expectedName, authorizer.Type, authorizer.Name))
		}

	}

	if missingTypes := requireNonWebhookTypes.Difference(seenModes); missingTypes.Len() > 0 {
		allErrors = append(allErrors, fmt.Errorf("missing required types: %v", sets.List(missingTypes)))
	}

	if len(allErrors) > 0 {
		return nil, utilerrors.NewAggregate(allErrors)
	}

	return authorizationConfiguration, nil
}
