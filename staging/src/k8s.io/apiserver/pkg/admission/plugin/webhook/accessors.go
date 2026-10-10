/*
Copyright 2019 The Kubernetes Authors.

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

package webhook

import (
	"sync"

	v1 "k8s.io/api/admissionregistration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/matchconditions"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/object"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

// WebhookAccessor provides a common interface to both mutating and validating webhook types.
type WebhookAccessor interface {
	// This accessor provides the methods needed to support matching against webhook
	// predicates
	namespace.NamespaceSelectorProvider
	object.ObjectSelectorProvider

	// GetUID gets a string that uniquely identifies the webhook.
	GetUID() string

	// GetConfigurationUID gets the UID of the webhook configuration that owns this webhook.
	GetConfigurationUID() types.UID

	// GetConfigurationName gets the name of the webhook configuration that owns this webhook.
	GetConfigurationName() string

	// GetRESTClient gets the webhook client
	GetRESTClient(clientManager *webhookutil.ClientManager) (*rest.RESTClient, error)

	// GetCompiledMatcher gets the compiled matcher object
	GetCompiledMatcher(compiler cel.ConditionCompiler) matchconditions.Matcher

	// GetName gets the webhook Name field. Note that the name is scoped to the webhook
	// configuration and does not provide a globally unique identity, if a unique identity is
	// needed, use GetUID.
	GetName() string
	// GetClientConfig gets the webhook ClientConfig field.
	GetClientConfig() v1.WebhookClientConfig
	// GetRules gets the webhook Rules field.
	GetRules() []v1.RuleWithOperations
	// GetFailurePolicy gets the webhook FailurePolicy field.
	GetFailurePolicy() *v1.FailurePolicyType
	// GetMatchPolicy gets the webhook MatchPolicy field.
	GetMatchPolicy() *v1.MatchPolicyType
	// GetNamespaceSelector gets the webhook NamespaceSelector field.
	GetNamespaceSelector() *metav1.LabelSelector
	// GetObjectSelector gets the webhook ObjectSelector field.
	GetObjectSelector() *metav1.LabelSelector
	// GetSideEffects gets the webhook SideEffects field.
	GetSideEffects() *v1.SideEffectClass
	// GetTimeoutSeconds gets the webhook TimeoutSeconds field.
	GetTimeoutSeconds() *int32
	// GetAdmissionReviewVersions gets the webhook AdmissionReviewVersions field.
	GetAdmissionReviewVersions() []string

	// GetMatchConditions gets the webhook match conditions field.
	GetMatchConditions() []v1.MatchCondition

	// GetMutatingWebhook if the accessor contains a MutatingWebhook, returns it and true, else returns false.
	GetMutatingWebhook() (*v1.MutatingWebhook, bool)
	// GetValidatingWebhook if the accessor contains a ValidatingWebhook, returns it and true, else returns false.
	GetValidatingWebhook() (*v1.ValidatingWebhook, bool)

	// GetType returns the type of the accessor (validate or admit)
	GetType() string

	GetKubeClient() kubernetes.Interface
}

// NewMutatingWebhookAccessor creates an accessor for a MutatingWebhook.
func NewMutatingWebhookAccessor(uid, configurationName string, configurationUID types.UID, h *v1.MutatingWebhook, kubeClient kubernetes.Interface) WebhookAccessor {
	return &mutatingWebhookAccessor{uid: uid, configurationName: configurationName, configurationUID: configurationUID, MutatingWebhook: h, kubeClient: kubeClient}
}

type mutatingWebhookAccessor struct {
	*v1.MutatingWebhook
	uid               string
	configurationName string
	configurationUID  types.UID

	initObjectSelector sync.Once
	objectSelector     labels.Selector
	objectSelectorErr  error

	initNamespaceSelector sync.Once
	namespaceSelector     labels.Selector
	namespaceSelectorErr  error

	initClient sync.Once
	client     *rest.RESTClient
	clientErr  error

	compileMatcher  sync.Once
	compiledMatcher matchconditions.Matcher

	kubeClient kubernetes.Interface
}

func (m *mutatingWebhookAccessor) GetUID() string {
	return m.uid
}

func (m *mutatingWebhookAccessor) GetConfigurationName() string {
	return m.configurationName
}

func (m *mutatingWebhookAccessor) GetConfigurationUID() types.UID {
	return m.configurationUID
}

func (m *mutatingWebhookAccessor) GetRESTClient(clientManager *webhookutil.ClientManager) (*rest.RESTClient, error) {
	m.initClient.Do(func() {
		m.client, m.clientErr = clientManager.HookClient(hookClientConfigForWebhook(m))
	})
	return m.client, m.clientErr
}

func (m *mutatingWebhookAccessor) GetType() string {
	return "admit"
}

func (m *mutatingWebhookAccessor) GetCompiledMatcher(compiler cel.ConditionCompiler) matchconditions.Matcher {
	m.compileMatcher.Do(func() {
		expressions := make([]cel.ExpressionAccessor, len(m.MutatingWebhook.MatchConditions))
		for i, matchCondition := range m.MutatingWebhook.MatchConditions {
			expressions[i] = &matchconditions.MatchCondition{
				Name:       matchCondition.Name,
				Expression: matchCondition.Expression,
			}
		}
		m.compiledMatcher = matchconditions.NewMatcher(compiler.CompileCondition(
			expressions,
			cel.OptionalVariableDeclarations{
				HasParams:     false,
				HasAuthorizer: true,
			},
			environment.StoredExpressions,
		), m.FailurePolicy, "webhook", "admit", m.Name)
	})
	return m.compiledMatcher
}

func (m *mutatingWebhookAccessor) GetParsedNamespaceSelector() (labels.Selector, error) {
	m.initNamespaceSelector.Do(func() {
		m.namespaceSelector, m.namespaceSelectorErr = metav1.LabelSelectorAsSelector(m.NamespaceSelector)
	})
	return m.namespaceSelector, m.namespaceSelectorErr
}

func (m *mutatingWebhookAccessor) GetParsedObjectSelector() (labels.Selector, error) {
	m.initObjectSelector.Do(func() {
		m.objectSelector, m.objectSelectorErr = metav1.LabelSelectorAsSelector(m.ObjectSelector)
	})
	return m.objectSelector, m.objectSelectorErr
}

func (m *mutatingWebhookAccessor) GetName() string {
	return m.Name
}

func (m *mutatingWebhookAccessor) GetClientConfig() v1.WebhookClientConfig {
	return m.ClientConfig
}

func (m *mutatingWebhookAccessor) GetRules() []v1.RuleWithOperations {
	return m.Rules
}

func (m *mutatingWebhookAccessor) GetFailurePolicy() *v1.FailurePolicyType {
	return m.FailurePolicy
}

func (m *mutatingWebhookAccessor) GetMatchPolicy() *v1.MatchPolicyType {
	return m.MatchPolicy
}

func (m *mutatingWebhookAccessor) GetNamespaceSelector() *metav1.LabelSelector {
	return m.NamespaceSelector
}

func (m *mutatingWebhookAccessor) GetObjectSelector() *metav1.LabelSelector {
	return m.ObjectSelector
}

func (m *mutatingWebhookAccessor) GetSideEffects() *v1.SideEffectClass {
	return m.SideEffects
}

func (m *mutatingWebhookAccessor) GetTimeoutSeconds() *int32 {
	return m.TimeoutSeconds
}

func (m *mutatingWebhookAccessor) GetAdmissionReviewVersions() []string {
	return m.AdmissionReviewVersions
}

func (m *mutatingWebhookAccessor) GetMatchConditions() []v1.MatchCondition {
	return m.MatchConditions
}

func (m *mutatingWebhookAccessor) GetMutatingWebhook() (*v1.MutatingWebhook, bool) {
	return m.MutatingWebhook, true
}

func (m *mutatingWebhookAccessor) GetValidatingWebhook() (*v1.ValidatingWebhook, bool) {
	return nil, false
}

func (m *mutatingWebhookAccessor) GetKubeClient() kubernetes.Interface {
	return m.kubeClient
}

// NewValidatingWebhookAccessor creates an accessor for a ValidatingWebhook.
func NewValidatingWebhookAccessor(uid, configurationName string, configurationUID types.UID, h *v1.ValidatingWebhook, kubeClient kubernetes.Interface) WebhookAccessor {
	return &validatingWebhookAccessor{uid: uid, configurationName: configurationName, configurationUID: configurationUID, ValidatingWebhook: h, kubeClient: kubeClient}
}

type validatingWebhookAccessor struct {
	*v1.ValidatingWebhook
	uid               string
	configurationName string
	configurationUID  types.UID

	initObjectSelector sync.Once
	objectSelector     labels.Selector
	objectSelectorErr  error

	initNamespaceSelector sync.Once
	namespaceSelector     labels.Selector
	namespaceSelectorErr  error

	initClient sync.Once
	client     *rest.RESTClient
	clientErr  error

	compileMatcher  sync.Once
	compiledMatcher matchconditions.Matcher

	kubeClient kubernetes.Interface
}

func (v *validatingWebhookAccessor) GetUID() string {
	return v.uid
}

func (v *validatingWebhookAccessor) GetConfigurationName() string {
	return v.configurationName
}

func (v *validatingWebhookAccessor) GetConfigurationUID() types.UID {
	return v.configurationUID
}

func (v *validatingWebhookAccessor) GetRESTClient(clientManager *webhookutil.ClientManager) (*rest.RESTClient, error) {
	v.initClient.Do(func() {
		v.client, v.clientErr = clientManager.HookClient(hookClientConfigForWebhook(v))
	})
	return v.client, v.clientErr
}

func (v *validatingWebhookAccessor) GetCompiledMatcher(compiler cel.ConditionCompiler) matchconditions.Matcher {
	v.compileMatcher.Do(func() {
		expressions := make([]cel.ExpressionAccessor, len(v.ValidatingWebhook.MatchConditions))
		for i, matchCondition := range v.ValidatingWebhook.MatchConditions {
			expressions[i] = &matchconditions.MatchCondition{
				Name:       matchCondition.Name,
				Expression: matchCondition.Expression,
			}
		}
		v.compiledMatcher = matchconditions.NewMatcher(compiler.CompileCondition(
			expressions,
			cel.OptionalVariableDeclarations{
				HasParams:     false,
				HasAuthorizer: true,
			},
			environment.StoredExpressions,
		), v.FailurePolicy, "webhook", "validating", v.Name)
	})
	return v.compiledMatcher
}

func (v *validatingWebhookAccessor) GetParsedNamespaceSelector() (labels.Selector, error) {
	v.initNamespaceSelector.Do(func() {
		v.namespaceSelector, v.namespaceSelectorErr = metav1.LabelSelectorAsSelector(v.NamespaceSelector)
	})
	return v.namespaceSelector, v.namespaceSelectorErr
}

func (v *validatingWebhookAccessor) GetParsedObjectSelector() (labels.Selector, error) {
	v.initObjectSelector.Do(func() {
		v.objectSelector, v.objectSelectorErr = metav1.LabelSelectorAsSelector(v.ObjectSelector)
	})
	return v.objectSelector, v.objectSelectorErr
}

func (m *validatingWebhookAccessor) GetType() string {
	return "validate"
}

func (v *validatingWebhookAccessor) GetName() string {
	return v.Name
}

func (v *validatingWebhookAccessor) GetClientConfig() v1.WebhookClientConfig {
	return v.ClientConfig
}

func (v *validatingWebhookAccessor) GetRules() []v1.RuleWithOperations {
	return v.Rules
}

func (v *validatingWebhookAccessor) GetFailurePolicy() *v1.FailurePolicyType {
	return v.FailurePolicy
}

func (v *validatingWebhookAccessor) GetMatchPolicy() *v1.MatchPolicyType {
	return v.MatchPolicy
}

func (v *validatingWebhookAccessor) GetNamespaceSelector() *metav1.LabelSelector {
	return v.NamespaceSelector
}

func (v *validatingWebhookAccessor) GetObjectSelector() *metav1.LabelSelector {
	return v.ObjectSelector
}

func (v *validatingWebhookAccessor) GetSideEffects() *v1.SideEffectClass {
	return v.SideEffects
}

func (v *validatingWebhookAccessor) GetTimeoutSeconds() *int32 {
	return v.TimeoutSeconds
}

func (v *validatingWebhookAccessor) GetAdmissionReviewVersions() []string {
	return v.AdmissionReviewVersions
}

func (v *validatingWebhookAccessor) GetMatchConditions() []v1.MatchCondition {
	return v.MatchConditions
}

func (v *validatingWebhookAccessor) GetMutatingWebhook() (*v1.MutatingWebhook, bool) {
	return nil, false
}

func (v *validatingWebhookAccessor) GetValidatingWebhook() (*v1.ValidatingWebhook, bool) {
	return v.ValidatingWebhook, true
}

func (v *validatingWebhookAccessor) GetKubeClient() kubernetes.Interface {
	return v.kubeClient
}

// hookClientConfigForWebhook construct a webhookutil.ClientConfig using a WebhookAccessor to access
// MutatingWebhook and ValidatingWebhook API objects. webhookutil.ClientConfig is used
// to create a HookClient and the purpose of the config struct is to share that with other packages
// that need to create a HookClient.
func hookClientConfigForWebhook(w WebhookAccessor) webhookutil.ClientConfig {
	whClientConfig := w.GetClientConfig()

	ret := webhookutil.ClientConfig{Name: w.GetName()} // FIXME: here

	var err error
	if caBundle := whClientConfig.CABundle; caBundle != nil {
		ret.CABundle, err = dynamiccertificates.NewStaticCAContent(w.GetName()+"_trust", caBundle) // FIXME: gotta return error now!
	} else if ctbSelector := whClientConfig.ClusterTrustBundle; ctbSelector != nil {
		ret.CABundle, err = dynamiccertificates.NewDynamicCAContentFromClusterTrustBundles(w.GetKubeClient(), w.GetName()+"_trust", *ctbSelector.SignerName)
	}
	if err != nil {
		klog.Error(err)
		// FIXME: return
	}

	if whClientConfig.URL != nil {
		ret.URL = *whClientConfig.URL
	}
	if whClientConfig.Service != nil {
		ret.Service = &webhookutil.ClientConfigService{
			Name:      whClientConfig.Service.Name,
			Namespace: whClientConfig.Service.Namespace,
		}
		if whClientConfig.Service.Port != nil {
			ret.Service.Port = *whClientConfig.Service.Port
		} else {
			ret.Service.Port = 443
		}
		if whClientConfig.Service.Path != nil {
			ret.Service.Path = *whClientConfig.Service.Path
		}
	}
	return ret
}
