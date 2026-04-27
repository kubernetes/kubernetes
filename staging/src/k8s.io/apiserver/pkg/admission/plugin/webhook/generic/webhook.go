/*
Copyright 2018 The Kubernetes Authors.

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

package generic

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apiserver/pkg/cel/environment"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"

	admissionv1 "k8s.io/api/admission/v1"
	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	v1 "k8s.io/api/admissionregistration/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/cel"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/object"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/predicates/rules"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/features"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
)

// Webhook is an abstract admission plugin with all the infrastructure to define Admit or Validate on-top.
type Webhook struct {
	*admission.Handler

	// Factories for creating webhook sources.
	apiSourceFactory    sourceFactory
	staticSourceFactory StaticSourceFactory

	// Configuration provided via admission config file and initializers.
	staticManifestsDir string // path to static webhook manifest directory (optional)
	apiServerID        string // identity of this API server instance, used for metrics

	// hookSource is the webhook source used at admission time. When staticManifestsDir
	// is configured, this is a composite source combining staticSource (loaded from disk)
	// with apiSource (from the API). Otherwise, it points directly to apiSource.
	hookSource Source
	// apiSource provides webhook configurations from the Kubernetes API (informer-based).
	apiSource Source
	// staticSource holds a reference to only the static (manifest-based) webhook source.
	// This can be used to route requests for excluded resources to static hooks only.
	staticSource ReloadableSource

	// Admission-time dependencies.
	namespaceInformer coreinformers.NamespaceInformer
	clientManager     *webhookutil.ClientManager
	namespaceMatcher  *namespace.Matcher
	objectMatcher     *object.Matcher
	dispatcher        Dispatcher
	filterCompiler    cel.ConditionCompiler
	authorizer        authorizer.Authorizer

	// Lifecycle.
	stopCh <-chan struct{}
}

var (
	_ genericadmissioninit.WantsExternalKubeClientSet = &Webhook{}
	_ genericadmissioninit.WantsDrainedNotification   = &Webhook{}
	_ genericadmissioninit.WantsAPIServerID           = &Webhook{}
	_ admission.Interface                             = &Webhook{}
)

type sourceFactory func(f informers.SharedInformerFactory) Source
type dispatcherFactory func(cm *webhookutil.ClientManager) Dispatcher

// ReloadableSource extends Source with a method to run a reload loop
// that watches for configuration changes and blocks until the context is canceled.
type ReloadableSource interface {
	Source
	// RunReloadLoop watches for configuration changes and reloads when detected.
	// It blocks until ctx is canceled.
	RunReloadLoop(ctx context.Context)
}

// StaticSourceFactory creates a static webhook source from a manifest directory.
// The returned Source should have LoadInitial() already called.
type StaticSourceFactory func(manifestsDir string) (ReloadableSource, error)

// NewWebhook creates a new generic admission webhook.
func NewWebhook(handler *admission.Handler, configFile io.Reader, sourceFactory sourceFactory, dispatcherFactory dispatcherFactory) (*Webhook, error) {
	cfg, err := config.LoadConfig(configFile)
	if err != nil {
		return nil, err
	}

	cm, err := webhookutil.NewClientManager(
		[]schema.GroupVersion{
			admissionv1beta1.SchemeGroupVersion,
			admissionv1.SchemeGroupVersion,
		},
		admissionv1beta1.AddToScheme,
		admissionv1.AddToScheme,
	)
	if err != nil {
		return nil, err
	}
	authInfoResolver, err := webhookutil.NewDefaultAuthenticationInfoResolver(cfg.KubeConfigFile)
	if err != nil {
		return nil, err
	}
	// Set defaults which may be overridden later.
	cm.SetAuthenticationInfoResolver(authInfoResolver)
	cm.SetServiceResolver(webhookutil.NewDefaultServiceResolver())

	return &Webhook{
		Handler:            handler,
		apiSourceFactory:   sourceFactory,
		staticManifestsDir: cfg.StaticManifestsDir,
		clientManager:      &cm,
		namespaceMatcher:   &namespace.Matcher{},
		objectMatcher:      &object.Matcher{},
		dispatcher:         dispatcherFactory(&cm),
		filterCompiler:     cel.NewConditionCompiler(environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())),
	}, nil
}

// SetAuthenticationInfoResolverWrapper sets the
// AuthenticationInfoResolverWrapper.
// TODO find a better way wire this, but keep this pull small for now.
func (a *Webhook) SetAuthenticationInfoResolverWrapper(wrapper webhookutil.AuthenticationInfoResolverWrapper) {
	a.clientManager.SetAuthenticationInfoResolverWrapper(wrapper)
}

// SetServiceResolver sets a service resolver for the webhook admission plugin.
// Passing a nil resolver does not have an effect, instead a default one will be used.
func (a *Webhook) SetServiceResolver(sr webhookutil.ServiceResolver) {
	a.clientManager.SetServiceResolver(sr)
}

// SetStaticSourceFactory sets the factory for creating static webhook sources.
// This should be called before SetExternalKubeInformerFactory.
func (a *Webhook) SetStaticSourceFactory(factory StaticSourceFactory) {
	a.staticSourceFactory = factory
}

// SetAPIServerID implements the WantsAPIServerID interface.
// The API server ID is used for metrics labeling and must be set before
// SetExternalKubeInformerFactory is called.
func (a *Webhook) SetAPIServerID(id string) {
	a.apiServerID = id
}

// GetAPIServerID returns the stored API server ID.
func (a *Webhook) GetAPIServerID() string {
	return a.apiServerID
}

// SetDrainedNotification implements the WantsDrainedNotification interface.
func (a *Webhook) SetDrainedNotification(stopCh <-chan struct{}) {
	a.stopCh = stopCh
}

// SetExternalKubeClientSet implements the WantsExternalKubeInformerFactory interface.
// It sets external ClientSet for admission plugins that need it
func (a *Webhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.namespaceMatcher.Client = client
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (a *Webhook) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	a.namespaceMatcher.NamespaceLister = namespaceInformer.Lister()
	a.namespaceInformer = namespaceInformer

	// Create the API-based source (stored for later use in ValidateInitialization)
	a.apiSource = a.apiSourceFactory(f)
}

func (a *Webhook) SetAuthorizer(authorizer authorizer.Authorizer) {
	a.authorizer = authorizer
}

// ValidateInitialization implements the InitializationValidator interface.
// Static source creation happens here (after all initializers have run) because
// SetManifestLoaders may be called after SetExternalKubeInformerFactory.
func (a *Webhook) ValidateInitialization() error {
	if a.apiSource == nil {
		return fmt.Errorf("kubernetes client is not properly setup")
	}
	if err := a.namespaceMatcher.Validate(); err != nil {
		return fmt.Errorf("namespaceMatcher is not properly setup: %v", err)
	}
	if err := a.clientManager.Validate(); err != nil {
		return fmt.Errorf("clientManager is not properly setup: %v", err)
	}

	// Guard: if static manifests dir is set but feature gate is off, return an error
	if len(a.staticManifestsDir) > 0 && !utilfeature.DefaultFeatureGate.Enabled(features.ManifestBasedAdmissionControlConfig) {
		return fmt.Errorf("static webhook manifests dir %q configured but %s feature gate is not enabled", a.staticManifestsDir, features.ManifestBasedAdmissionControlConfig)
	}

	// Construct hookSource. The static source path (which starts goroutines)
	// is guarded to avoid duplicate construction. The API-only path is a cheap
	// pointer assignment that must reflect the latest apiSource.
	if len(a.staticManifestsDir) > 0 {
		if a.hookSource == nil {
			if a.staticSourceFactory == nil {
				return fmt.Errorf("static webhook manifests configured in %q but no static source factory is set", a.staticManifestsDir)
			}
			if a.stopCh == nil {
				return fmt.Errorf("stopCh not set: WantsDrainedNotification must be called before ValidateInitialization")
			}
			staticSource, err := a.staticSourceFactory(a.staticManifestsDir)
			if err != nil {
				return fmt.Errorf("failed to load static webhook manifests from %q: %w", a.staticManifestsDir, err)
			}
			a.staticSource = staticSource
			// Start the file watcher in a background goroutine, tied to server shutdown
			staticCtx, staticCancel := context.WithCancel(context.Background())
			go func() {
				defer staticCancel()
				<-a.stopCh
			}()
			go staticSource.RunReloadLoop(staticCtx)
			// Use composite source that combines static + API sources
			a.hookSource = NewCompositeWebhookSource(staticSource, a.apiSource)
		}
	} else {
		a.hookSource = a.apiSource
	}

	a.SetReadyFunc(func() bool {
		return a.namespaceInformer.Informer().HasSynced() && a.hookSource.HasSynced()
	})

	return nil
}

// ShouldCallHook returns invocation details if the webhook should be called, nil if the webhook should not be called,
// or an error if an error was encountered during evaluation.
func (a *Webhook) ShouldCallHook(ctx context.Context, h webhook.WebhookAccessor, attr admission.Attributes, o admission.ObjectInterfaces, v VersionedAttributeAccessor) (*WebhookInvocation, *apierrors.StatusError) {
	matches, matchNsErr := a.namespaceMatcher.MatchNamespaceSelector(h, attr)
	// Should not return an error here for webhooks which do not apply to the request, even if err is an unexpected scenario.
	if !matches && matchNsErr == nil {
		return nil, nil
	}

	// Should not return an error here for webhooks which do not apply to the request, even if err is an unexpected scenario.
	matches, matchObjErr := a.objectMatcher.MatchObjectSelector(h, attr)
	if !matches && matchObjErr == nil {
		return nil, nil
	}

	var invocation *WebhookInvocation
	for _, r := range h.GetRules() {
		m := rules.Matcher{Rule: r, Attr: attr}
		if m.Matches() {
			invocation = &WebhookInvocation{
				Webhook:     h,
				Resource:    attr.GetResource(),
				Subresource: attr.GetSubresource(),
				Kind:        attr.GetKind(),
			}
			break
		}
	}
	if invocation == nil && h.GetMatchPolicy() != nil && *h.GetMatchPolicy() == v1.Equivalent {
		attrWithOverride := &attrWithResourceOverride{Attributes: attr}
		equivalents := o.GetEquivalentResourceMapper().EquivalentResourcesFor(attr.GetResource(), attr.GetSubresource())
		// honor earlier rules first
	OuterLoop:
		for _, r := range h.GetRules() {
			// see if the rule matches any of the equivalent resources
			for _, equivalent := range equivalents {
				if equivalent == attr.GetResource() {
					// exclude attr.GetResource(), which we already checked
					continue
				}
				attrWithOverride.resource = equivalent
				m := rules.Matcher{Rule: r, Attr: attrWithOverride}
				if m.Matches() {
					kind := o.GetEquivalentResourceMapper().KindFor(equivalent, attr.GetSubresource())
					if kind.Empty() {
						return nil, apierrors.NewInternalError(fmt.Errorf("unable to convert to %v: unknown kind", equivalent))
					}
					invocation = &WebhookInvocation{
						Webhook:     h,
						Resource:    equivalent,
						Subresource: attr.GetSubresource(),
						Kind:        kind,
					}
					break OuterLoop
				}
			}
		}
	}

	if invocation == nil {
		return nil, nil
	}
	if matchNsErr != nil {
		return nil, matchNsErr
	}
	if matchObjErr != nil {
		return nil, matchObjErr
	}
	matchConditions := h.GetMatchConditions()
	if len(matchConditions) > 0 {
		versionedAttr, err := v.VersionedAttribute(invocation.Kind)
		if err != nil {
			return nil, apierrors.NewInternalError(err)
		}

		matcher := h.GetCompiledMatcher(a.filterCompiler)
		matchResult := matcher.Match(ctx, versionedAttr, nil, a.authorizer)

		if matchResult.Error != nil {
			klog.Warningf("Failed evaluating match conditions, failing closed %v: %v", h.GetName(), matchResult.Error)
			return nil, apierrors.NewForbidden(attr.GetResource().GroupResource(), attr.GetName(), matchResult.Error)
		} else if !matchResult.Matches {
			admissionmetrics.Metrics.ObserveMatchConditionExclusion(ctx, h.GetName(), "webhook", h.GetType(), string(attr.GetOperation()))
			// if no match, always skip webhook
			return nil, nil
		}
	}

	return invocation, nil
}

type attrWithResourceOverride struct {
	admission.Attributes
	resource schema.GroupVersionResource
}

func (a *attrWithResourceOverride) GetResource() schema.GroupVersionResource { return a.resource }

// Dispatch is called by the downstream Validate or Admit methods.
func (a *Webhook) Dispatch(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if rules.IsExemptAdmissionConfigurationResource(attr) {
		// Admission config resources are excluded from API-based webhooks to
		// prevent circular dependencies. However, static (manifest-based) webhooks
		// are safe to evaluate since they don't have self-referential concerns.
		if a.staticSource != nil {
			if !a.staticSource.HasSynced() {
				return admission.NewForbidden(attr, fmt.Errorf("not yet ready to handle request"))
			}
			hooks := a.staticSource.Webhooks()
			return a.dispatcher.Dispatch(ctx, attr, o, hooks)
		}
		return nil
	}
	if !a.WaitForReady() {
		return admission.NewForbidden(attr, fmt.Errorf("not yet ready to handle request"))
	}
	hooks := a.hookSource.Webhooks()
	return a.dispatcher.Dispatch(ctx, attr, o, hooks)
}
