/*
Copyright 2017 The Kubernetes Authors.

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

// Package webhook delegates admission checks to dynamically configured webhooks.
package webhook

import (
	"context"
	"fmt"
	"io"
	"net/url"
	"path"
	"sync"

	"github.com/golang/glog"

	admissionv1alpha1 "k8s.io/api/admission/v1alpha1"
	"k8s.io/api/admissionregistration/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/configuration"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

const (
	// Name of admission plug-in
	PluginName = "GenericAdmissionWebhook"
)

type ErrCallingWebhook struct {
	WebhookName string
	Reason      error
}

func (e *ErrCallingWebhook) Error() string {
	if e.Reason != nil {
		return fmt.Sprintf("failed calling admission webhook %q: %v", e.WebhookName, e.Reason)
	}
	return fmt.Sprintf("failed calling admission webhook %q; no further details available", e.WebhookName)
}

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(configFile io.Reader) (admission.Interface, error) {
		plugin, err := NewGenericAdmissionWebhook(configFile)
		if err != nil {
			return nil, err
		}

		return plugin, nil
	})
}

// WebhookSource can list dynamic webhook plugins.
type WebhookSource interface {
	Run(stopCh <-chan struct{})
	Webhooks() (*v1alpha1.ValidatingWebhookConfiguration, error)
}

// NewGenericAdmissionWebhook returns a generic admission webhook plugin.
func NewGenericAdmissionWebhook(configFile io.Reader) (*GenericAdmissionWebhook, error) {
	kubeconfigFile := ""
	if configFile != nil {
		// TODO: move this to a versioned configuration file format
		var config AdmissionConfig
		d := yaml.NewYAMLOrJSONDecoder(configFile, 4096)
		err := d.Decode(&config)
		if err != nil {
			return nil, err
		}
		kubeconfigFile = config.KubeConfigFile
	}
	authInfoResolver, err := newDefaultAuthenticationInfoResolver(kubeconfigFile)
	if err != nil {
		return nil, err
	}

	return &GenericAdmissionWebhook{
		Handler: admission.NewHandler(
			admission.Connect,
			admission.Create,
			admission.Delete,
			admission.Update,
		),
		authInfoResolver: authInfoResolver,
		serviceResolver:  defaultServiceResolver{},
	}, nil
}

// GenericAdmissionWebhook is an implementation of admission.Interface.
type GenericAdmissionWebhook struct {
	*admission.Handler
	hookSource           WebhookSource
	serviceResolver      ServiceResolver
	negotiatedSerializer runtime.NegotiatedSerializer

	authInfoResolver AuthenticationInfoResolver
}

// serviceResolver knows how to convert a service reference into an actual location.
type ServiceResolver interface {
	ResolveEndpoint(namespace, name string) (*url.URL, error)
}

var (
	_ = genericadmissioninit.WantsExternalKubeClientSet(&GenericAdmissionWebhook{})
)

// TODO find a better way wire this, but keep this pull small for now.
func (a *GenericAdmissionWebhook) SetAuthenticationInfoResolverWrapper(wrapper AuthenticationInfoResolverWrapper) {
	if wrapper != nil {
		a.authInfoResolver = wrapper(a.authInfoResolver)
	}
}

// SetServiceResolver sets a service resolver for the webhook admission plugin.
// Passing a nil resolver does not have an effect, instead a default one will be used.
func (a *GenericAdmissionWebhook) SetServiceResolver(sr ServiceResolver) {
	if sr != nil {
		a.serviceResolver = sr
	}
}

// SetScheme sets a serializer(NegotiatedSerializer) which is derived from the scheme
func (a *GenericAdmissionWebhook) SetScheme(scheme *runtime.Scheme) {
	if scheme != nil {
		a.negotiatedSerializer = serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: serializer.NewCodecFactory(scheme).LegacyCodec(admissionv1alpha1.SchemeGroupVersion),
		})
	}
}

// WantsExternalKubeClientSet defines a function which sets external ClientSet for admission plugins that need it
func (a *GenericAdmissionWebhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.hookSource = configuration.NewValidatingWebhookConfigurationManager(client.AdmissionregistrationV1alpha1().ValidatingWebhookConfigurations())
}

// ValidateInitialization implements the InitializationValidator interface.
func (a *GenericAdmissionWebhook) ValidateInitialization() error {
	if a.hookSource == nil {
		return fmt.Errorf("the GenericAdmissionWebhook admission plugin requires a Kubernetes client to be provided")
	}
	if a.negotiatedSerializer == nil {
		return fmt.Errorf("the GenericAdmissionWebhook admission plugin requires a runtime.Scheme to be provided to derive a serializer")
	}
	go a.hookSource.Run(wait.NeverStop)
	return nil
}

func (a *GenericAdmissionWebhook) loadConfiguration(attr admission.Attributes) (*v1alpha1.ValidatingWebhookConfiguration, error) {
	hookConfig, err := a.hookSource.Webhooks()
	// if Webhook configuration is disabled, fail open
	if err == configuration.ErrDisabled {
		return &v1alpha1.ValidatingWebhookConfiguration{}, nil
	}
	if err != nil {
		e := apierrors.NewServerTimeout(attr.GetResource().GroupResource(), string(attr.GetOperation()), 1)
		e.ErrStatus.Message = fmt.Sprintf("Unable to refresh the Webhook configuration: %v", err)
		e.ErrStatus.Reason = "LoadingConfiguration"
		e.ErrStatus.Details.Causes = append(e.ErrStatus.Details.Causes, metav1.StatusCause{
			Type:    "ValidatingWebhookConfigurationFailure",
			Message: "An error has occurred while refreshing the ValidatingWebhook configuration, no resources can be created/updated/deleted/connected until a refresh succeeds.",
		})
		return nil, e
	}
	return hookConfig, nil
}

// Admit makes an admission decision based on the request attributes.
func (a *GenericAdmissionWebhook) Admit(attr admission.Attributes) error {
	hookConfig, err := a.loadConfiguration(attr)
	if err != nil {
		return err
	}
	hooks := hookConfig.Webhooks
	ctx := context.TODO()

	errCh := make(chan error, len(hooks))
	wg := sync.WaitGroup{}
	wg.Add(len(hooks))
	for i := range hooks {
		go func(hook *v1alpha1.Webhook) {
			defer wg.Done()

			err := a.callHook(ctx, hook, attr)
			if err == nil {
				return
			}

			ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1alpha1.Ignore
			if callErr, ok := err.(*ErrCallingWebhook); ok {
				if ignoreClientCallFailures {
					glog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
					utilruntime.HandleError(callErr)
					// Since we are failing open to begin with, we do not send an error down the channel
					return
				}

				glog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
				errCh <- apierrors.NewInternalError(err)
				return
			}

			glog.Warningf("rejected by webhook %q: %#v", hook.Name, err)
			errCh <- err
		}(&hooks[i])
	}
	wg.Wait()
	close(errCh)

	var errs []error
	for e := range errCh {
		errs = append(errs, e)
	}
	if len(errs) == 0 {
		return nil
	}
	if len(errs) > 1 {
		for i := 1; i < len(errs); i++ {
			// TODO: merge status errors; until then, just return the first one.
			utilruntime.HandleError(errs[i])
		}
	}
	return errs[0]
}

func (a *GenericAdmissionWebhook) callHook(ctx context.Context, h *v1alpha1.Webhook, attr admission.Attributes) error {
	matches := false
	for _, r := range h.Rules {
		m := RuleMatcher{Rule: r, Attr: attr}
		if m.Matches() {
			matches = true
			break
		}
	}
	if !matches {
		return nil
	}

	// Make the webhook request
	request := createAdmissionReview(attr)
	client, err := a.hookClient(h)
	if err != nil {
		return &ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	response := &admissionv1alpha1.AdmissionReview{}
	if err := client.Post().Context(ctx).Body(&request).Do().Into(response); err != nil {
		return &ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	if response.Status.Allowed {
		return nil
	}

	return toStatusErr(h.Name, response.Status.Result)
}

// toStatusErr returns a StatusError with information about the webhook controller
func toStatusErr(name string, result *metav1.Status) *apierrors.StatusError {
	deniedBy := fmt.Sprintf("admission webhook %q denied the request", name)
	const noExp = "without explanation"

	if result == nil {
		result = &metav1.Status{Status: metav1.StatusFailure}
	}

	switch {
	case len(result.Message) > 0:
		result.Message = fmt.Sprintf("%s: %s", deniedBy, result.Message)
	case len(result.Reason) > 0:
		result.Message = fmt.Sprintf("%s: %s", deniedBy, result.Reason)
	default:
		result.Message = fmt.Sprintf("%s %s", deniedBy, noExp)
	}

	return &apierrors.StatusError{
		ErrStatus: *result,
	}
}

func (a *GenericAdmissionWebhook) hookClient(h *v1alpha1.Webhook) (*rest.RESTClient, error) {
	serverName := h.ClientConfig.Service.Name + "." + h.ClientConfig.Service.Namespace + ".svc"
	u, err := a.serviceResolver.ResolveEndpoint(h.ClientConfig.Service.Namespace, h.ClientConfig.Service.Name)
	if err != nil {
		return nil, err
	}

	// TODO: cache these instead of constructing one each time
	restConfig, err := a.authInfoResolver.ClientConfigFor(serverName)
	if err != nil {
		return nil, err
	}
	cfg := rest.CopyConfig(restConfig)
	cfg.Host = u.Host
	cfg.APIPath = path.Join(u.Path, h.ClientConfig.URLPath)
	cfg.TLSClientConfig.ServerName = serverName
	cfg.TLSClientConfig.CAData = h.ClientConfig.CABundle
	cfg.ContentConfig.NegotiatedSerializer = a.negotiatedSerializer
	cfg.ContentConfig.ContentType = runtime.ContentTypeJSON
	return rest.UnversionedRESTClientFor(cfg)
}
