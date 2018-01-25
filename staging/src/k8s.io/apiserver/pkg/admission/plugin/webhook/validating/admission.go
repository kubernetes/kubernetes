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

// Package validating delegates admission checks to dynamically configured
// validating webhooks.
package validating

import (
	"context"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/golang/glog"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/configuration"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
	webhookerrors "k8s.io/apiserver/pkg/admission/plugin/webhook/errors"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/namespace"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/request"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/rules"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/versioned"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
)

const (
	// Name of admission plug-in
	PluginName = "ValidatingAdmissionWebhook"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(configFile io.Reader) (admission.Interface, error) {
		plugin, err := NewValidatingAdmissionWebhook(configFile)
		if err != nil {
			return nil, err
		}

		return plugin, nil
	})
}

// WebhookSource can list dynamic webhook plugins.
type WebhookSource interface {
	Run(stopCh <-chan struct{})
	Webhooks() (*v1beta1.ValidatingWebhookConfiguration, error)
}

// NewValidatingAdmissionWebhook returns a generic admission webhook plugin.
func NewValidatingAdmissionWebhook(configFile io.Reader) (*ValidatingAdmissionWebhook, error) {
	kubeconfigFile, err := config.LoadConfig(configFile)
	if err != nil {
		return nil, err
	}

	cm, err := config.NewClientManager()
	if err != nil {
		return nil, err
	}
	authInfoResolver, err := config.NewDefaultAuthenticationInfoResolver(kubeconfigFile)
	if err != nil {
		return nil, err
	}
	// Set defaults which may be overridden later.
	cm.SetAuthenticationInfoResolver(authInfoResolver)
	cm.SetServiceResolver(config.NewDefaultServiceResolver())

	return &ValidatingAdmissionWebhook{
		Handler: admission.NewHandler(
			admission.Connect,
			admission.Create,
			admission.Delete,
			admission.Update,
		),
		clientManager: cm,
	}, nil
}

var _ admission.ValidationInterface = &ValidatingAdmissionWebhook{}

// ValidatingAdmissionWebhook is an implementation of admission.Interface.
type ValidatingAdmissionWebhook struct {
	*admission.Handler
	hookSource       WebhookSource
	namespaceMatcher namespace.Matcher
	clientManager    config.ClientManager
	convertor        versioned.Convertor
}

var (
	_ = genericadmissioninit.WantsExternalKubeClientSet(&ValidatingAdmissionWebhook{})
)

// TODO find a better way wire this, but keep this pull small for now.
func (a *ValidatingAdmissionWebhook) SetAuthenticationInfoResolverWrapper(wrapper config.AuthenticationInfoResolverWrapper) {
	a.clientManager.SetAuthenticationInfoResolverWrapper(wrapper)
}

// SetServiceResolver sets a service resolver for the webhook admission plugin.
// Passing a nil resolver does not have an effect, instead a default one will be used.
func (a *ValidatingAdmissionWebhook) SetServiceResolver(sr config.ServiceResolver) {
	a.clientManager.SetServiceResolver(sr)
}

// SetScheme sets a serializer(NegotiatedSerializer) which is derived from the scheme
func (a *ValidatingAdmissionWebhook) SetScheme(scheme *runtime.Scheme) {
	if scheme != nil {
		a.clientManager.SetNegotiatedSerializer(serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: serializer.NewCodecFactory(scheme).LegacyCodec(admissionv1beta1.SchemeGroupVersion),
		}))
		a.convertor.Scheme = scheme
	}
}

// WantsExternalKubeClientSet defines a function which sets external ClientSet for admission plugins that need it
func (a *ValidatingAdmissionWebhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.namespaceMatcher.Client = client
	a.hookSource = configuration.NewValidatingWebhookConfigurationManager(client.AdmissionregistrationV1beta1().ValidatingWebhookConfigurations())
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (a *ValidatingAdmissionWebhook) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	a.namespaceMatcher.NamespaceLister = namespaceInformer.Lister()
	a.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

// ValidateInitialization implements the InitializationValidator interface.
func (a *ValidatingAdmissionWebhook) ValidateInitialization() error {
	if a.hookSource == nil {
		return fmt.Errorf("ValidatingAdmissionWebhook admission plugin requires a Kubernetes client to be provided")
	}
	if err := a.namespaceMatcher.Validate(); err != nil {
		return fmt.Errorf("ValidatingAdmissionWebhook.namespaceMatcher is not properly setup: %v", err)
	}
	if err := a.clientManager.Validate(); err != nil {
		return fmt.Errorf("ValidatingAdmissionWebhook.clientManager is not properly setup: %v", err)
	}
	if err := a.convertor.Validate(); err != nil {
		return fmt.Errorf("ValidatingAdmissionWebhook.convertor is not properly setup: %v", err)
	}
	go a.hookSource.Run(wait.NeverStop)
	return nil
}

func (a *ValidatingAdmissionWebhook) loadConfiguration(attr admission.Attributes) (*v1beta1.ValidatingWebhookConfiguration, error) {
	hookConfig, err := a.hookSource.Webhooks()
	// if Webhook configuration is disabled, fail open
	if err == configuration.ErrDisabled {
		return &v1beta1.ValidatingWebhookConfiguration{}, nil
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

// Validate makes an admission decision based on the request attributes.
func (a *ValidatingAdmissionWebhook) Validate(attr admission.Attributes) error {
	hookConfig, err := a.loadConfiguration(attr)
	if err != nil {
		return err
	}
	hooks := hookConfig.Webhooks
	ctx := context.TODO()

	var relevantHooks []*v1beta1.Webhook
	for i := range hooks {
		call, err := a.shouldCallHook(&hooks[i], attr)
		if err != nil {
			return err
		}
		if call {
			relevantHooks = append(relevantHooks, &hooks[i])
		}
	}

	if len(relevantHooks) == 0 {
		// no matching hooks
		return nil
	}

	// convert the object to the external version before sending it to the webhook
	versionedAttr := versioned.Attributes{
		Attributes: attr,
	}
	if oldObj := attr.GetOldObject(); oldObj != nil {
		out, err := a.convertor.ConvertToGVK(oldObj, attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttr.OldObject = out
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := a.convertor.ConvertToGVK(obj, attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttr.Object = out
	}

	wg := sync.WaitGroup{}
	errCh := make(chan error, len(relevantHooks))
	wg.Add(len(relevantHooks))
	for i := range relevantHooks {
		go func(hook *v1beta1.Webhook) {
			defer wg.Done()

			t := time.Now()
			err := a.callHook(ctx, hook, versionedAttr)
			admissionmetrics.Metrics.ObserveWebhook(time.Since(t), err != nil, attr, "validating", hook.Name)
			if err == nil {
				return
			}

			ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1beta1.Ignore
			if callErr, ok := err.(*webhookerrors.ErrCallingWebhook); ok {
				if ignoreClientCallFailures {
					glog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
					utilruntime.HandleError(callErr)
					return
				}

				glog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
				errCh <- apierrors.NewInternalError(err)
				return
			}

			glog.Warningf("rejected by webhook %q: %#v", hook.Name, err)
			errCh <- err
		}(relevantHooks[i])
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

// TODO: factor into a common place along with the validating webhook version.
func (a *ValidatingAdmissionWebhook) shouldCallHook(h *v1beta1.Webhook, attr admission.Attributes) (bool, *apierrors.StatusError) {
	var matches bool
	for _, r := range h.Rules {
		m := rules.Matcher{Rule: r, Attr: attr}
		if m.Matches() {
			matches = true
			break
		}
	}
	if !matches {
		return false, nil
	}

	return a.namespaceMatcher.MatchNamespaceSelector(h, attr)
}

func (a *ValidatingAdmissionWebhook) callHook(ctx context.Context, h *v1beta1.Webhook, attr admission.Attributes) error {
	// Make the webhook request
	request := request.CreateAdmissionReview(attr)
	client, err := a.clientManager.HookClient(h)
	if err != nil {
		return &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	response := &admissionv1beta1.AdmissionReview{}
	if err := client.Post().Context(ctx).Body(&request).Do().Into(response); err != nil {
		return &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	if response.Response == nil {
		return &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Webhook response was absent")}
	}
	if response.Response.Allowed {
		return nil
	}
	return webhookerrors.ToStatusErr(h.Name, response.Response.Result)
}
