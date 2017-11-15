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

	admissionv1alpha1 "k8s.io/api/admission/v1alpha1"
	"k8s.io/api/admissionregistration/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/configuration"
	genericadmissioninit "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/rules"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
)

const (
	// Name of admission plug-in
	PluginName = "GenericAdmissionWebhook"
)

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

	return &GenericAdmissionWebhook{
		Handler: admission.NewHandler(
			admission.Connect,
			admission.Create,
			admission.Delete,
			admission.Update,
		),
		clientManager: cm,
	}, nil
}

// GenericAdmissionWebhook is an implementation of admission.Interface.
type GenericAdmissionWebhook struct {
	*admission.Handler
	hookSource      WebhookSource
	namespaceLister corelisters.NamespaceLister
	client          clientset.Interface
	convertor       runtime.ObjectConvertor
	creator         runtime.ObjectCreater
	clientManager   config.ClientManager
}

var (
	_ = genericadmissioninit.WantsExternalKubeClientSet(&GenericAdmissionWebhook{})
)

// TODO find a better way wire this, but keep this pull small for now.
func (a *GenericAdmissionWebhook) SetAuthenticationInfoResolverWrapper(wrapper config.AuthenticationInfoResolverWrapper) {
	a.clientManager.SetAuthenticationInfoResolverWrapper(wrapper)
}

// SetServiceResolver sets a service resolver for the webhook admission plugin.
// Passing a nil resolver does not have an effect, instead a default one will be used.
func (a *GenericAdmissionWebhook) SetServiceResolver(sr config.ServiceResolver) {
	a.clientManager.SetServiceResolver(sr)
}

// SetScheme sets a serializer(NegotiatedSerializer) which is derived from the scheme
func (a *GenericAdmissionWebhook) SetScheme(scheme *runtime.Scheme) {
	if scheme != nil {
		a.clientManager.SetNegotiatedSerializer(serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: serializer.NewCodecFactory(scheme).LegacyCodec(admissionv1alpha1.SchemeGroupVersion),
		}))
		a.convertor = scheme
		a.creator = scheme
	}
}

// WantsExternalKubeClientSet defines a function which sets external ClientSet for admission plugins that need it
func (a *GenericAdmissionWebhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.client = client
	a.hookSource = configuration.NewValidatingWebhookConfigurationManager(client.AdmissionregistrationV1alpha1().ValidatingWebhookConfigurations())
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (a *GenericAdmissionWebhook) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	a.namespaceLister = namespaceInformer.Lister()
	a.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

// ValidateInitialization implements the InitializationValidator interface.
func (a *GenericAdmissionWebhook) ValidateInitialization() error {
	if a.hookSource == nil {
		return fmt.Errorf("the GenericAdmissionWebhook admission plugin requires a Kubernetes client to be provided")
	}
	if a.namespaceLister == nil {
		return fmt.Errorf("the GenericAdmissionWebhook admission plugin requires a namespaceLister")
	}
	if err := a.clientManager.Validate(); err != nil {
		return fmt.Errorf("the GenericAdmissionWebhook.clientManager is not properly setup: %v", err)
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

// TODO: move this object to a common package
type versionedAttributes struct {
	admission.Attributes
	oldObject runtime.Object
	object    runtime.Object
}

func (v versionedAttributes) GetObject() runtime.Object {
	return v.object
}

func (v versionedAttributes) GetOldObject() runtime.Object {
	return v.oldObject
}

// TODO: move this method to a common package
func (a *GenericAdmissionWebhook) convertToGVK(obj runtime.Object, gvk schema.GroupVersionKind) (runtime.Object, error) {
	// Unlike other resources, custom resources do not have internal version, so
	// if obj is a custom resource, it should not need conversion.
	if obj.GetObjectKind().GroupVersionKind() == gvk {
		return obj, nil
	}
	out, err := a.creator.New(gvk)
	if err != nil {
		return nil, err
	}
	err = a.convertor.Convert(obj, out, nil)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Admit makes an admission decision based on the request attributes.
func (a *GenericAdmissionWebhook) Admit(attr admission.Attributes) error {
	hookConfig, err := a.loadConfiguration(attr)
	if err != nil {
		return err
	}
	hooks := hookConfig.Webhooks
	ctx := context.TODO()

	var relevantHooks []*v1alpha1.Webhook
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
	versionedAttr := versionedAttributes{
		Attributes: attr,
	}
	if oldObj := attr.GetOldObject(); oldObj != nil {
		out, err := a.convertToGVK(oldObj, attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttr.oldObject = out
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := a.convertToGVK(obj, attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
		versionedAttr.object = out
	}

	wg := sync.WaitGroup{}
	errCh := make(chan error, len(relevantHooks))
	wg.Add(len(relevantHooks))
	for i := range relevantHooks {
		go func(hook *v1alpha1.Webhook) {
			defer wg.Done()

			t := time.Now()
			err := a.callHook(ctx, hook, versionedAttr)
			admission.Metrics.ObserveWebhook(time.Since(t), err != nil, hook, attr)
			if err == nil {
				return
			}

			ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1alpha1.Ignore
			if callErr, ok := err.(*config.ErrCallingWebhook); ok {
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

// TODO: move this method to a common package
func (a *GenericAdmissionWebhook) getNamespaceLabels(attr admission.Attributes) (map[string]string, error) {
	// If the request itself is creating or updating a namespace, then get the
	// labels from attr.Object, because namespaceLister doesn't have the latest
	// namespace yet.
	//
	// However, if the request is deleting a namespace, then get the label from
	// the namespace in the namespaceLister, because a delete request is not
	// going to change the object, and attr.Object will be a DeleteOptions
	// rather than a namespace object.
	if attr.GetResource().Resource == "namespaces" &&
		len(attr.GetSubresource()) == 0 &&
		(attr.GetOperation() == admission.Create || attr.GetOperation() == admission.Update) {
		accessor, err := meta.Accessor(attr.GetObject())
		if err != nil {
			return nil, err
		}
		return accessor.GetLabels(), nil
	}

	namespaceName := attr.GetNamespace()
	namespace, err := a.namespaceLister.Get(namespaceName)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	if apierrors.IsNotFound(err) {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespace, err = a.client.Core().Namespaces().Get(namespaceName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
	}
	return namespace.Labels, nil
}

// TODO: move this method to a common package
// whether the request is exempted by the webhook because of the
// namespaceSelector of the webhook.
func (a *GenericAdmissionWebhook) exemptedByNamespaceSelector(h *v1alpha1.Webhook, attr admission.Attributes) (bool, *apierrors.StatusError) {
	namespaceName := attr.GetNamespace()
	if len(namespaceName) == 0 && attr.GetResource().Resource != "namespaces" {
		// If the request is about a cluster scoped resource, and it is not a
		// namespace, it is exempted from all webhooks for now.
		// TODO: figure out a way selective exempt cluster scoped resources.
		// Also update the comment in types.go
		return true, nil
	}
	namespaceLabels, err := a.getNamespaceLabels(attr)
	// this means the namespace is not found, for backwards compatibility,
	// return a 404
	if apierrors.IsNotFound(err) {
		status, ok := err.(apierrors.APIStatus)
		if !ok {
			return false, apierrors.NewInternalError(err)
		}
		return false, &apierrors.StatusError{status.Status()}
	}
	if err != nil {
		return false, apierrors.NewInternalError(err)
	}
	// TODO: adding an LRU cache to cache the translation
	selector, err := metav1.LabelSelectorAsSelector(h.NamespaceSelector)
	if err != nil {
		return false, apierrors.NewInternalError(err)
	}
	return !selector.Matches(labels.Set(namespaceLabels)), nil
}

// TODO: move this method to a common package
func (a *GenericAdmissionWebhook) shouldCallHook(h *v1alpha1.Webhook, attr admission.Attributes) (bool, *apierrors.StatusError) {
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

	excluded, err := a.exemptedByNamespaceSelector(h, attr)
	if err != nil {
		return false, err
	}
	return !excluded, nil
}

func (a *GenericAdmissionWebhook) callHook(ctx context.Context, h *v1alpha1.Webhook, attr admission.Attributes) error {
	// Make the webhook request
	request := createAdmissionReview(attr)
	client, err := a.clientManager.HookClient(h)
	if err != nil {
		return &config.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	response := &admissionv1alpha1.AdmissionReview{}
	if err := client.Post().Context(ctx).Body(&request).Do().Into(response); err != nil {
		return &config.ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	if response.Status.Allowed {
		return nil
	}

	return toStatusErr(h.Name, response.Status.Result)
}

// TODO: move this function to a common package
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
