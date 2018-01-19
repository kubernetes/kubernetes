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

// Package mutating delegates admission checks to dynamically configured
// mutating webhooks.
package mutating

import (
	"context"
	"fmt"
	"io"
	"time"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/golang/glog"

	admissionv1beta1 "k8s.io/api/admission/v1beta1"
	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
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
	PluginName = "MutatingAdmissionWebhook"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(configFile io.Reader) (admission.Interface, error) {
		plugin, err := NewMutatingWebhook(configFile)
		if err != nil {
			return nil, err
		}

		return plugin, nil
	})
}

// WebhookSource can list dynamic webhook plugins.
type WebhookSource interface {
	Run(stopCh <-chan struct{})
	Webhooks() (*v1beta1.MutatingWebhookConfiguration, error)
}

// NewMutatingWebhook returns a generic admission webhook plugin.
func NewMutatingWebhook(configFile io.Reader) (*MutatingWebhook, error) {
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

	return &MutatingWebhook{
		Handler: admission.NewHandler(
			admission.Connect,
			admission.Create,
			admission.Delete,
			admission.Update,
		),
		clientManager: cm,
	}, nil
}

var _ admission.MutationInterface = &MutatingWebhook{}

// MutatingWebhook is an implementation of admission.Interface.
type MutatingWebhook struct {
	*admission.Handler
	hookSource       WebhookSource
	namespaceMatcher namespace.Matcher
	clientManager    config.ClientManager
	convertor        versioned.Convertor
	defaulter        runtime.ObjectDefaulter
	jsonSerializer   runtime.Serializer
}

var (
	_ = genericadmissioninit.WantsExternalKubeClientSet(&MutatingWebhook{})
)

// TODO find a better way wire this, but keep this pull small for now.
func (a *MutatingWebhook) SetAuthenticationInfoResolverWrapper(wrapper config.AuthenticationInfoResolverWrapper) {
	a.clientManager.SetAuthenticationInfoResolverWrapper(wrapper)
}

// SetServiceResolver sets a service resolver for the webhook admission plugin.
// Passing a nil resolver does not have an effect, instead a default one will be used.
func (a *MutatingWebhook) SetServiceResolver(sr config.ServiceResolver) {
	a.clientManager.SetServiceResolver(sr)
}

// SetScheme sets a serializer(NegotiatedSerializer) which is derived from the scheme
func (a *MutatingWebhook) SetScheme(scheme *runtime.Scheme) {
	if scheme != nil {
		a.clientManager.SetNegotiatedSerializer(serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: serializer.NewCodecFactory(scheme).LegacyCodec(admissionv1beta1.SchemeGroupVersion),
		}))
		a.convertor.Scheme = scheme
		a.defaulter = scheme
		a.jsonSerializer = json.NewSerializer(json.DefaultMetaFactory, scheme, scheme, false)
	}
}

// WantsExternalKubeClientSet defines a function which sets external ClientSet for admission plugins that need it
func (a *MutatingWebhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.namespaceMatcher.Client = client
	a.hookSource = configuration.NewMutatingWebhookConfigurationManager(client.AdmissionregistrationV1beta1().MutatingWebhookConfigurations())
}

// SetExternalKubeInformerFactory implements the WantsExternalKubeInformerFactory interface.
func (a *MutatingWebhook) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	a.namespaceMatcher.NamespaceLister = namespaceInformer.Lister()
	a.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

// ValidateInitialization implements the InitializationValidator interface.
func (a *MutatingWebhook) ValidateInitialization() error {
	if a.hookSource == nil {
		return fmt.Errorf("MutatingWebhook admission plugin requires a Kubernetes client to be provided")
	}
	if a.jsonSerializer == nil {
		return fmt.Errorf("MutatingWebhook admission plugin's jsonSerializer is not properly setup")
	}
	if err := a.namespaceMatcher.Validate(); err != nil {
		return fmt.Errorf("MutatingWebhook.namespaceMatcher is not properly setup: %v", err)
	}
	if err := a.clientManager.Validate(); err != nil {
		return fmt.Errorf("MutatingWebhook.clientManager is not properly setup: %v", err)
	}
	if err := a.convertor.Validate(); err != nil {
		return fmt.Errorf("MutatingWebhook.convertor is not properly setup: %v", err)
	}
	if a.defaulter == nil {
		return fmt.Errorf("MutatingWebhook.defaulter is not properly setup")
	}
	go a.hookSource.Run(wait.NeverStop)
	return nil
}

func (a *MutatingWebhook) loadConfiguration(attr admission.Attributes) (*v1beta1.MutatingWebhookConfiguration, error) {
	hookConfig, err := a.hookSource.Webhooks()
	// if Webhook configuration is disabled, fail open
	if err == configuration.ErrDisabled {
		return &v1beta1.MutatingWebhookConfiguration{}, nil
	}
	if err != nil {
		e := apierrors.NewServerTimeout(attr.GetResource().GroupResource(), string(attr.GetOperation()), 1)
		e.ErrStatus.Message = fmt.Sprintf("Unable to refresh the Webhook configuration: %v", err)
		e.ErrStatus.Reason = "LoadingConfiguration"
		e.ErrStatus.Details.Causes = append(e.ErrStatus.Details.Causes, metav1.StatusCause{
			Type:    "MutatingWebhookConfigurationFailure",
			Message: "An error has occurred while refreshing the MutatingWebhook configuration, no resources can be created/updated/deleted/connected until a refresh succeeds.",
		})
		return nil, e
	}
	return hookConfig, nil
}

// Admit makes an admission decision based on the request attributes.
func (a *MutatingWebhook) Admit(attr admission.Attributes) error {
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

	for _, hook := range relevantHooks {
		t := time.Now()
		err := a.callAttrMutatingHook(ctx, hook, versionedAttr)
		admissionmetrics.Metrics.ObserveWebhook(time.Since(t), err != nil, attr, "admit", hook.Name)
		if err == nil {
			continue
		}

		ignoreClientCallFailures := hook.FailurePolicy != nil && *hook.FailurePolicy == v1beta1.Ignore
		if callErr, ok := err.(*webhookerrors.ErrCallingWebhook); ok {
			if ignoreClientCallFailures {
				glog.Warningf("Failed calling webhook, failing open %v: %v", hook.Name, callErr)
				utilruntime.HandleError(callErr)
				continue
			}
			glog.Warningf("Failed calling webhook, failing closed %v: %v", hook.Name, err)
		}
		return apierrors.NewInternalError(err)
	}

	// convert attr.Object to the internal version
	return a.convertor.Convert(versionedAttr.Object, attr.GetObject())
}

// TODO: factor into a common place along with the validating webhook version.
func (a *MutatingWebhook) shouldCallHook(h *v1beta1.Webhook, attr admission.Attributes) (bool, *apierrors.StatusError) {
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

// note that callAttrMutatingHook updates attr
func (a *MutatingWebhook) callAttrMutatingHook(ctx context.Context, h *v1beta1.Webhook, attr versioned.Attributes) error {
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

	if !response.Response.Allowed {
		return webhookerrors.ToStatusErr(h.Name, response.Response.Result)
	}

	patchJS := response.Response.Patch
	if len(patchJS) == 0 {
		return nil
	}
	patchObj, err := jsonpatch.DecodePatch(patchJS)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	objJS, err := runtime.Encode(a.jsonSerializer, attr.Object)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	patchedJS, err := patchObj.Apply(objJS)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	if _, _, err := a.jsonSerializer.Decode(patchedJS, nil, attr.Object); err != nil {
		return apierrors.NewInternalError(err)
	}
	a.defaulter.Default(attr.Object)
	return nil
}
