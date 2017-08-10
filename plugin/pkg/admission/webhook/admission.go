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
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/golang/glog"

	admissionv1alpha1 "k8s.io/api/admission/v1alpha1"
	"k8s.io/api/admissionregistration/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api"
	admissionv1alpha1helper "k8s.io/kubernetes/pkg/apis/admission/v1alpha1"
	admissioninit "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/kubeapiserver/admission/configuration"

	// install the clientgo admission API for use with api registry
	_ "k8s.io/kubernetes/pkg/apis/admission/install"
)

var (
	groupVersions = []schema.GroupVersion{
		admissionv1alpha1.SchemeGroupVersion,
	}
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
	plugins.Register("GenericAdmissionWebhook", func(configFile io.Reader) (admission.Interface, error) {
		plugin, err := NewGenericAdmissionWebhook()
		if err != nil {
			return nil, err
		}

		return plugin, nil
	})
}

// WebhookSource can list dynamic webhook plugins.
type WebhookSource interface {
	Run(stopCh <-chan struct{})
	ExternalAdmissionHooks() (*v1alpha1.ExternalAdmissionHookConfiguration, error)
}

// NewGenericAdmissionWebhook returns a generic admission webhook plugin.
func NewGenericAdmissionWebhook() (*GenericAdmissionWebhook, error) {
	return &GenericAdmissionWebhook{
		Handler: admission.NewHandler(
			admission.Connect,
			admission.Create,
			admission.Delete,
			admission.Update,
		),
		negotiatedSerializer: serializer.NegotiatedSerializerWrapper(runtime.SerializerInfo{
			Serializer: api.Codecs.LegacyCodec(admissionv1alpha1.SchemeGroupVersion),
		}),
	}, nil
}

// GenericAdmissionWebhook is an implementation of admission.Interface.
type GenericAdmissionWebhook struct {
	*admission.Handler
	hookSource           WebhookSource
	serviceResolver      admissioninit.ServiceResolver
	negotiatedSerializer runtime.NegotiatedSerializer
	clientCert           []byte
	clientKey            []byte
	proxyTransport       *http.Transport
}

var (
	_ = admissioninit.WantsServiceResolver(&GenericAdmissionWebhook{})
	_ = admissioninit.WantsClientCert(&GenericAdmissionWebhook{})
	_ = admissioninit.WantsExternalKubeClientSet(&GenericAdmissionWebhook{})
)

func (a *GenericAdmissionWebhook) SetProxyTransport(pt *http.Transport) {
	a.proxyTransport = pt
}

func (a *GenericAdmissionWebhook) SetServiceResolver(sr admissioninit.ServiceResolver) {
	a.serviceResolver = sr
}

func (a *GenericAdmissionWebhook) SetClientCert(cert, key []byte) {
	a.clientCert = cert
	a.clientKey = key
}

func (a *GenericAdmissionWebhook) SetExternalKubeClientSet(client clientset.Interface) {
	a.hookSource = configuration.NewExternalAdmissionHookConfigurationManager(client.Admissionregistration().ExternalAdmissionHookConfigurations())
}

func (a *GenericAdmissionWebhook) Validate() error {
	if a.hookSource == nil {
		return fmt.Errorf("the GenericAdmissionWebhook admission plugin requires a Kubernetes client to be provided")
	}
	go a.hookSource.Run(wait.NeverStop)
	return nil
}

func (a *GenericAdmissionWebhook) loadConfiguration(attr admission.Attributes) (*v1alpha1.ExternalAdmissionHookConfiguration, error) {
	hookConfig, err := a.hookSource.ExternalAdmissionHooks()
	// if ExternalAdmissionHook configuration is disabled, fail open
	if err == configuration.ErrDisabled {
		return &v1alpha1.ExternalAdmissionHookConfiguration{}, nil
	}
	if err != nil {
		e := apierrors.NewServerTimeout(attr.GetResource().GroupResource(), string(attr.GetOperation()), 1)
		e.ErrStatus.Message = fmt.Sprintf("Unable to refresh the ExternalAdmissionHook configuration: %v", err)
		e.ErrStatus.Reason = "LoadingConfiguration"
		e.ErrStatus.Details.Causes = append(e.ErrStatus.Details.Causes, metav1.StatusCause{
			Type:    "ExternalAdmissionHookConfigurationFailure",
			Message: "An error has occurred while refreshing the externalAdmissionHook configuration, no resources can be created/updated/deleted/connected until a refresh succeeds.",
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
	hooks := hookConfig.ExternalAdmissionHooks
	ctx := context.TODO()

	errCh := make(chan error, len(hooks))
	wg := sync.WaitGroup{}
	wg.Add(len(hooks))
	for i := range hooks {
		go func(hook *v1alpha1.ExternalAdmissionHook) {
			defer wg.Done()
			if err := a.callHook(ctx, hook, attr); err == nil {
				return
			} else if callErr, ok := err.(*ErrCallingWebhook); ok {
				glog.Warningf("Failed calling webhook %v: %v", hook.Name, callErr)
				utilruntime.HandleError(callErr)
				// Since we are failing open to begin with, we do not send an error down the channel
			} else {
				glog.Warningf("rejected by webhook %v %t: %v", hook.Name, err, err)
				errCh <- err
			}
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

func (a *GenericAdmissionWebhook) callHook(ctx context.Context, h *v1alpha1.ExternalAdmissionHook, attr admission.Attributes) error {
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
	request := admissionv1alpha1helper.NewAdmissionReview(attr)
	client, err := a.hookClient(h)
	if err != nil {
		return &ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}
	if err := client.Post().Context(ctx).Body(&request).Do().Into(&request); err != nil {
		return &ErrCallingWebhook{WebhookName: h.Name, Reason: err}
	}

	if request.Status.Allowed {
		return nil
	}

	if request.Status.Result == nil {
		return fmt.Errorf("admission webhook %q denied the request without explanation", h.Name)
	}

	return &apierrors.StatusError{
		ErrStatus: *request.Status.Result,
	}
}

func (a *GenericAdmissionWebhook) hookClient(h *v1alpha1.ExternalAdmissionHook) (*rest.RESTClient, error) {
	u, err := a.serviceResolver.ResolveEndpoint(h.ClientConfig.Service.Namespace, h.ClientConfig.Service.Name)
	if err != nil {
		return nil, err
	}

	var dial func(network, addr string) (net.Conn, error)
	if a.proxyTransport != nil && a.proxyTransport.Dial != nil {
		dial = a.proxyTransport.Dial
	}

	// TODO: cache these instead of constructing one each time
	cfg := &rest.Config{
		Host:    u.Host,
		APIPath: u.Path,
		TLSClientConfig: rest.TLSClientConfig{
			ServerName: h.ClientConfig.Service.Name + "." + h.ClientConfig.Service.Namespace + ".svc",
			CAData:     h.ClientConfig.CABundle,
			CertData:   a.clientCert,
			KeyData:    a.clientKey,
		},
		UserAgent: "kube-apiserver-admission",
		Timeout:   30 * time.Second,
		ContentConfig: rest.ContentConfig{
			NegotiatedSerializer: a.negotiatedSerializer,
		},
		Dial: dial,
	}
	return rest.UnversionedRESTClientFor(cfg)
}
