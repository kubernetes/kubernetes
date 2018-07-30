package conversion

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
)

// nopConverter is a converter that only sets the apiVersion fields, but does not real conversion.
type webhookConverter struct {
	validVersions map[schema.GroupVersion]bool
	clientManager config.ClientManager
}

var _ runtime.ObjectConvertor = &webhookConverter{}

func newWebhookConverter(validVersions map[schema.GroupVersion]bool) (*webhookConverter, error) {
	clientManager, err := config.NewClientManager()
	if err != nil {
		return nil, err
	}
	return &webhookConverter{
		clientManager: clientManager,
		validVersions: validVersions,
	}, nil
}

func (webhookConverter) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return "", "", errors.New("unstructured cannot convert field labels")
}

func (c *webhookConverter) Convert(in, out, context interface{}) error {
	unstructIn, ok := in.(*unstructured.Unstructured)
	if !ok {
		return fmt.Errorf("input type %T in not valid for unstructured conversion", in)
	}

	unstructOut, ok := out.(*unstructured.Unstructured)
	if !ok {
		return fmt.Errorf("output type %T in not valid for unstructured conversion", out)
	}

	outGVK := unstructOut.GroupVersionKind()
	if !c.validVersions[outGVK.GroupVersion()] {
		return fmt.Errorf("request to convert CRD from an invalid group/version: %s", outGVK.String())
	}
	inGVK := unstructIn.GroupVersionKind()
	if !c.validVersions[inGVK.GroupVersion()] {
		return fmt.Errorf("request to convert CRD to an invalid group/version: %s", inGVK.String())
	}

	unstructOut.SetUnstructuredContent(unstructIn.UnstructuredContent())
	_, err := c.ConvertToVersion(unstructOut, outGVK.GroupVersion())
	if err != nil {
		return err
	}
	return nil
}

func (c *webhookConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	kind := in.GetObjectKind().GroupVersionKind()
	gvk, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{kind})
	if !ok {
		// TODO: should this be a typed error?
		return nil, fmt.Errorf("%v is unstructured and is not suitable for converting to %q", kind, target)
	}
	if !c.validVersions[gvk.GroupVersion()] {
		return nil, fmt.Errorf("request to convert CRD to an invalid group/version: %s", gvk.String())
	}
	if in.GetObjectKind().GroupVersionKind() == gvk {
		// No conversion is required
		return in, nil
	}

	// Call the webhook

	return in, nil
}


/*
k8s.io/apiserver/pkg/admission/plugin/webhook/mutating/dispatcher.go:81

// note that callAttrMutatingHook updates attr
func (a *mutatingDispatcher) callAttrMutatingHook(ctx context.Context, h *v1beta1.Webhook, attr *generic.VersionedAttributes) error {
	// Make the webhook request
	request := request.CreateAdmissionReview(attr)
	client, err := a.cm.HookClient(h)
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
	objJS, err := runtime.Encode(a.plugin.jsonSerializer, attr.VersionedObject)
	if err != nil {
		return apierrors.NewInternalError(err)
	}
	patchedJS, err := patchObj.Apply(objJS)
	if err != nil {
		return apierrors.NewInternalError(err)
	}

	var newVersionedObject runtime.Object
	if _, ok := attr.VersionedObject.(*unstructured.Unstructured); ok {
		// Custom Resources don't have corresponding Go struct's.
		// They are represented as Unstructured.
		newVersionedObject = &unstructured.Unstructured{}
	} else {
		newVersionedObject, err = a.plugin.scheme.New(attr.GetKind())
		if err != nil {
			return apierrors.NewInternalError(err)
		}
	}
	// TODO: if we have multiple mutating webhooks, we can remember the json
	// instead of encoding and decoding for each one.
	if _, _, err := a.plugin.jsonSerializer.Decode(patchedJS, nil, newVersionedObject); err != nil {
		return apierrors.NewInternalError(err)
	}
	attr.VersionedObject = newVersionedObject
	a.plugin.scheme.Default(attr.VersionedObject)
	return nil
}

 */

/*
k8s.io/apiserver/pkg/admission/plugin/webhook/config/client.go:107

// HookClient get a RESTClient from the cache, or constructs one based on the
// webhook configuration.
func (cm *ClientManager) HookClient(h *v1beta1.Webhook) (*rest.RESTClient, error) {
	cacheKey, err := json.Marshal(h.ClientConfig)
	if err != nil {
		return nil, err
	}
	if client, ok := cm.cache.Get(string(cacheKey)); ok {
		return client.(*rest.RESTClient), nil
	}

	complete := func(cfg *rest.Config) (*rest.RESTClient, error) {
		// Combine CAData from the config with any existing CA bundle provided
		if len(cfg.TLSClientConfig.CAData) > 0 {
			cfg.TLSClientConfig.CAData = append(cfg.TLSClientConfig.CAData, '\n')
		}
		cfg.TLSClientConfig.CAData = append(cfg.TLSClientConfig.CAData, h.ClientConfig.CABundle...)

		cfg.ContentConfig.NegotiatedSerializer = cm.negotiatedSerializer
		cfg.ContentConfig.ContentType = runtime.ContentTypeJSON
		client, err := rest.UnversionedRESTClientFor(cfg)
		if err == nil {
			cm.cache.Add(string(cacheKey), client)
		}
		return client, err
	}

	if svc := h.ClientConfig.Service; svc != nil {
		restConfig, err := cm.authInfoResolver.ClientConfigForService(svc.Name, svc.Namespace)
		if err != nil {
			return nil, err
		}
		cfg := rest.CopyConfig(restConfig)
		serverName := svc.Name + "." + svc.Namespace + ".svc"
		host := serverName + ":443"
		cfg.Host = "https://" + host
		if svc.Path != nil {
			cfg.APIPath = *svc.Path
		}
		// Set the server name if not already set
		if len(cfg.TLSClientConfig.ServerName) == 0 {
			cfg.TLSClientConfig.ServerName = serverName
		}

		delegateDialer := cfg.Dial
		if delegateDialer == nil {
			var d net.Dialer
			delegateDialer = d.DialContext
		}
		cfg.Dial = func(ctx context.Context, network, addr string) (net.Conn, error) {
			if addr == host {
				u, err := cm.serviceResolver.ResolveEndpoint(svc.Namespace, svc.Name)
				if err != nil {
					return nil, err
				}
				addr = u.Host
			}
			return delegateDialer(ctx, network, addr)
		}

		return complete(cfg)
	}

	if h.ClientConfig.URL == nil {
		return nil, &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: ErrNeedServiceOrURL}
	}

	u, err := url.Parse(*h.ClientConfig.URL)
	if err != nil {
		return nil, &webhookerrors.ErrCallingWebhook{WebhookName: h.Name, Reason: fmt.Errorf("Unparsable URL: %v", err)}
	}

	restConfig, err := cm.authInfoResolver.ClientConfigFor(u.Host)
	if err != nil {
		return nil, err
	}

	cfg := rest.CopyConfig(restConfig)
	cfg.Host = u.Scheme + "://" + u.Host
	cfg.APIPath = u.Path

	return complete(cfg)
}

 */