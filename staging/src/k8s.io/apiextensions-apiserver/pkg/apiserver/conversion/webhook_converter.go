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

package conversion

import (
	"context"
	"errors"
	"fmt"
	"time"

	"go.opentelemetry.io/otel/attribute"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/tracing"
)

type webhookConverterFactory struct {
	clientManager webhook.ClientManager
}

func newWebhookConverterFactory(serviceResolver webhook.ServiceResolver, authResolverWrapper webhook.AuthenticationInfoResolverWrapper) (*webhookConverterFactory, error) {
	clientManager, err := webhook.NewClientManager(
		[]schema.GroupVersion{v1.SchemeGroupVersion, v1beta1.SchemeGroupVersion},
		v1beta1.AddToScheme,
		v1.AddToScheme,
	)
	if err != nil {
		return nil, err
	}
	authInfoResolver, err := webhook.NewDefaultAuthenticationInfoResolver("")
	if err != nil {
		return nil, err
	}
	// Set defaults which may be overridden later.
	clientManager.SetAuthenticationInfoResolver(authInfoResolver)
	clientManager.SetAuthenticationInfoResolverWrapper(authResolverWrapper)
	clientManager.SetServiceResolver(serviceResolver)
	return &webhookConverterFactory{clientManager}, nil
}

// webhookConverter is a converter that calls an external webhook to do the CR conversion.
type webhookConverter struct {
	clientManager webhook.ClientManager
	restClient    *rest.RESTClient
	name          string
	nopConverter  nopConverter

	conversionReviewVersions []string
}

func webhookClientConfigForCRD(crd *v1.CustomResourceDefinition) *webhook.ClientConfig {
	apiConfig := crd.Spec.Conversion.Webhook.ClientConfig
	ret := webhook.ClientConfig{
		Name:     fmt.Sprintf("conversion_webhook_for_%s", crd.Name),
		CABundle: apiConfig.CABundle,
	}
	if apiConfig.URL != nil {
		ret.URL = *apiConfig.URL
	}
	if apiConfig.Service != nil {
		ret.Service = &webhook.ClientConfigService{
			Name:      apiConfig.Service.Name,
			Namespace: apiConfig.Service.Namespace,
			Port:      *apiConfig.Service.Port,
		}
		if apiConfig.Service.Path != nil {
			ret.Service.Path = *apiConfig.Service.Path
		}
	}
	return &ret
}

var _ CRConverter = &webhookConverter{}

func (f *webhookConverterFactory) NewWebhookConverter(crd *v1.CustomResourceDefinition) (*webhookConverter, error) {
	restClient, err := f.clientManager.HookClient(*webhookClientConfigForCRD(crd))
	if err != nil {
		return nil, err
	}
	return &webhookConverter{
		clientManager: f.clientManager,
		restClient:    restClient,
		name:          crd.Name,
		nopConverter:  nopConverter{},

		conversionReviewVersions: crd.Spec.Conversion.Webhook.ConversionReviewVersions,
	}, nil
}

// createConversionReviewObjects returns ConversionReview request and response objects for the first supported version found in conversionReviewVersions.
func createConversionReviewObjects(conversionReviewVersions []string, objects *unstructured.UnstructuredList, apiVersion string, requestUID types.UID) (request, response runtime.Object, err error) {
	rawObjects := make([]runtime.RawExtension, len(objects.Items))
	for i := range objects.Items {
		rawObjects[i] = runtime.RawExtension{
			Object: &objects.Items[i],
		}
	}

	for _, version := range conversionReviewVersions {
		switch version {
		case v1beta1.SchemeGroupVersion.Version:
			return &v1beta1.ConversionReview{
				Request: &v1beta1.ConversionRequest{
					Objects:           rawObjects,
					DesiredAPIVersion: apiVersion,
					UID:               requestUID,
				},
				Response: &v1beta1.ConversionResponse{},
			}, &v1beta1.ConversionReview{}, nil
		case v1.SchemeGroupVersion.Version:
			return &v1.ConversionReview{
				Request: &v1.ConversionRequest{
					Objects:           rawObjects,
					DesiredAPIVersion: apiVersion,
					UID:               requestUID,
				},
				Response: &v1.ConversionResponse{},
			}, &v1.ConversionReview{}, nil
		}
	}
	return nil, nil, fmt.Errorf("no supported conversion review versions")
}

func getRawExtensionObject(rx runtime.RawExtension) (*unstructured.Unstructured, error) {
	if rx.Object != nil {
		u, ok := rx.Object.(*unstructured.Unstructured)
		if !ok {
			return nil, fmt.Errorf("unexpected type %T", rx.Object)
		}
		return u, nil
	}
	u := unstructured.Unstructured{}
	err := u.UnmarshalJSON(rx.Raw)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// getConvertedObjectsFromResponse validates the response, and returns the converted objects.
// if the response is malformed, an error is returned instead.
// if the response does not indicate success, the error message is returned instead.
func getConvertedObjectsFromResponse(expectedUID types.UID, response runtime.Object) (convertedObjects []runtime.RawExtension, err error) {
	switch response := response.(type) {
	case *v1.ConversionReview:
		// Verify GVK to make sure we decoded what we intended to
		v1GVK := v1.SchemeGroupVersion.WithKind("ConversionReview")
		if response.GroupVersionKind() != v1GVK {
			return nil, fmt.Errorf("expected webhook response of %v, got %v", v1GVK.String(), response.GroupVersionKind().String())
		}

		if response.Response == nil {
			return nil, fmt.Errorf("no response provided")
		}

		// Verify UID to make sure this response was actually meant for the request we sent
		if response.Response.UID != expectedUID {
			return nil, fmt.Errorf("expected response.uid=%q, got %q", expectedUID, response.Response.UID)
		}

		if response.Response.Result.Status != metav1.StatusSuccess {
			// TODO: Return a webhook specific error to be able to convert it to meta.Status
			if len(response.Response.Result.Message) > 0 {
				return nil, errors.New(response.Response.Result.Message)
			}
			return nil, fmt.Errorf("response.result.status was '%s', not 'Success'", response.Response.Result.Status)
		}

		return response.Response.ConvertedObjects, nil

	case *v1beta1.ConversionReview:
		// v1beta1 processing did not verify GVK or UID, so skip those for compatibility

		if response.Response == nil {
			return nil, fmt.Errorf("no response provided")
		}

		if response.Response.Result.Status != metav1.StatusSuccess {
			// TODO: Return a webhook specific error to be able to convert it to meta.Status
			if len(response.Response.Result.Message) > 0 {
				return nil, errors.New(response.Response.Result.Message)
			}
			return nil, fmt.Errorf("response.result.status was '%s', not 'Success'", response.Response.Result.Status)
		}

		return response.Response.ConvertedObjects, nil

	default:
		return nil, fmt.Errorf("unrecognized response type: %T", response)
	}
}

func (c *webhookConverter) Convert(in *unstructured.UnstructuredList, toGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
	ctx := context.TODO()
	requestUID := uuid.NewUUID()
	desiredAPIVersion := toGV.String()
	request, response, err := createConversionReviewObjects(c.conversionReviewVersions, in, desiredAPIVersion, requestUID)
	if err != nil {
		return nil, err
	}
	t := time.Now()
	objCount := len(in.Items)

	ctx, span := tracing.Start(ctx, "Call conversion webhook",
		attribute.String("custom-resource-definition", c.name),
		attribute.String("desired-api-version", desiredAPIVersion),
		attribute.Int("object-count", objCount),
		attribute.String("UID", string(requestUID)))
	// Only log conversion webhook traces that exceed a 8ms per object limit plus a 50ms request overhead allowance.
	// The per object limit uses the SLO for conversion webhooks (~4ms per object) plus time to serialize/deserialize
	// the conversion request on the apiserver side (~4ms per object).
	defer span.End(time.Duration(50+8*objCount) * time.Millisecond)

	// TODO: Figure out if adding one second timeout make sense here.
	r := c.restClient.Post().Body(request).Do(ctx)
	if err := r.Into(response); err != nil {
		Metrics.ObserveConversionWebhookFailure(ctx, time.Since(t), ConversionWebhookCallFailure)
		// TODO: Return a webhook specific error to be able to convert it to meta.Status
		return nil, fmt.Errorf("conversion webhook for %v failed: %v", in.GetObjectKind().GroupVersionKind(), err)
	}
	span.AddEvent("Request completed")

	convertedObjects, err := getConvertedObjectsFromResponse(requestUID, response)
	if err != nil {
		Metrics.ObserveConversionWebhookFailure(ctx, time.Since(t), ConversionWebhookCallFailure)
		return nil, fmt.Errorf("conversion webhook for %v failed: %v", in.GetObjectKind().GroupVersionKind(), err)
	}

	out := &unstructured.UnstructuredList{}
	out.Items = make([]unstructured.Unstructured, len(convertedObjects))
	for i := range convertedObjects {
		u, err := getRawExtensionObject(convertedObjects[i])
		if err != nil {
			return nil, err
		}
		out.Items[i] = *u
	}

	Metrics.ObserveConversionWebhookSuccess(ctx, time.Since(t))
	return out, nil
}
