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

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"

	internal "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

type webhookConverterFactory struct {
	clientManager webhook.ClientManager
}

func newWebhookConverterFactory(serviceResolver webhook.ServiceResolver, authResolverWrapper webhook.AuthenticationInfoResolverWrapper) (*webhookConverterFactory, error) {
	clientManager, err := webhook.NewClientManager(v1beta1.SchemeGroupVersion, v1beta1.AddToScheme)
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
	validVersions map[schema.GroupVersion]bool
	clientManager webhook.ClientManager
	restClient    *rest.RESTClient
	name          string
	nopConverter  nopConverter
}

func webhookClientConfigForCRD(crd *internal.CustomResourceDefinition) *webhook.ClientConfig {
	apiConfig := crd.Spec.Conversion.WebhookClientConfig
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
		}
		if apiConfig.Service.Path != nil {
			ret.Service.Path = *apiConfig.Service.Path
		}
	}
	return &ret
}

var _ runtime.ObjectConvertor = &webhookConverter{}

func (f *webhookConverterFactory) NewWebhookConverter(validVersions map[schema.GroupVersion]bool, crd *internal.CustomResourceDefinition) (*webhookConverter, error) {
	restClient, err := f.clientManager.HookClient(*webhookClientConfigForCRD(crd))
	if err != nil {
		return nil, err
	}
	return &webhookConverter{
		clientManager: f.clientManager,
		validVersions: validVersions,
		restClient:    restClient,
		name:          crd.Name,
		nopConverter:  nopConverter{validVersions: validVersions},
	}, nil
}

func (webhookConverter) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
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
		return fmt.Errorf("request to convert CR from an invalid group/version: %s", outGVK.String())
	}
	inGVK := unstructIn.GroupVersionKind()
	if !c.validVersions[inGVK.GroupVersion()] {
		return fmt.Errorf("request to convert CR to an invalid group/version: %s", inGVK.String())
	}

	converted, err := c.ConvertToVersion(unstructIn, outGVK.GroupVersion())
	if err != nil {
		return err
	}
	unstructuredConverted, ok := converted.(runtime.Unstructured)
	if !ok {
		// this should not happened
		return fmt.Errorf("CR conversion failed")
	}
	unstructOut.SetUnstructuredContent(unstructuredConverted.UnstructuredContent())
	return nil
}

func createConversionReview(obj runtime.Object, apiVersion string) *v1beta1.ConversionReview {
	listObj, isList := obj.(*unstructured.UnstructuredList)
	var objects []runtime.RawExtension
	if isList {
		for i := 0; i < len(listObj.Items); i++ {
			// Only sent item for conversion, if the apiVersion is different
			if listObj.Items[i].GetAPIVersion() != apiVersion {
				objects = append(objects, runtime.RawExtension{Object: &listObj.Items[i]})
			}
		}
	} else {
		if obj.GetObjectKind().GroupVersionKind().GroupVersion().String() != apiVersion {
			objects = []runtime.RawExtension{{Object: obj}}
		}
	}
	return &v1beta1.ConversionReview{
		Request: &v1beta1.ConversionRequest{
			Objects:           objects,
			DesiredAPIVersion: apiVersion,
			UID:               uuid.NewUUID(),
		},
		Response: &v1beta1.ConversionResponse{},
	}
}

func getRawExtensionObject(rx runtime.RawExtension) (runtime.Object, error) {
	if rx.Object != nil {
		return rx.Object, nil
	}
	u := unstructured.Unstructured{}
	err := u.UnmarshalJSON(rx.Raw)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

// getTargetGroupVersion returns group/version which should be used to convert in objects to.
// String version of the return item is APIVersion.
func getTargetGroupVersion(in runtime.Object, target runtime.GroupVersioner) (schema.GroupVersion, error) {
	fromGVK := in.GetObjectKind().GroupVersionKind()
	toGVK, ok := target.KindForGroupVersionKinds([]schema.GroupVersionKind{fromGVK})
	if !ok {
		// TODO: should this be a typed error?
		return schema.GroupVersion{}, fmt.Errorf("%v is unstructured and is not suitable for converting to %q", fromGVK.String(), target)
	}
	return toGVK.GroupVersion(), nil
}

func (c *webhookConverter) ConvertToVersion(in runtime.Object, target runtime.GroupVersioner) (runtime.Object, error) {
	// In general, the webhook should not do any defaulting or validation. A special case of that is an empty object
	// conversion that must result an empty object and practically is the same as nopConverter.
	// A smoke test in API machinery calls the converter on empty objects. As this case happens consistently
	// it special cased here not to call webhook converter. The test initiated here:
	// https://github.com/kubernetes/kubernetes/blob/dbb448bbdcb9e440eee57024ffa5f1698956a054/staging/src/k8s.io/apiserver/pkg/storage/cacher/cacher.go#L201
	if isEmptyUnstructuredObject(in) {
		return c.nopConverter.ConvertToVersion(in, target)
	}

	toGV, err := getTargetGroupVersion(in, target)
	if err != nil {
		return nil, err
	}
	if !c.validVersions[toGV] {
		return nil, fmt.Errorf("request to convert CR to an invalid group/version: %s", toGV.String())
	}
	fromGV := in.GetObjectKind().GroupVersionKind().GroupVersion()
	if !c.validVersions[fromGV] {
		return nil, fmt.Errorf("request to convert CR from an invalid group/version: %s", fromGV.String())
	}
	listObj, isList := in.(*unstructured.UnstructuredList)
	if isList {
		for i, item := range listObj.Items {
			fromGV := item.GroupVersionKind().GroupVersion()
			if !c.validVersions[fromGV] {
				return nil, fmt.Errorf("input list has invalid group/version `%v` at `%v` index", fromGV, i)
			}
		}
	}

	request := createConversionReview(in, toGV.String())
	if len(request.Request.Objects) == 0 {
		if !isList {
			return in, nil
		}
		out := listObj.DeepCopy()
		out.SetAPIVersion(toGV.String())
		return out, nil
	}
	response := &v1beta1.ConversionReview{}
	// TODO: Figure out if adding one second timeout make sense here.
	ctx := context.TODO()
	r := c.restClient.Post().Context(ctx).Body(request).Do()
	if err := r.Into(response); err != nil {
		// TODO: Return a webhook specific error to be able to convert it to meta.Status
		return nil, fmt.Errorf("calling to conversion webhook failed for %s: %v", c.name, err)
	}

	if response.Response == nil {
		// TODO: Return a webhook specific error to be able to convert it to meta.Status
		return nil, fmt.Errorf("conversion webhook response was absent for %s", c.name)
	}

	if response.Response.Result.Status != v1.StatusSuccess {
		// TODO return status message as error
		return nil, fmt.Errorf("conversion request failed for %v, Response: %v", in.GetObjectKind(), response)
	}

	if len(response.Response.ConvertedObjects) != len(request.Request.Objects) {
		return nil, fmt.Errorf("expected %v converted objects, got %v", len(request.Request.Objects), len(response.Response.ConvertedObjects))
	}

	if isList {
		convertedList := listObj.DeepCopy()
		// Collection of items sent for conversion is different than list items
		// because only items that needed conversion has been sent.
		convertedIndex := 0
		for i := 0; i < len(listObj.Items); i++ {
			if listObj.Items[i].GetAPIVersion() == toGV.String() {
				// This item has not been sent for conversion, skip it.
				continue
			}
			converted, err := getRawExtensionObject(response.Response.ConvertedObjects[convertedIndex])
			convertedIndex++
			original := listObj.Items[i]
			if err != nil {
				return nil, fmt.Errorf("invalid converted object at index %v: %v", convertedIndex, err)
			}
			if e, a := toGV, converted.GetObjectKind().GroupVersionKind().GroupVersion(); e != a {
				return nil, fmt.Errorf("invalid converted object at index %v: invalid groupVersion, e=%v, a=%v", convertedIndex, e, a)
			}
			if e, a := original.GetObjectKind().GroupVersionKind().Kind, converted.GetObjectKind().GroupVersionKind().Kind; e != a {
				return nil, fmt.Errorf("invalid converted object at index %v: invalid kind, e=%v, a=%v", convertedIndex, e, a)
			}
			unstructConverted, ok := converted.(*unstructured.Unstructured)
			if !ok {
				// this should not happened
				return nil, fmt.Errorf("CR conversion failed")
			}
			if err := validateConvertedObject(&listObj.Items[i], unstructConverted); err != nil {
				return nil, fmt.Errorf("invalid converted object at index %v: %v", convertedIndex, err)
			}
			convertedList.Items[i] = *unstructConverted
		}
		convertedList.SetAPIVersion(toGV.String())
		return convertedList, nil
	}

	if len(response.Response.ConvertedObjects) != 1 {
		// This should not happened
		return nil, fmt.Errorf("CR conversion failed")
	}
	converted, err := getRawExtensionObject(response.Response.ConvertedObjects[0])
	if err != nil {
		return nil, err
	}
	if e, a := toGV, converted.GetObjectKind().GroupVersionKind().GroupVersion(); e != a {
		return nil, fmt.Errorf("invalid converted object: invalid groupVersion, e=%v, a=%v", e, a)
	}
	if e, a := in.GetObjectKind().GroupVersionKind().Kind, converted.GetObjectKind().GroupVersionKind().Kind; e != a {
		return nil, fmt.Errorf("invalid converted object: invalid kind, e=%v, a=%v", e, a)
	}
	unstructConverted, ok := converted.(*unstructured.Unstructured)
	if !ok {
		// this should not happened
		return nil, fmt.Errorf("CR conversion failed")
	}
	unstructIn, ok := in.(*unstructured.Unstructured)
	if !ok {
		// this should not happened
		return nil, fmt.Errorf("CR conversion failed")
	}
	if err := validateConvertedObject(unstructIn, unstructConverted); err != nil {
		return nil, fmt.Errorf("invalid converted object: %v", err)
	}
	return converted, nil
}

func validateConvertedObject(unstructIn, unstructOut *unstructured.Unstructured) error {
	if e, a := unstructIn.GetKind(), unstructOut.GetKind(); e != a {
		return fmt.Errorf("must have the same kind: %v != %v", e, a)
	}
	if e, a := unstructIn.GetName(), unstructOut.GetName(); e != a {
		return fmt.Errorf("must have the same name: %v != %v", e, a)
	}
	if e, a := unstructIn.GetNamespace(), unstructOut.GetNamespace(); e != a {
		return fmt.Errorf("must have the same namespace: %v != %v", e, a)
	}
	if e, a := unstructIn.GetUID(), unstructOut.GetUID(); e != a {
		return fmt.Errorf("must have the same UID: %v != %v", e, a)
	}
	return nil
}

// isEmptyUnstructuredObject returns true if in is an empty unstructured object, i.e. an unstructured object that does
// not have any field except apiVersion and kind.
func isEmptyUnstructuredObject(in runtime.Object) bool {
	u, ok := in.(*unstructured.Unstructured)
	if !ok {
		return false
	}
	if len(u.Object) != 2 {
		return false
	}
	if _, ok := u.Object["kind"]; !ok {
		return false
	}
	if _, ok := u.Object["apiVersion"]; !ok {
		return false
	}
	return true
}
