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

	internal "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1validation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/rest"
	utiltrace "k8s.io/utils/trace"
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
			Port:      apiConfig.Service.Port,
		}
		if apiConfig.Service.Path != nil {
			ret.Service.Path = *apiConfig.Service.Path
		}
	}
	return &ret
}

var _ crConverterInterface = &webhookConverter{}

func (f *webhookConverterFactory) NewWebhookConverter(crd *internal.CustomResourceDefinition) (*webhookConverter, error) {
	restClient, err := f.clientManager.HookClient(*webhookClientConfigForCRD(crd))
	if err != nil {
		return nil, err
	}
	return &webhookConverter{
		clientManager: f.clientManager,
		restClient:    restClient,
		name:          crd.Name,
		nopConverter:  nopConverter{},

		conversionReviewVersions: crd.Spec.Conversion.ConversionReviewVersions,
	}, nil
}

// hasConversionReviewVersion check whether a version is accepted by a given webhook.
func (c *webhookConverter) hasConversionReviewVersion(v string) bool {
	for _, b := range c.conversionReviewVersions {
		if b == v {
			return true
		}
	}
	return false
}

// getObjectsToConvert returns a list of objects requiring conversion.
// if obj is a list, getObjectsToConvert returns a (potentially empty) list of the items that are not already in the desired version.
// if obj is not a list, and is already in the desired version, getObjectsToConvert returns an empty list.
// if obj is not a list, and is not already in the desired version, getObjectsToConvert returns a list containing only obj.
func getObjectsToConvert(obj runtime.Object, apiVersion string) []runtime.RawExtension {
	listObj, isList := obj.(*unstructured.UnstructuredList)
	var objects []runtime.RawExtension
	if isList {
		for i := range listObj.Items {
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
	return objects
}

// createConversionReviewObjects returns ConversionReview request and response objects for the first supported version found in conversionReviewVersions.
func createConversionReviewObjects(conversionReviewVersions []string, objects []runtime.RawExtension, apiVersion string, requestUID types.UID) (request, response runtime.Object, err error) {
	for _, version := range conversionReviewVersions {
		switch version {
		case v1beta1.SchemeGroupVersion.Version:
			return &v1beta1.ConversionReview{
				Request: &v1beta1.ConversionRequest{
					Objects:           objects,
					DesiredAPIVersion: apiVersion,
					UID:               requestUID,
				},
				Response: &v1beta1.ConversionResponse{},
			}, &v1beta1.ConversionReview{}, nil
		case v1.SchemeGroupVersion.Version:
			return &v1.ConversionReview{
				Request: &v1.ConversionRequest{
					Objects:           objects,
					DesiredAPIVersion: apiVersion,
					UID:               requestUID,
				},
				Response: &v1.ConversionResponse{},
			}, &v1.ConversionReview{}, nil
		}
	}
	return nil, nil, fmt.Errorf("no supported conversion review versions")
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

func (c *webhookConverter) Convert(in runtime.Object, toGV schema.GroupVersion) (runtime.Object, error) {
	// In general, the webhook should not do any defaulting or validation. A special case of that is an empty object
	// conversion that must result an empty object and practically is the same as nopConverter.
	// A smoke test in API machinery calls the converter on empty objects. As this case happens consistently
	// it special cased here not to call webhook converter. The test initiated here:
	// https://github.com/kubernetes/kubernetes/blob/dbb448bbdcb9e440eee57024ffa5f1698956a054/staging/src/k8s.io/apiserver/pkg/storage/cacher/cacher.go#L201
	if isEmptyUnstructuredObject(in) {
		return c.nopConverter.Convert(in, toGV)
	}

	listObj, isList := in.(*unstructured.UnstructuredList)

	requestUID := uuid.NewUUID()
	desiredAPIVersion := toGV.String()
	objectsToConvert := getObjectsToConvert(in, desiredAPIVersion)
	request, response, err := createConversionReviewObjects(c.conversionReviewVersions, objectsToConvert, desiredAPIVersion, requestUID)
	if err != nil {
		return nil, err
	}

	objCount := len(objectsToConvert)
	if objCount == 0 {
		// no objects needed conversion
		if !isList {
			// for a single item, return as-is
			return in, nil
		}
		// for a list, set the version of the top-level list object (all individual objects are already in the correct version)
		out := listObj.DeepCopy()
		out.SetAPIVersion(toGV.String())
		return out, nil
	}

	trace := utiltrace.New("Call conversion webhook",
		utiltrace.Field{"custom-resource-definition", c.name},
		utiltrace.Field{"desired-api-version", desiredAPIVersion},
		utiltrace.Field{"object-count", objCount},
		utiltrace.Field{"UID", requestUID})
	// Only log conversion webhook traces that exceed a 8ms per object limit plus a 50ms request overhead allowance.
	// The per object limit uses the SLO for conversion webhooks (~4ms per object) plus time to serialize/deserialize
	// the conversion request on the apiserver side (~4ms per object).
	defer trace.LogIfLong(time.Duration(50+8*objCount) * time.Millisecond)

	// TODO: Figure out if adding one second timeout make sense here.
	ctx := context.TODO()
	r := c.restClient.Post().Context(ctx).Body(request).Do()
	if err := r.Into(response); err != nil {
		// TODO: Return a webhook specific error to be able to convert it to meta.Status
		return nil, fmt.Errorf("conversion webhook for %v failed: %v", in.GetObjectKind().GroupVersionKind(), err)
	}
	trace.Step("Request completed")

	convertedObjects, err := getConvertedObjectsFromResponse(requestUID, response)
	if err != nil {
		return nil, fmt.Errorf("conversion webhook for %v failed: %v", in.GetObjectKind().GroupVersionKind(), err)
	}

	if len(convertedObjects) != len(objectsToConvert) {
		return nil, fmt.Errorf("conversion webhook for %v returned %d objects, expected %d", in.GetObjectKind().GroupVersionKind(), len(convertedObjects), len(objectsToConvert))
	}

	if isList {
		// start a deepcopy of the input and fill in the converted objects from the response at the right spots.
		// The response list might be sparse because objects had the right version already.
		convertedList := listObj.DeepCopy()
		convertedIndex := 0
		for i := range convertedList.Items {
			original := &convertedList.Items[i]
			if original.GetAPIVersion() == toGV.String() {
				// This item has not been sent for conversion, and therefore does not show up in the response.
				// convertedList has the right item already.
				continue
			}
			converted, err := getRawExtensionObject(convertedObjects[convertedIndex])
			if err != nil {
				return nil, fmt.Errorf("conversion webhook for %v returned invalid converted object at index %v: %v", in.GetObjectKind().GroupVersionKind(), convertedIndex, err)
			}
			convertedIndex++
			if expected, got := toGV, converted.GetObjectKind().GroupVersionKind().GroupVersion(); expected != got {
				return nil, fmt.Errorf("conversion webhook for %v returned invalid converted object at index %v: invalid groupVersion (expected %v, received %v)", in.GetObjectKind().GroupVersionKind(), convertedIndex, expected, got)
			}
			if expected, got := original.GetObjectKind().GroupVersionKind().Kind, converted.GetObjectKind().GroupVersionKind().Kind; expected != got {
				return nil, fmt.Errorf("conversion webhook for %v returned invalid converted object at index %v: invalid kind (expected %v, received %v)", in.GetObjectKind().GroupVersionKind(), convertedIndex, expected, got)
			}
			unstructConverted, ok := converted.(*unstructured.Unstructured)
			if !ok {
				// this should not happened
				return nil, fmt.Errorf("conversion webhook for %v returned invalid converted object at index %v: invalid type, expected=Unstructured, got=%T", in.GetObjectKind().GroupVersionKind(), convertedIndex, converted)
			}
			if err := validateConvertedObject(original, unstructConverted); err != nil {
				return nil, fmt.Errorf("conversion webhook for %v returned invalid converted object at index %v: %v", in.GetObjectKind().GroupVersionKind(), convertedIndex, err)
			}
			if err := restoreObjectMeta(original, unstructConverted); err != nil {
				return nil, fmt.Errorf("conversion webhook for %v returned invalid metadata in object at index %v: %v", in.GetObjectKind().GroupVersionKind(), convertedIndex, err)
			}
			convertedList.Items[i] = *unstructConverted
		}
		convertedList.SetAPIVersion(toGV.String())
		return convertedList, nil
	}

	if len(convertedObjects) != 1 {
		// This should not happened
		return nil, fmt.Errorf("conversion webhook for %v failed, no objects returned", in.GetObjectKind())
	}
	converted, err := getRawExtensionObject(convertedObjects[0])
	if err != nil {
		return nil, err
	}
	if e, a := toGV, converted.GetObjectKind().GroupVersionKind().GroupVersion(); e != a {
		return nil, fmt.Errorf("conversion webhook for %v returned invalid object at index 0: invalid groupVersion (expected %v, received %v)", in.GetObjectKind().GroupVersionKind(), e, a)
	}
	if e, a := in.GetObjectKind().GroupVersionKind().Kind, converted.GetObjectKind().GroupVersionKind().Kind; e != a {
		return nil, fmt.Errorf("conversion webhook for %v returned invalid object at index 0: invalid kind (expected %v, received %v)", in.GetObjectKind().GroupVersionKind(), e, a)
	}
	unstructConverted, ok := converted.(*unstructured.Unstructured)
	if !ok {
		// this should not happened
		return nil, fmt.Errorf("conversion webhook for %v failed, unexpected type %T at index 0", in.GetObjectKind().GroupVersionKind(), converted)
	}
	unstructIn, ok := in.(*unstructured.Unstructured)
	if !ok {
		// this should not happened
		return nil, fmt.Errorf("conversion webhook for %v failed unexpected input type %T", in.GetObjectKind().GroupVersionKind(), in)
	}
	if err := validateConvertedObject(unstructIn, unstructConverted); err != nil {
		return nil, fmt.Errorf("conversion webhook for %v returned invalid object: %v", in.GetObjectKind().GroupVersionKind(), err)
	}
	if err := restoreObjectMeta(unstructIn, unstructConverted); err != nil {
		return nil, fmt.Errorf("conversion webhook for %v returned invalid metadata: %v", in.GetObjectKind().GroupVersionKind(), err)
	}
	return converted, nil
}

// validateConvertedObject checks that ObjectMeta fields match, with the exception of
// labels and annotations.
func validateConvertedObject(in, out *unstructured.Unstructured) error {
	if e, a := in.GetKind(), out.GetKind(); e != a {
		return fmt.Errorf("must have the same kind: %v != %v", e, a)
	}
	if e, a := in.GetName(), out.GetName(); e != a {
		return fmt.Errorf("must have the same name: %v != %v", e, a)
	}
	if e, a := in.GetNamespace(), out.GetNamespace(); e != a {
		return fmt.Errorf("must have the same namespace: %v != %v", e, a)
	}
	if e, a := in.GetUID(), out.GetUID(); e != a {
		return fmt.Errorf("must have the same UID: %v != %v", e, a)
	}
	return nil
}

// restoreObjectMeta deep-copies metadata from original into converted, while preserving labels and annotations from converted.
func restoreObjectMeta(original, converted *unstructured.Unstructured) error {
	obj, found := converted.Object["metadata"]
	if !found {
		return fmt.Errorf("missing metadata in converted object")
	}
	responseMetaData, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid metadata of type %T in converted object", obj)
	}

	if _, ok := original.Object["metadata"]; !ok {
		// the original will always have metadata. But just to be safe, let's clear in converted
		// with an empty object instead of nil, to be able to add labels and annotations below.
		converted.Object["metadata"] = map[string]interface{}{}
	} else {
		converted.Object["metadata"] = runtime.DeepCopyJSONValue(original.Object["metadata"])
	}

	obj = converted.Object["metadata"]
	convertedMetaData, ok := obj.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid metadata of type %T in input object", obj)
	}

	for _, fld := range []string{"labels", "annotations"} {
		obj, found := responseMetaData[fld]
		if !found || obj == nil {
			delete(convertedMetaData, fld)
			continue
		}
		responseField, ok := obj.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid metadata.%s of type %T in converted object", fld, obj)
		}

		originalField, ok := convertedMetaData[fld].(map[string]interface{})
		if !ok && convertedMetaData[fld] != nil {
			return fmt.Errorf("invalid metadata.%s of type %T in original object", fld, convertedMetaData[fld])
		}

		somethingChanged := len(originalField) != len(responseField)
		for k, v := range responseField {
			if _, ok := v.(string); !ok {
				return fmt.Errorf("metadata.%s[%s] must be a string, but is %T in converted object", fld, k, v)
			}
			if originalField[k] != interface{}(v) {
				somethingChanged = true
			}
		}

		if somethingChanged {
			stringMap := make(map[string]string, len(responseField))
			for k, v := range responseField {
				stringMap[k] = v.(string)
			}
			var errs field.ErrorList
			if fld == "labels" {
				errs = metav1validation.ValidateLabels(stringMap, field.NewPath("metadata", "labels"))
			} else {
				errs = apivalidation.ValidateAnnotations(stringMap, field.NewPath("metadata", "annotation"))
			}
			if len(errs) > 0 {
				return errs.ToAggregate()
			}
		}

		convertedMetaData[fld] = responseField
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
