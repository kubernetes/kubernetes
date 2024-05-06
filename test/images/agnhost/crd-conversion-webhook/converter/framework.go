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

package converter

import (
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/munnerz/goautoneg"

	"k8s.io/klog/v2"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// convertFunc is the user defined function for any conversion. The code in this file is a
// template that can be use for any CR conversion given this function.
type convertFunc func(Object *unstructured.Unstructured, version string) (*unstructured.Unstructured, metav1.Status)

func statusErrorWithMessage(msg string, params ...interface{}) metav1.Status {
	return metav1.Status{
		Message: fmt.Sprintf(msg, params...),
		Status:  metav1.StatusFailure,
	}
}

func statusSucceed() metav1.Status {
	return metav1.Status{
		Status: metav1.StatusSuccess,
	}
}

// doConversionV1beta1 converts the requested objects in the v1beta1 ConversionRequest using the given conversion function and
// returns a conversion response. Failures are reported with the Reason in the conversion response.
func doConversionV1beta1(convertRequest *v1beta1.ConversionRequest, convert convertFunc) *v1beta1.ConversionResponse {
	var convertedObjects []runtime.RawExtension
	for _, obj := range convertRequest.Objects {
		cr := unstructured.Unstructured{}
		if err := cr.UnmarshalJSON(obj.Raw); err != nil {
			klog.Error(err)
			return &v1beta1.ConversionResponse{
				Result: metav1.Status{
					Message: fmt.Sprintf("failed to unmarshall object (%v) with error: %v", string(obj.Raw), err),
					Status:  metav1.StatusFailure,
				},
			}
		}
		convertedCR, status := convert(&cr, convertRequest.DesiredAPIVersion)
		if status.Status != metav1.StatusSuccess {
			klog.Error(status.String())
			return &v1beta1.ConversionResponse{
				Result: status,
			}
		}
		convertedCR.SetAPIVersion(convertRequest.DesiredAPIVersion)
		convertedObjects = append(convertedObjects, runtime.RawExtension{Object: convertedCR})
	}
	return &v1beta1.ConversionResponse{
		ConvertedObjects: convertedObjects,
		Result:           statusSucceed(),
	}
}

// doConversionV1 converts the requested objects in the v1 ConversionRequest using the given conversion function and
// returns a conversion response. Failures are reported with the Reason in the conversion response.
func doConversionV1(convertRequest *v1.ConversionRequest, convert convertFunc) *v1.ConversionResponse {
	var convertedObjects []runtime.RawExtension
	for _, obj := range convertRequest.Objects {
		cr := unstructured.Unstructured{}
		if err := cr.UnmarshalJSON(obj.Raw); err != nil {
			klog.Error(err)
			return &v1.ConversionResponse{
				Result: metav1.Status{
					Message: fmt.Sprintf("failed to unmarshall object (%v) with error: %v", string(obj.Raw), err),
					Status:  metav1.StatusFailure,
				},
			}
		}
		convertedCR, status := convert(&cr, convertRequest.DesiredAPIVersion)
		if status.Status != metav1.StatusSuccess {
			klog.Error(status.String())
			return &v1.ConversionResponse{
				Result: status,
			}
		}
		convertedCR.SetAPIVersion(convertRequest.DesiredAPIVersion)
		convertedObjects = append(convertedObjects, runtime.RawExtension{Object: convertedCR})
	}
	return &v1.ConversionResponse{
		ConvertedObjects: convertedObjects,
		Result:           statusSucceed(),
	}
}

func serve(w http.ResponseWriter, r *http.Request, convert convertFunc) {
	var body []byte
	if r.Body != nil {
		if data, err := io.ReadAll(r.Body); err == nil {
			body = data
		}
	}

	contentType := r.Header.Get("Content-Type")
	serializer := getInputSerializer(contentType)
	if serializer == nil {
		msg := fmt.Sprintf("invalid Content-Type header `%s`", contentType)
		klog.Errorf(msg)
		http.Error(w, msg, http.StatusBadRequest)
		return
	}

	klog.V(2).Infof("handling request: %v", body)
	obj, gvk, err := serializer.Decode(body, nil, nil)
	if err != nil {
		msg := fmt.Sprintf("failed to deserialize body (%v) with error %v", string(body), err)
		klog.Error(err)
		http.Error(w, msg, http.StatusBadRequest)
		return
	}

	var responseObj runtime.Object
	switch *gvk {
	case v1beta1.SchemeGroupVersion.WithKind("ConversionReview"):
		convertReview, ok := obj.(*v1beta1.ConversionReview)
		if !ok {
			msg := fmt.Sprintf("Expected v1beta1.ConversionReview but got: %T", obj)
			klog.Errorf(msg)
			http.Error(w, msg, http.StatusBadRequest)
			return
		}
		convertReview.Response = doConversionV1beta1(convertReview.Request, convert)
		convertReview.Response.UID = convertReview.Request.UID
		klog.V(2).Info(fmt.Sprintf("sending response: %v", convertReview.Response))

		// reset the request, it is not needed in a response.
		convertReview.Request = &v1beta1.ConversionRequest{}
		responseObj = convertReview
	case v1.SchemeGroupVersion.WithKind("ConversionReview"):
		convertReview, ok := obj.(*v1.ConversionReview)
		if !ok {
			msg := fmt.Sprintf("Expected v1.ConversionReview but got: %T", obj)
			klog.Errorf(msg)
			http.Error(w, msg, http.StatusBadRequest)
			return
		}
		convertReview.Response = doConversionV1(convertReview.Request, convert)
		convertReview.Response.UID = convertReview.Request.UID
		klog.V(2).Info(fmt.Sprintf("sending response: %v", convertReview.Response))

		// reset the request, it is not needed in a response.
		convertReview.Request = &v1.ConversionRequest{}
		responseObj = convertReview
	default:
		msg := fmt.Sprintf("Unsupported group version kind: %v", gvk)
		klog.Error(err)
		http.Error(w, msg, http.StatusBadRequest)
		return
	}

	accept := r.Header.Get("Accept")
	outSerializer := getOutputSerializer(accept)
	if outSerializer == nil {
		msg := fmt.Sprintf("invalid accept header `%s`", accept)
		klog.Errorf(msg)
		http.Error(w, msg, http.StatusBadRequest)
		return
	}
	err = outSerializer.Encode(responseObj, w)
	if err != nil {
		klog.Error(err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// ServeExampleConvert servers endpoint for the example converter defined as convertExampleCRD function.
func ServeExampleConvert(w http.ResponseWriter, r *http.Request) {
	serve(w, r, convertExampleCRD)
}

type mediaType struct {
	Type, SubType string
}

var scheme = runtime.NewScheme()

func init() {
	addToScheme(scheme)
}

func addToScheme(scheme *runtime.Scheme) {
	utilruntime.Must(v1.AddToScheme(scheme))
	utilruntime.Must(v1beta1.AddToScheme(scheme))
}

var serializers = map[mediaType]runtime.Serializer{
	{"application", "json"}: json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{Pretty: false}),
	{"application", "yaml"}: json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{Yaml: true}),
}

func getInputSerializer(contentType string) runtime.Serializer {
	parts := strings.SplitN(contentType, "/", 2)
	if len(parts) != 2 {
		return nil
	}
	return serializers[mediaType{parts[0], parts[1]}]
}

func getOutputSerializer(accept string) runtime.Serializer {
	if len(accept) == 0 {
		return serializers[mediaType{"application", "json"}]
	}

	clauses := goautoneg.ParseAccept(accept)
	for _, clause := range clauses {
		for k, v := range serializers {
			switch {
			case clause.Type == k.Type && clause.SubType == k.SubType,
				clause.Type == k.Type && clause.SubType == "*",
				clause.Type == "*" && clause.SubType == "*":
				return v
			}
		}
	}

	return nil
}
