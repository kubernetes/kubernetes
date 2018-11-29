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

package handlers

import (
	"context"
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// transformResponseObject takes an object loaded from storage and performs any necessary transformations.
// Will write the complete response object.
func transformResponseObject(ctx context.Context, scope RequestScope, req *http.Request, w http.ResponseWriter, statusCode int, mediaType negotiation.MediaTypeOptions, result runtime.Object) {
	// status objects are ignored for transformation
	if _, ok := result.(*metav1.Status); ok {
		responsewriters.WriteObject(statusCode, scope.Kind.GroupVersion(), scope.Serializer, result, w, req)
		return
	}

	// ensure the self link and empty list array are set
	if err := setObjectSelfLink(ctx, result, req, scope.Namer); err != nil {
		scope.err(err, w, req)
		return
	}

	trace := scope.Trace

	// If conversion was allowed by the scope, perform it before writing the response
	switch target := mediaType.Convert; {

	case target == nil:
		trace.Step("Writing response")
		responsewriters.WriteObject(statusCode, scope.Kind.GroupVersion(), scope.Serializer, result, w, req)

	case target.Kind == "PartialObjectMetadata" && target.GroupVersion() == metav1beta1.SchemeGroupVersion:
		partial, err := asV1Beta1PartialObjectMetadata(result)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		if err := writeMetaInternalVersion(partial, statusCode, w, req, &scope, target.GroupVersion()); err != nil {
			scope.err(err, w, req)
			return
		}

	case target.Kind == "PartialObjectMetadataList" && target.GroupVersion() == metav1beta1.SchemeGroupVersion:
		trace.Step("Processing list items")
		partial, err := asV1Beta1PartialObjectMetadataList(result)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		if err := writeMetaInternalVersion(partial, statusCode, w, req, &scope, target.GroupVersion()); err != nil {
			scope.err(err, w, req)
			return
		}

	case target.Kind == "Table" && target.GroupVersion() == metav1beta1.SchemeGroupVersion:
		opts := &metav1beta1.TableOptions{}
		trace.Step("Decoding parameters")
		if err := metav1beta1.ParameterCodec.DecodeParameters(req.URL.Query(), metav1beta1.SchemeGroupVersion, opts); err != nil {
			scope.err(err, w, req)
			return
		}

		table, err := asV1Beta1Table(ctx, result, opts, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		if err := writeMetaInternalVersion(table, statusCode, w, req, &scope, target.GroupVersion()); err != nil {
			scope.err(err, w, req)
			return
		}

	default:
		// this block should only be hit if scope AllowsConversion is incorrect
		accepted, _ := negotiation.MediaTypesForSerializer(metainternalversion.Codecs)
		err := negotiation.NewNotAcceptableError(accepted)
		scope.err(err, w, req)
	}
}

// errNotAcceptable indicates Accept negotiation has failed
type errNotAcceptable struct {
	message string
}

func newNotAcceptableError(message string) error {
	return errNotAcceptable{message}
}

func (e errNotAcceptable) Error() string {
	return e.message
}

func (e errNotAcceptable) Status() metav1.Status {
	return metav1.Status{
		Status:  metav1.StatusFailure,
		Code:    http.StatusNotAcceptable,
		Reason:  metav1.StatusReason("NotAcceptable"),
		Message: e.Error(),
	}
}

func asV1Beta1Table(ctx context.Context, result runtime.Object, opts *metav1beta1.TableOptions, scope RequestScope) (runtime.Object, error) {
	trace := scope.Trace

	trace.Step("Converting to table")
	table, err := scope.TableConvertor.ConvertToTable(ctx, result, opts)
	if err != nil {
		return nil, err
	}

	trace.Step("Processing rows")
	for i := range table.Rows {
		item := &table.Rows[i]
		switch opts.IncludeObject {
		case metav1beta1.IncludeObject:
			item.Object.Object, err = scope.Convertor.ConvertToVersion(item.Object.Object, scope.Kind.GroupVersion())
			if err != nil {
				return nil, err
			}
		// TODO: rely on defaulting for the value here?
		case metav1beta1.IncludeMetadata, "":
			m, err := meta.Accessor(item.Object.Object)
			if err != nil {
				return nil, err
			}
			// TODO: turn this into an internal type and do conversion in order to get object kind automatically set?
			partial := meta.AsPartialObjectMetadata(m)
			partial.GetObjectKind().SetGroupVersionKind(metav1beta1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
			item.Object.Object = partial
		case metav1beta1.IncludeNone:
			item.Object.Object = nil
		default:
			// TODO: move this to validation on the table options?
			err = errors.NewBadRequest(fmt.Sprintf("unrecognized includeObject value: %q", opts.IncludeObject))
			return nil, err
		}
	}

	return table, nil
}

func asV1Beta1PartialObjectMetadata(result runtime.Object) (runtime.Object, error) {
	if meta.IsListType(result) {
		// TODO: this should be calculated earlier
		err := newNotAcceptableError(fmt.Sprintf("you requested PartialObjectMetadata, but the requested object is a list (%T)", result))
		return nil, err
	}
	m, err := meta.Accessor(result)
	if err != nil {
		return nil, err
	}
	partial := meta.AsPartialObjectMetadata(m)
	partial.GetObjectKind().SetGroupVersionKind(metav1beta1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
	return partial, nil
}

func asV1Beta1PartialObjectMetadataList(result runtime.Object) (runtime.Object, error) {
	if !meta.IsListType(result) {
		// TODO: this should be calculated earlier
		return nil, newNotAcceptableError(fmt.Sprintf("you requested PartialObjectMetadataList, but the requested object is not a list (%T)", result))
	}
	list := &metav1beta1.PartialObjectMetadataList{}
	err := meta.EachListItem(result, func(obj runtime.Object) error {
		m, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		partial := meta.AsPartialObjectMetadata(m)
		partial.GetObjectKind().SetGroupVersionKind(metav1beta1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
		list.Items = append(list.Items, partial)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return list, nil
}

func writeMetaInternalVersion(obj runtime.Object, statusCode int, w http.ResponseWriter, req *http.Request, restrictions negotiation.EndpointRestrictions, target schema.GroupVersion) error {
	// renegotiate under the internal version
	_, info, err := negotiation.NegotiateOutputMediaType(req, metainternalversion.Codecs, restrictions)
	if err != nil {
		return err
	}
	encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, target)
	responsewriters.SerializeObject(info.MediaType, encoder, w, req, statusCode, obj)
	return nil
}
