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
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// transformResponseObject takes an object loaded from storage and performs any necessary transformations.
// Will write the complete response object.
func transformResponseObject(ctx context.Context, scope RequestScope, req *http.Request, w http.ResponseWriter, statusCode int, result runtime.Object) {
	// TODO: fetch the media type much earlier in request processing and pass it into this method.
	trace := scope.Trace
	mediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, &scope)
	if err != nil {
		status := responsewriters.ErrorToAPIStatus(err)
		trace.Step("Writing raw JSON response")
		responsewriters.WriteRawJSON(int(status.Code), status, w)
		return
	}

	// If conversion was allowed by the scope, perform it before writing the response
	if target := mediaType.Convert; target != nil {
		switch {

		case target.Kind == "PartialObjectMetadata" && target.GroupVersion() == metav1beta1.SchemeGroupVersion:
			if meta.IsListType(result) {
				// TODO: this should be calculated earlier
				err = newNotAcceptableError(fmt.Sprintf("you requested PartialObjectMetadata, but the requested object is a list (%T)", result))
				scope.err(err, w, req)
				return
			}
			m, err := meta.Accessor(result)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			partial := meta.AsPartialObjectMetadata(m)
			partial.GetObjectKind().SetGroupVersionKind(metav1beta1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))

			// renegotiate under the internal version
			_, info, err := negotiation.NegotiateOutputMediaType(req, metainternalversion.Codecs, &scope)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, metav1beta1.SchemeGroupVersion)
			trace.Step(fmt.Sprintf("Serializing response as type %s", info.MediaType))
			responsewriters.SerializeObject(info.MediaType, encoder, w, req, statusCode, partial)
			return

		case target.Kind == "PartialObjectMetadataList" && target.GroupVersion() == metav1beta1.SchemeGroupVersion:
			if !meta.IsListType(result) {
				// TODO: this should be calculated earlier
				err = newNotAcceptableError(fmt.Sprintf("you requested PartialObjectMetadataList, but the requested object is not a list (%T)", result))
				scope.err(err, w, req)
				return
			}
			list := &metav1beta1.PartialObjectMetadataList{}
			trace.Step("Processing list items")
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
				scope.err(err, w, req)
				return
			}

			// renegotiate under the internal version
			_, info, err := negotiation.NegotiateOutputMediaType(req, metainternalversion.Codecs, &scope)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, metav1beta1.SchemeGroupVersion)
			trace.Step(fmt.Sprintf("Serializing response as type %s", info.MediaType))
			responsewriters.SerializeObject(info.MediaType, encoder, w, req, statusCode, list)
			return

		case target.Kind == "Table" && target.GroupVersion() == metav1beta1.SchemeGroupVersion:
			// TODO: relax the version abstraction
			// TODO: skip if this is a status response (delete without body)?

			opts := &metav1beta1.TableOptions{}
			trace.Step("Decoding parameters")
			if err := metav1beta1.ParameterCodec.DecodeParameters(req.URL.Query(), metav1beta1.SchemeGroupVersion, opts); err != nil {
				scope.err(err, w, req)
				return
			}

			trace.Step("Converting to table")
			table, err := scope.TableConvertor.ConvertToTable(ctx, result, opts)
			if err != nil {
				scope.err(err, w, req)
				return
			}

			trace.Step("Processing rows")
			for i := range table.Rows {
				item := &table.Rows[i]
				switch opts.IncludeObject {
				case metav1beta1.IncludeObject:
					item.Object.Object, err = scope.Convertor.ConvertToVersion(item.Object.Object, scope.Kind.GroupVersion())
					if err != nil {
						scope.err(err, w, req)
						return
					}
				// TODO: rely on defaulting for the value here?
				case metav1beta1.IncludeMetadata, "":
					m, err := meta.Accessor(item.Object.Object)
					if err != nil {
						scope.err(err, w, req)
						return
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
					scope.err(err, w, req)
				}
			}

			// renegotiate under the internal version
			_, info, err := negotiation.NegotiateOutputMediaType(req, metainternalversion.Codecs, &scope)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, metav1beta1.SchemeGroupVersion)
			trace.Step(fmt.Sprintf("Serializing response as type %s", info.MediaType))
			responsewriters.SerializeObject(info.MediaType, encoder, w, req, statusCode, table)
			return

		default:
			// this block should only be hit if scope AllowsConversion is incorrect
			accepted, _ := negotiation.MediaTypesForSerializer(metainternalversion.Codecs)
			err := negotiation.NewNotAcceptableError(accepted)
			status := responsewriters.ErrorToAPIStatus(err)
			trace.Step("Writing raw JSON response")
			responsewriters.WriteRawJSON(int(status.Code), status, w)
			return
		}
	}

	trace.Step("Writing response")
	responsewriters.WriteObject(statusCode, scope.Kind.GroupVersion(), scope.Serializer, result, w, req)
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
