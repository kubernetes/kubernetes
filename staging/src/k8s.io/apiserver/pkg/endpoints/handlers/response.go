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
	"net/http"

	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// transformResponseObject takes an object loaded from storage and performs any necessary transformations.
// Will write the complete response object.
func transformResponseObject(ctx request.Context, scope RequestScope, req *http.Request, w http.ResponseWriter, statusCode int, result runtime.Object) {
	// TODO: use returned serializer
	mediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, &scope)
	if err != nil {
		status := responsewriters.ErrorToAPIStatus(err)
		responsewriters.WriteRawJSON(int(status.Code), status, w)
		return
	}

	// If conversion was allowed by the scope, perform it before writing the response
	if target := mediaType.Convert; target != nil {
		switch {
		case target.Kind == "TableList" && target.GroupVersion() == metav1alpha1.SchemeGroupVersion:
			// TODO: relax the version abstraction
			// TODO: skip if this is a status response (delete without body)?

			opts := &metav1alpha1.TableListOptions{}
			if err := metav1alpha1.ParameterCodec.DecodeParameters(req.URL.Query(), metav1alpha1.SchemeGroupVersion, opts); err != nil {
				scope.err(err, w, req)
				return
			}

			table, err := scope.TableConvertor.ConvertToTableList(ctx, result, opts)
			if err != nil {
				scope.err(err, w, req)
				return
			}

			for i := range table.Items {
				item := &table.Items[i]
				if opts.IncludeObject {
					item.Object.Object, err = scope.Convertor.ConvertToVersion(item.Object.Object, scope.Kind.GroupVersion())
					if err != nil {
						scope.err(err, w, req)
						return
					}
				} else {
					m, err := meta.Accessor(item.Object.Object)
					if err != nil {
						scope.err(err, w, req)
						return
					}
					// TODO: turn this into an internal type and do conversion in order to get object kind automatically set?
					partial := meta.AsPartialObjectMetadata(m)
					partial.GetObjectKind().SetGroupVersionKind(metav1alpha1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
					item.Object.Object = partial
				}
			}

			// renegotiate under the internal version
			_, info, err := negotiation.NegotiateOutputMediaType(req, metainternalversion.Codecs, &scope)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, metav1alpha1.SchemeGroupVersion)
			responsewriters.SerializeObject(info.MediaType, encoder, w, req, statusCode, table)
			return

		default:
			// this block should only be hit if scope AllowsConversion is incorrect
			accepted, _ := negotiation.MediaTypesForSerializer(metainternalversion.Codecs)
			err := negotiation.NewNotAcceptableError(accepted)
			status := responsewriters.ErrorToAPIStatus(err)
			responsewriters.WriteRawJSON(int(status.Code), status, w)
			return
		}
	}

	responsewriters.WriteObject(statusCode, scope.Kind.GroupVersion(), scope.Serializer, result, w, req)
}
