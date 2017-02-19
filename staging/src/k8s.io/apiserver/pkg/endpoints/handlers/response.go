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
	"github.com/emicklei/go-restful"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// transformResponseObject takes an object loaded from storage and performs any necessary transformations.
// Will write the complete response object.
func transformResponseObject(ctx request.Context, scope RequestScope, req *restful.Request, res *restful.Response, statusCode int, result runtime.Object) {
	// TODO: use returned serializer
	mediaType, _, err := negotiation.NegotiateOutputMediaType(req.Request, scope.Serializer, &scope)
	if err != nil {
		status := responsewriters.ErrorToAPIStatus(err)
		responsewriters.WriteRawJSON(int(status.Code), status, res.ResponseWriter)
		return
	}

	// If conversion was allowed by the scope, perform it before writing the response
	if target := mediaType.Convert; target != nil {
		switch {
		case target.Kind == "TableList" && target.GroupVersion() == metav1alpha1.SchemeGroupVersion:
			// TODO: relax the version abstraction
			// TODO: skip if this is a status response (delete without body)?

			table, err := scope.TableConvertor.ConvertToTableList(ctx, result, nil)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}

			// renegotiate under the internal version
			_, info, err := negotiation.NegotiateOutputMediaType(req.Request, metainternalversion.Codecs, &scope)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
			encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, metav1alpha1.SchemeGroupVersion)
			responsewriters.SerializeObject(info.MediaType, encoder, res.ResponseWriter, req.Request, statusCode, table)
			return

		default:
			// this block should only be hit if scope AllowsConversion is incorrect
			accepted, _ := negotiation.MediaTypesForSerializer(metainternalversion.Codecs)
			err := negotiation.NewNotAcceptableError(accepted)
			status := responsewriters.ErrorToAPIStatus(err)
			responsewriters.WriteRawJSON(int(status.Code), status, res.ResponseWriter)
			return
		}
	}

	responsewriters.WriteObject(statusCode, scope.Kind.GroupVersion(), scope.Serializer, result, res.ResponseWriter, req.Request)
}
