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
	"fmt"
	"net/http"
	"net/url"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

// transformResponseObject takes an object loaded from storage and performs any necessary transformations,
// and returns the resulting object. Caller is expected to write the response as appropriate.
func transformResponseObject(ctx request.Context, e rest.Exporter, exportOptions metav1.ExportOptions,
	scope RequestScope, params url.Values, mediaType negotiation.MediaTypeOptions, result runtime.Object) (runtime.Object, error) {

	var err error
	if exportOptions.Export {
		if e == nil {
			err = errors.NewBadRequest(fmt.Sprintf("export of %q is not supported", scope.Resource.Resource))
			return result, err
		}

		// Apply export logic to every item of a list:
		if meta.IsListType(result) {
			items, err := meta.ExtractList(result)
			if err != nil {
				return result, err
			}
			for i := range items {
				items[i], err = e.Export(ctx, items[i], exportOptions)
				if err != nil {
					return result, err
				}
			}
		} else {
			result, err = e.Export(ctx, result, exportOptions)
			if err != nil {
				return result, err
			}
		}
	}

	// If conversion was allowed by the scope, perform it before writing the response
	if target := mediaType.Convert; target != nil {
		switch {

		case target.Kind == "PartialObjectMetadata" && target.GroupVersion() == metav1alpha1.SchemeGroupVersion:
			if meta.IsListType(result) {
				// TODO: this should be calculated earlier
				err = newNotAcceptableError(fmt.Sprintf("you requested PartialObjectMetadata, but the requested object is a list (%T)", result))
				return result, err
			}
			m, err := meta.Accessor(result)
			if err != nil {
				return result, err
			}
			partial := meta.AsPartialObjectMetadata(m)
			partial.GetObjectKind().SetGroupVersionKind(metav1alpha1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
			return partial, nil

		case target.Kind == "PartialObjectMetadataList" && target.GroupVersion() == metav1alpha1.SchemeGroupVersion:
			if !meta.IsListType(result) {
				// TODO: this should be calculated earlier
				err = newNotAcceptableError(fmt.Sprintf("you requested PartialObjectMetadataList, but the requested object is not a list (%T)", result))
				return result, err
			}
			list := &metav1alpha1.PartialObjectMetadataList{}
			err := meta.EachListItem(result, func(obj runtime.Object) error {
				m, err := meta.Accessor(obj)
				if err != nil {
					return err
				}
				partial := meta.AsPartialObjectMetadata(m)
				partial.GetObjectKind().SetGroupVersionKind(metav1alpha1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
				list.Items = append(list.Items, partial)
				return nil
			})
			if err != nil {
				return result, err
			}

			return list, nil

		case target.Kind == "Table" && target.GroupVersion() == metav1alpha1.SchemeGroupVersion:
			// TODO: relax the version abstraction
			// TODO: skip if this is a status response (delete without body)?

			opts := &metav1alpha1.TableOptions{}
			if err := metav1alpha1.ParameterCodec.DecodeParameters(params, metav1alpha1.SchemeGroupVersion, opts); err != nil {
				return result, err
			}

			table, err := scope.TableConvertor.ConvertToTable(ctx, result, opts)
			if err != nil {
				return result, err
			}

			for i := range table.Rows {
				item := &table.Rows[i]
				switch opts.IncludeObject {
				case metav1alpha1.IncludeObject:
					item.Object.Object, err = scope.Convertor.ConvertToVersion(item.Object.Object, scope.Kind.GroupVersion())
					if err != nil {
						return result, err
					}
				// TODO: rely on defaulting for the value here?
				case metav1alpha1.IncludeMetadata, "":
					m, err := meta.Accessor(item.Object.Object)
					if err != nil {
						return result, err
					}
					// TODO: turn this into an internal type and do conversion in order to get object kind automatically set?
					partial := meta.AsPartialObjectMetadata(m)
					partial.GetObjectKind().SetGroupVersionKind(metav1alpha1.SchemeGroupVersion.WithKind("PartialObjectMetadata"))
					item.Object.Object = partial
				case metav1alpha1.IncludeNone:
					item.Object.Object = nil
				default:
					// TODO: move this to validation on the table options?
					err = errors.NewBadRequest(fmt.Sprintf("unrecognized includeObject value: %q", opts.IncludeObject))
					return table, err
				}
			}

			return table, nil

		default:
			// this block should only be hit if scope AllowsConversion is incorrect
			accepted, _ := negotiation.MediaTypesForSerializer(metainternalversion.Codecs)
			err := negotiation.NewNotAcceptableError(accepted)
			return result, err
		}
	}

	return result, nil
}

// buildExportOptions handles the two paths by which export can be triggered
// today (export=true query parameter, and export=1 in Accept header), and
// returns an ExportOptions that can be used throughout response
// transformations. An error here should indicate a bad request.
func buildExportOptions(scope RequestScope, req *http.Request, mediaType negotiation.MediaTypeOptions,
	e rest.Exporter) (metav1.ExportOptions, error) {
	exportOptions := metav1.ExportOptions{}
	if err := metainternalversion.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, &exportOptions); err != nil {
		return exportOptions, err
	}
	if mediaType.Export {
		exportOptions.Export = true
	}

	if exportOptions.Export && e == nil {
		err := errors.NewBadRequest(fmt.Sprintf("export of %q is not supported", scope.Resource.Resource))
		return exportOptions, err
	}
	return exportOptions, nil
}

// transformAndWriteResponseObject applies object transformations, and writes the response.
func transformAndWriteResponseObject(ctx request.Context, e rest.Exporter, scope RequestScope, req *http.Request, w http.ResponseWriter, statusCode int, result runtime.Object) {

	mediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, &scope)
	if err != nil {
		status := responsewriters.ErrorToAPIStatus(err)
		responsewriters.WriteRawJSON(int(status.Code), status, w)
		return
	}

	exportOptions, err := buildExportOptions(scope, req, mediaType, e)
	if err != nil {
		status := responsewriters.ErrorToAPIStatus(err)
		responsewriters.WriteRawJSON(int(status.Code), status, w)
		return
	}

	transformedResult, err := transformResponseObject(ctx, e, exportOptions, scope, req.URL.Query(),
		mediaType, result)
	if err != nil {
		status := responsewriters.ErrorToAPIStatus(err)
		responsewriters.WriteRawJSON(int(status.Code), status, w)
		return
	}

	if target := mediaType.Convert; target != nil {
		// renegotiate under the internal version
		_, info, err := negotiation.NegotiateOutputMediaType(req, metainternalversion.Codecs, &scope)
		if err != nil {
			status := responsewriters.ErrorToAPIStatus(err)
			responsewriters.WriteRawJSON(int(status.Code), status, w)
			return
		}
		encoder := metainternalversion.Codecs.EncoderForVersion(info.Serializer, metav1alpha1.SchemeGroupVersion)
		responsewriters.SerializeObject(info.MediaType, encoder, w, req, statusCode, transformedResult)
		return
	}

	responsewriters.WriteObject(ctx, statusCode, scope.Kind.GroupVersion(), scope.Serializer, transformedResult, w, req)
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
