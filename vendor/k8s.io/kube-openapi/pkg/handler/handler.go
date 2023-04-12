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

package handler

import (
	"bytes"
	"crypto/sha512"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/NYTimes/gziphandler"
	"github.com/emicklei/go-restful/v3"
	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/google/gnostic/openapiv2"
	"github.com/google/uuid"
	"github.com/munnerz/goautoneg"
	klog "k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	subTypeProtobufDeprecated = "com.github.proto-openapi.spec.v2@v1.0+protobuf"
	subTypeProtobuf           = "com.github.proto-openapi.spec.v2.v1.0+protobuf"
	subTypeJSON               = "json"
)

func computeETag(data []byte) string {
	if data == nil {
		return ""
	}
	return fmt.Sprintf("%X", sha512.Sum512(data))
}

type timedSpec struct {
	spec         []byte
	lastModified time.Time
}

// OpenAPIService is the service responsible for serving OpenAPI spec. It has
// the ability to safely change the spec while serving it.
type OpenAPIService struct {
	specCache  cached.Replaceable[*spec.Swagger]
	jsonCache  cached.Data[timedSpec]
	protoCache cached.Data[timedSpec]
}

// NewOpenAPIService builds an OpenAPIService starting with the given spec.
func NewOpenAPIService(swagger *spec.Swagger) *OpenAPIService {
	return NewOpenAPIServiceLazy(cached.NewResultOK(swagger, uuid.New().String()))
}

// NewOpenAPIServiceLazy builds an OpenAPIService from lazy spec.
func NewOpenAPIServiceLazy(swagger cached.Data[*spec.Swagger]) *OpenAPIService {
	o := &OpenAPIService{}
	o.UpdateSpecLazy(swagger)

	o.jsonCache = cached.NewTransformer[*spec.Swagger](func(result cached.Result[*spec.Swagger]) cached.Result[timedSpec] {
		if result.Err != nil {
			return cached.NewResultErr[timedSpec](result.Err)
		}
		json, err := result.Data.MarshalJSON()
		if err != nil {
			return cached.NewResultErr[timedSpec](err)
		}
		return cached.NewResultOK(timedSpec{spec: json, lastModified: time.Now()}, computeETag(json))
	}, &o.specCache)
	o.protoCache = cached.NewTransformer(func(result cached.Result[timedSpec]) cached.Result[timedSpec] {
		if result.Err != nil {
			return cached.NewResultErr[timedSpec](result.Err)
		}
		proto, err := ToProtoBinary(result.Data.spec)
		if err != nil {
			return cached.NewResultErr[timedSpec](err)
		}
		// We can re-use the same etag as json because of the Vary header.
		return cached.NewResultOK(timedSpec{spec: proto, lastModified: result.Data.lastModified}, result.Etag)
	}, o.jsonCache)
	return o
}

func (o *OpenAPIService) getSwaggerBytes() (timedSpec, string, error) {
	result := o.jsonCache.Get()
	return result.Data, result.Etag, result.Err
}

func (o *OpenAPIService) getSwaggerPbBytes() (timedSpec, string, error) {
	result := o.protoCache.Get()
	return result.Data, result.Etag, result.Err
}

func (o *OpenAPIService) UpdateSpec(swagger *spec.Swagger) error {
	o.UpdateSpecLazy(cached.NewResultOK(swagger, uuid.New().String()))
	return nil
}

func (o *OpenAPIService) UpdateSpecLazy(swagger cached.Data[*spec.Swagger]) {
	o.specCache.Replace(swagger)
}

func ToProtoBinary(json []byte) ([]byte, error) {
	document, err := openapi_v2.ParseDocument(json)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(document)
}

// RegisterOpenAPIVersionedService registers a handler to provide access to provided swagger spec.
//
// Deprecated: use OpenAPIService.RegisterOpenAPIVersionedService instead.
func RegisterOpenAPIVersionedService(spec *spec.Swagger, servePath string, handler common.PathHandler) (*OpenAPIService, error) {
	o := NewOpenAPIService(spec)
	return o, o.RegisterOpenAPIVersionedService(servePath, handler)
}

// RegisterOpenAPIVersionedService registers a handler to provide access to provided swagger spec.
func (o *OpenAPIService) RegisterOpenAPIVersionedService(servePath string, handler common.PathHandler) error {
	accepted := []struct {
		Type                string
		SubType             string
		ReturnedContentType string
		GetDataAndEtag      cached.Data[timedSpec]
	}{
		{"application", subTypeJSON, "application/" + subTypeJSON, o.jsonCache},
		{"application", subTypeProtobufDeprecated, "application/" + subTypeProtobuf, o.protoCache},
		{"application", subTypeProtobuf, "application/" + subTypeProtobuf, o.protoCache},
	}

	handler.Handle(servePath, gziphandler.GzipHandler(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			decipherableFormats := r.Header.Get("Accept")
			if decipherableFormats == "" {
				decipherableFormats = "*/*"
			}
			clauses := goautoneg.ParseAccept(decipherableFormats)
			w.Header().Add("Vary", "Accept")
			for _, clause := range clauses {
				for _, accepts := range accepted {
					if clause.Type != accepts.Type && clause.Type != "*" {
						continue
					}
					if clause.SubType != accepts.SubType && clause.SubType != "*" {
						continue
					}
					// serve the first matching media type in the sorted clause list
					result := accepts.GetDataAndEtag.Get()
					if result.Err != nil {
						klog.Errorf("Error in OpenAPI handler: %s", result.Err)
						// only return a 503 if we have no older cache data to serve
						if result.Data.spec == nil {
							w.WriteHeader(http.StatusServiceUnavailable)
							return
						}
					}
					// Set Content-Type header in the reponse
					w.Header().Set("Content-Type", accepts.ReturnedContentType)

					// ETag must be enclosed in double quotes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
					w.Header().Set("Etag", strconv.Quote(result.Etag))
					// ServeContent will take care of caching using eTag.
					http.ServeContent(w, r, servePath, result.Data.lastModified, bytes.NewReader(result.Data.spec))
					return
				}
			}
			// Return 406 for not acceptable format
			w.WriteHeader(406)
			return
		}),
	))

	return nil
}

// BuildAndRegisterOpenAPIVersionedService builds the spec and registers a handler to provide access to it.
// Use this method if your OpenAPI spec is static. If you want to update the spec, use BuildOpenAPISpec then RegisterOpenAPIVersionedService.
//
// Deprecated: BuildAndRegisterOpenAPIVersionedServiceFromRoutes should be used instead.
func BuildAndRegisterOpenAPIVersionedService(servePath string, webServices []*restful.WebService, config *common.Config, handler common.PathHandler) (*OpenAPIService, error) {
	return BuildAndRegisterOpenAPIVersionedServiceFromRoutes(servePath, restfuladapter.AdaptWebServices(webServices), config, handler)
}

// BuildAndRegisterOpenAPIVersionedServiceFromRoutes builds the spec and registers a handler to provide access to it.
// Use this method if your OpenAPI spec is static. If you want to update the spec, use BuildOpenAPISpec then RegisterOpenAPIVersionedService.
func BuildAndRegisterOpenAPIVersionedServiceFromRoutes(servePath string, routeContainers []common.RouteContainer, config *common.Config, handler common.PathHandler) (*OpenAPIService, error) {
	spec, err := builder.BuildOpenAPISpecFromRoutes(routeContainers, config)
	if err != nil {
		return nil, err
	}
	o := NewOpenAPIService(spec)
	return o, o.RegisterOpenAPIVersionedService(servePath, handler)
}
