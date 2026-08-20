/*
Copyright The Kubernetes Authors.

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

package routes

import (
	"bytes"
	"crypto/sha512"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/NYTimes/gziphandler"
	"github.com/munnerz/goautoneg"

	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// This file implements the bytes-mode OpenAPI v2 service used when the
// OpenAPIV2BytesCache feature gate is enabled. It serves the same content,
// content types and ETags as kube-openapi's handler.OpenAPIService (which the
// HTTP serving logic below mirrors), but retains only marshaled spec bytes:
// parsed *spec.Swagger values passed to the update methods are marshaled and
// released rather than held for the lifetime of the server.

const (
	openAPIV2SubTypeProtobufDeprecated = "com.github.proto-openapi.spec.v2@v1.0+protobuf"
	openAPIV2SubTypeProtobuf           = "com.github.proto-openapi.spec.v2.v1.0+protobuf"
	openAPIV2SubTypeJSON               = "json"
)

func computeOpenAPIV2ETag(data []byte) string {
	if data == nil {
		return ""
	}
	return fmt.Sprintf("%X", sha512.Sum512(data))
}

type timedSpecBytes struct {
	spec         []byte
	lastModified time.Time
}

// OpenAPIV2BytesService serves the /openapi/v2 endpoint from marshaled spec
// bytes. It has the ability to safely change the spec while serving it.
type OpenAPIV2BytesService struct {
	bytesSource *cached.Atomic[[]byte]
	jsonCache   cached.Value[timedSpecBytes]
	protoCache  cached.Value[timedSpecBytes]
}

// NewOpenAPIV2BytesService builds an OpenAPIV2BytesService starting with the
// given marshaled JSON spec.
func NewOpenAPIV2BytesService(jsonSpec []byte) *OpenAPIV2BytesService {
	o := newOpenAPIV2BytesService()
	o.UpdateSpecFromBytes(jsonSpec)
	return o
}

// NewOpenAPIV2BytesServiceLazy builds an OpenAPIV2BytesService from a lazy
// source of marshaled JSON spec bytes. The source is only re-evaluated when
// its etag changes.
func NewOpenAPIV2BytesServiceLazy(jsonSpec cached.Value[[]byte]) *OpenAPIV2BytesService {
	o := newOpenAPIV2BytesService()
	o.bytesSource.Store(jsonSpec)
	return o
}

func newOpenAPIV2BytesService() *OpenAPIV2BytesService {
	o := &OpenAPIV2BytesService{bytesSource: &cached.Atomic[[]byte]{}}
	o.jsonCache = cached.Transform[[]byte](func(json []byte, etag string, err error) (timedSpecBytes, string, error) {
		if err != nil {
			return timedSpecBytes{}, "", err
		}
		// Serve a content-hash ETag rather than passing through the source's
		// etag, so that the served ETag is identical to what
		// handler.OpenAPIService serves for the same spec and only changes
		// when the content actually changes.
		return timedSpecBytes{spec: json, lastModified: time.Now()}, computeOpenAPIV2ETag(json), nil
	}, o.bytesSource)
	o.protoCache = cached.Transform(func(ts timedSpecBytes, etag string, err error) (timedSpecBytes, string, error) {
		if err != nil {
			return timedSpecBytes{}, "", err
		}
		proto, err := handler.ToProtoBinary(ts.spec)
		if err != nil {
			return timedSpecBytes{}, "", err
		}
		// We can re-use the same etag as json because of the Vary header.
		return timedSpecBytes{spec: proto, lastModified: ts.lastModified}, etag, nil
	}, o.jsonCache)
	return o
}

// UpdateSpec replaces the served spec with the given spec. The spec is
// marshaled immediately and not retained.
func (o *OpenAPIV2BytesService) UpdateSpec(swagger *spec.Swagger) error {
	json, err := swagger.MarshalJSON()
	if err != nil {
		return err
	}
	o.UpdateSpecFromBytes(json)
	return nil
}

// UpdateSpecLazy replaces the source of the served spec with a lazy source of
// parsed specs. The source is only re-evaluated when its etag changes, and
// each evaluated spec is marshaled and released rather than retained by the
// service. This method makes the service usable where a
// *handler.OpenAPIService is updated via its UpdateSpecLazy method.
func (o *OpenAPIV2BytesService) UpdateSpecLazy(swagger cached.Value[*spec.Swagger]) {
	o.bytesSource.Store(cached.Transform[*spec.Swagger](func(s *spec.Swagger, etag string, err error) ([]byte, string, error) {
		if err != nil {
			return nil, "", err
		}
		json, err := s.MarshalJSON()
		if err != nil {
			return nil, "", err
		}
		return json, computeOpenAPIV2ETag(json), nil
	}, swagger))
}

// UpdateSpecFromBytes replaces the served spec with the given marshaled JSON
// spec.
func (o *OpenAPIV2BytesService) UpdateSpecFromBytes(jsonSpec []byte) {
	o.bytesSource.Store(cached.Static(jsonSpec, computeOpenAPIV2ETag(jsonSpec)))
}

// UpdateSpecLazyBytes replaces the source of the served spec with a lazy
// source of marshaled JSON spec bytes.
func (o *OpenAPIV2BytesService) UpdateSpecLazyBytes(jsonSpec cached.Value[[]byte]) {
	o.bytesSource.Store(jsonSpec)
}

// RegisterOpenAPIVersionedService registers a handler to provide access to the
// swagger spec. The serving behavior (content negotiation, ETag handling,
// compression) mirrors handler.OpenAPIService.RegisterOpenAPIVersionedService.
func (o *OpenAPIV2BytesService) RegisterOpenAPIVersionedService(servePath string, handler common.PathHandler) {
	accepted := []struct {
		Type                string
		SubType             string
		ReturnedContentType string
		GetDataAndEtag      cached.Value[timedSpecBytes]
	}{
		{"application", openAPIV2SubTypeJSON, "application/" + openAPIV2SubTypeJSON, o.jsonCache},
		{"application", openAPIV2SubTypeProtobufDeprecated, "application/" + openAPIV2SubTypeProtobuf, o.protoCache},
		{"application", openAPIV2SubTypeProtobuf, "application/" + openAPIV2SubTypeProtobuf, o.protoCache},
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
					ts, etag, err := accepts.GetDataAndEtag.Get()
					if err != nil {
						klog.Errorf("Error in OpenAPI handler: %s", err)
						// only return a 503 if we have no older cache data to serve
						if ts.spec == nil {
							w.WriteHeader(http.StatusServiceUnavailable)
							return
						}
					}
					w.Header().Set("Content-Type", accepts.ReturnedContentType)

					// ETag must be enclosed in double quotes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
					w.Header().Set("Etag", strconv.Quote(etag))
					// ServeContent will take care of caching using eTag.
					http.ServeContent(w, r, servePath, ts.lastModified, bytes.NewReader(ts.spec))
					return
				}
			}
			// Return 406 for not acceptable format
			w.WriteHeader(http.StatusNotAcceptable)
		}),
	))
}
