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
	"compress/gzip"
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"sync"
	"time"

	"github.com/NYTimes/gziphandler"
	"github.com/emicklei/go-restful"
	"github.com/golang/protobuf/proto"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	"github.com/munnerz/goautoneg"
	"gopkg.in/yaml.v2"
	klog "k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	jsonExt = ".json"

	mimeJson = "application/json"
	// TODO(mehdy): change @68f4ded to a version tag when gnostic add version tags.
	mimePb   = "application/com.github.googleapis.gnostic.OpenAPIv2@68f4ded+protobuf"
	mimePbGz = "application/x-gzip"
)

// OpenAPIService is the service responsible for serving OpenAPI spec. It has
// the ability to safely change the spec while serving it.
type OpenAPIService struct {
	// rwMutex protects All members of this service.
	rwMutex sync.RWMutex

	lastModified time.Time

	jsonCache  cache
	protoCache cache
}

type cache struct {
	BuildCache func() ([]byte, error)
	once       sync.Once
	bytes      []byte
	etag       string
	err        error
}

func (c *cache) Get() ([]byte, string, error) {
	c.once.Do(func() {
		bytes, err := c.BuildCache()
		// if there is an error updating the cache, there can be situations where
		// c.bytes contains a valid value (carried over from the previous update)
		// but c.err is also not nil; the cache user is expected to check for this
		c.err = err
		if c.err == nil {
			// don't override previous spec if we had an error
			c.bytes = bytes
			c.etag = computeETag(c.bytes)
		}
	})
	return c.bytes, c.etag, c.err
}

func (c *cache) New(cacheBuilder func() ([]byte, error)) cache {
	return cache{
		bytes:      c.bytes,
		etag:       c.etag,
		BuildCache: cacheBuilder,
	}
}

func init() {
	mime.AddExtensionType(".json", mimeJson)
	mime.AddExtensionType(".pb-v1", mimePb)
	mime.AddExtensionType(".gz", mimePbGz)
}

func computeETag(data []byte) string {
	if data == nil {
		return ""
	}
	return fmt.Sprintf("\"%X\"", sha512.Sum512(data))
}

// NewOpenAPIService builds an OpenAPIService starting with the given spec.
func NewOpenAPIService(spec *spec.Swagger) (*OpenAPIService, error) {
	o := &OpenAPIService{}
	if err := o.UpdateSpec(spec); err != nil {
		return nil, err
	}
	return o, nil
}

func (o *OpenAPIService) getSwaggerBytes() ([]byte, string, time.Time, error) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	specBytes, etag, err := o.jsonCache.Get()
	if err != nil {
		return nil, "", time.Time{}, err
	}
	return specBytes, etag, o.lastModified, nil
}

func (o *OpenAPIService) getSwaggerPbBytes() ([]byte, string, time.Time, error) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	specPb, etag, err := o.protoCache.Get()
	if err != nil {
		return nil, "", time.Time{}, err
	}
	return specPb, etag, o.lastModified, nil
}

func (o *OpenAPIService) UpdateSpec(openapiSpec *spec.Swagger) (err error) {
	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()
	o.jsonCache = o.jsonCache.New(func() ([]byte, error) {
		return json.Marshal(openapiSpec)
	})
	o.protoCache = o.protoCache.New(func() ([]byte, error) {
		json, _, err := o.jsonCache.Get()
		if err != nil {
			return nil, err
		}
		return ToProtoBinary(json)
	})
	o.lastModified = time.Now()

	return nil
}

func jsonToYAML(j map[string]interface{}) yaml.MapSlice {
	if j == nil {
		return nil
	}
	ret := make(yaml.MapSlice, 0, len(j))
	for k, v := range j {
		ret = append(ret, yaml.MapItem{k, jsonToYAMLValue(v)})
	}
	return ret
}

func jsonToYAMLValue(j interface{}) interface{} {
	switch j := j.(type) {
	case map[string]interface{}:
		return jsonToYAML(j)
	case []interface{}:
		ret := make([]interface{}, len(j))
		for i := range j {
			ret[i] = jsonToYAMLValue(j[i])
		}
		return ret
	case float64:
		// replicate the logic in https://github.com/go-yaml/yaml/blob/51d6538a90f86fe93ac480b35f37b2be17fef232/resolve.go#L151
		if i64 := int64(j); j == float64(i64) {
			if i := int(i64); i64 == int64(i) {
				return i
			}
			return i64
		}
		if ui64 := uint64(j); j == float64(ui64) {
			return ui64
		}
		return j
	case int64:
		if i := int(j); j == int64(i) {
			return i
		}
		return j
	}
	return j
}

func ToProtoBinary(json []byte) ([]byte, error) {
	document, err := openapi_v2.ParseDocument(json)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(document)
}

func toGzip(data []byte) []byte {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	zw.Write(data)
	zw.Close()
	return buf.Bytes()
}

// RegisterOpenAPIVersionedService registers a handler to provide access to provided swagger spec.
//
// Deprecated: use OpenAPIService.RegisterOpenAPIVersionedService instead.
func RegisterOpenAPIVersionedService(spec *spec.Swagger, servePath string, handler common.PathHandler) (*OpenAPIService, error) {
	o, err := NewOpenAPIService(spec)
	if err != nil {
		return nil, err
	}
	return o, o.RegisterOpenAPIVersionedService(servePath, handler)
}

// RegisterOpenAPIVersionedService registers a handler to provide access to provided swagger spec.
func (o *OpenAPIService) RegisterOpenAPIVersionedService(servePath string, handler common.PathHandler) error {
	accepted := []struct {
		Type           string
		SubType        string
		GetDataAndETag func() ([]byte, string, time.Time, error)
	}{
		{"application", "json", o.getSwaggerBytes},
		{"application", "com.github.proto-openapi.spec.v2@v1.0+protobuf", o.getSwaggerPbBytes},
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
					data, etag, lastModified, err := accepts.GetDataAndETag()
					if err != nil {
						klog.Errorf("Error in OpenAPI handler: %s", err)
						// only return a 503 if we have no older cache data to serve
						if data == nil {
							w.WriteHeader(http.StatusServiceUnavailable)
							return
						}
					}
					w.Header().Set("Etag", etag)
					// ServeContent will take care of caching using eTag.
					http.ServeContent(w, r, servePath, lastModified, bytes.NewReader(data))
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
func BuildAndRegisterOpenAPIVersionedService(servePath string, webServices []*restful.WebService, config *common.Config, handler common.PathHandler) (*OpenAPIService, error) {
	spec, err := builder.BuildOpenAPISpec(webServices, config)
	if err != nil {
		return nil, err
	}
	o, err := NewOpenAPIService(spec)
	if err != nil {
		return nil, err
	}
	return o, o.RegisterOpenAPIVersionedService(servePath, handler)
}
