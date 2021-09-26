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
	"fmt"
	"mime"
	"net/http"
	"sync"
	"time"

	"github.com/NYTimes/gziphandler"
	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/compiler"
	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	jsoniter "github.com/json-iterator/go"
	"github.com/munnerz/goautoneg"
	"gopkg.in/yaml.v2"

	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
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

	specBytes []byte
	specPb    []byte
	specPbGz  []byte

	specBytesETag string
	specPbETag    string
	specPbGzETag  string
}

func init() {
	mime.AddExtensionType(".json", mimeJson)
	mime.AddExtensionType(".pb-v1", mimePb)
	mime.AddExtensionType(".gz", mimePbGz)
}

func computeETag(data []byte) string {
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

func (o *OpenAPIService) getSwaggerBytes() ([]byte, string, time.Time) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	return o.specBytes, o.specBytesETag, o.lastModified
}

func (o *OpenAPIService) getSwaggerPbBytes() ([]byte, string, time.Time) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	return o.specPb, o.specPbETag, o.lastModified
}

func (o *OpenAPIService) getSwaggerPbGzBytes() ([]byte, string, time.Time) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	return o.specPbGz, o.specPbGzETag, o.lastModified
}

func (o *OpenAPIService) UpdateSpec(openapiSpec *spec.Swagger) (err error) {
	specBytes, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(openapiSpec)
	if err != nil {
		return err
	}
	var json map[string]interface{}
	if err := jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(specBytes, &json); err != nil {
		return err
	}
	specPb, err := ToProtoBinary(json)
	if err != nil {
		return err
	}
	specPbGz := toGzip(specPb)

	specBytesETag := computeETag(specBytes)
	specPbETag := computeETag(specPb)
	specPbGzETag := computeETag(specPbGz)

	lastModified := time.Now()

	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()

	o.specBytes = specBytes
	o.specPb = specPb
	o.specPbGz = specPbGz
	o.specBytesETag = specBytesETag
	o.specPbETag = specPbETag
	o.specPbGzETag = specPbGzETag
	o.lastModified = lastModified

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

func ToProtoBinary(json map[string]interface{}) ([]byte, error) {
	document, err := openapi_v2.NewDocument(jsonToYAML(json), compiler.NewContext("$root", nil))
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
		GetDataAndETag func() ([]byte, string, time.Time)
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
					data, etag, lastModified := accepts.GetDataAndETag()
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
