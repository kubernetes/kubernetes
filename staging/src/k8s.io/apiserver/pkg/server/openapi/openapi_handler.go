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

package openapi

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/go-openapi/spec"
	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"
	"gopkg.in/yaml.v2"
	genericmux "k8s.io/apiserver/pkg/server/mux"
)

type OpenAPIService struct {
	orgSpec      *spec.Swagger
	specBytes    []byte
	specPb       []byte
	specPbGz     []byte
	lastModified time.Time
	updateHooks  []func(*http.Request)
}

// RegisterOpenAPIService registers a handler to provides standard OpenAPI specification.
func RegisterOpenAPIService(openapiSpec *spec.Swagger, servePath string, mux *genericmux.PathRecorderMux) (*OpenAPIService, error) {
	if !strings.HasSuffix(servePath, JSON_EXT) {
		return nil, fmt.Errorf("Serving path must ends with \"%s\".", JSON_EXT)
	}

	servePathBase := servePath[:len(servePath)-len(JSON_EXT)]

	o := OpenAPIService{}
	if err := o.UpdateSpec(openapiSpec); err != nil {
		return nil, err
	}

	mime.AddExtensionType(".json", MIME_JSON)
	mime.AddExtensionType(".pb-v1", MIME_PB)
	mime.AddExtensionType(".gz", MIME_PB_GZ)

	type fileInfo struct {
		ext     string
		getData func() []byte
	}

	files := []fileInfo{
		{".json", o.getSwaggerBytes},
		{"-2.0.0.json", o.getSwaggerBytes},
		{"-2.0.0.pb-v1", o.getSwaggerPbBytes},
		{"-2.0.0.pb-v1.gz", o.getSwaggerPbGzBytes},
	}

	for _, file := range files {
		path := servePathBase + file.ext
		getData := file.getData
		mux.HandleFunc(path, func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != path {
				w.WriteHeader(http.StatusNotFound)
				w.Write([]byte("Path not found!"))
				return
			}
			o.update(r)
			data := getData()
			etag := computeEtag(data)
			w.Header().Set("Etag", etag)
			// ServeContent will take care of caching using eTag.
			http.ServeContent(w, r, path, o.lastModified, bytes.NewReader(data))
		})
	}

	return &o, nil
}

func (o *OpenAPIService) getSwaggerBytes() []byte {
	return o.specBytes
}

func (o *OpenAPIService) getSwaggerPbBytes() []byte {
	return o.specPb
}

func (o *OpenAPIService) getSwaggerPbGzBytes() []byte {
	return o.specPbGz
}

func (o *OpenAPIService) GetSpec() *spec.Swagger {
	return o.orgSpec
}

func (o *OpenAPIService) UpdateSpec(openapiSpec *spec.Swagger) (err error) {
	o.orgSpec = openapiSpec
	o.specBytes, err = json.MarshalIndent(openapiSpec, " ", " ")
	if err != nil {
		return err
	}
	o.specPb, err = toProtoBinary(o.specBytes)
	if err != nil {
		return err
	}
	o.specPbGz = toGzip(o.specPb)
	o.lastModified = time.Now()

	return nil
}

func toProtoBinary(spec []byte) ([]byte, error) {
	var info yaml.MapSlice
	err := yaml.Unmarshal(spec, &info)
	if err != nil {
		return nil, err
	}
	document, err := openapi_v2.NewDocument(info, compiler.NewContext("$root", nil))
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

// Adds an update hook to be called on each spec request. The hook is responsible
// to call UpdateSpec method.
func (o *OpenAPIService) AddUpdateHook(hook func(*http.Request)) {
	o.updateHooks = append(o.updateHooks, hook)
}

func (o *OpenAPIService) update(r *http.Request) {
	for _, h := range o.updateHooks {
		h(r)
	}
}
