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
	"gopkg.in/yaml.v2"
	"mime"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/NYTimes/gziphandler"
	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/googleapis/gnostic/compiler"

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

	orgSpec      *spec.Swagger
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

// BuildAndRegisterOpenAPIService builds the spec and registers a handler to provides access to it.
// Use this method if your OpenAPI spec is static. If you want to update the spec, use BuildOpenAPISpec then RegisterOpenAPIService.
func BuildAndRegisterOpenAPIService(servePath string, webServices []*restful.WebService, config *common.Config, handler common.PathHandler) (*OpenAPIService, error) {
	spec, err := builder.BuildOpenAPISpec(webServices, config)
	if err != nil {
		return nil, err
	}
	return RegisterOpenAPIService(spec, servePath, handler)
}

// RegisterOpenAPIService registers a handler to provides access to provided swagger spec.
// Note: servePath should end with ".json" as the RegisterOpenAPIService assume it is serving a
// json file and will also serve .pb and .gz files.
func RegisterOpenAPIService(openapiSpec *spec.Swagger, servePath string, handler common.PathHandler) (*OpenAPIService, error) {
	if !strings.HasSuffix(servePath, jsonExt) {
		return nil, fmt.Errorf("serving path must end with \"%s\"", jsonExt)
	}

	servePathBase := strings.TrimSuffix(servePath, jsonExt)

	o := OpenAPIService{}
	if err := o.UpdateSpec(openapiSpec); err != nil {
		return nil, err
	}

	type fileInfo struct {
		ext            string
		getDataAndETag func() ([]byte, string, time.Time)
	}

	files := []fileInfo{
		{".json", o.getSwaggerBytes},
		{"-2.0.0.json", o.getSwaggerBytes},
		{"-2.0.0.pb-v1", o.getSwaggerPbBytes},
		{"-2.0.0.pb-v1.gz", o.getSwaggerPbGzBytes},
	}

	for _, file := range files {
		path := servePathBase + file.ext
		getDataAndETag := file.getDataAndETag
		handler.Handle(path, gziphandler.GzipHandler(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				data, etag, lastModified := getDataAndETag()
				w.Header().Set("Etag", etag)

				// ServeContent will take care of caching using eTag.
				http.ServeContent(w, r, path, lastModified, bytes.NewReader(data))
			}),
		))
	}

	return &o, nil
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
	orgSpec := openapiSpec
	specBytes, err := json.MarshalIndent(openapiSpec, " ", " ")
	if err != nil {
		return err
	}
	specPb, err := toProtoBinary(specBytes)
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

	o.orgSpec = orgSpec
	o.specBytes = specBytes
	o.specPb = specPb
	o.specPbGz = specPbGz
	o.specBytesETag = specBytesETag
	o.specPbETag = specPbETag
	o.specPbGzETag = specPbGzETag
	o.lastModified = lastModified

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
