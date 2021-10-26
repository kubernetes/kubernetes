/*
Copyright 2021 The Kubernetes Authors.

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

package handler3

import (
	"bytes"
	"compress/gzip"
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	openapi_v3 "github.com/googleapis/gnostic/openapiv3"
	"github.com/munnerz/goautoneg"
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

	v3Schema map[string]*OpenAPIV3Group
}

type OpenAPIV3Group struct {
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
	o.v3Schema = make(map[string]*OpenAPIV3Group)
	return o, nil
}

func (o *OpenAPIService) getGroupBytes() ([]byte, error) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	keys := make([]string, len(o.v3Schema))
	i := 0
	for k := range o.v3Schema {
		keys[i] = k
		i++
	}

	sort.Strings(keys)

	group := make(map[string][]string)
	// TODO: Is there a standard for outputting an array of paths?
	group["Paths"] = keys

	j, _ := json.Marshal(group)

	return j, nil
}

func (o *OpenAPIService) getSingleGroupBytes(getType string, group string) ([]byte, string, time.Time, error) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	v, ok := o.v3Schema[group]
	if !ok {
		return nil, "", time.Now(), fmt.Errorf("Cannot find CRD group %s", group)
	}
	if getType == "json" {
		return v.specBytes, v.specBytesETag, v.lastModified, nil
	} else if getType == "com.github.proto-openapi.spec.v2@v1.0+protobuf" {
		return v.specPb, v.specPbETag, v.lastModified, nil
	}
	return nil, "", time.Now(), fmt.Errorf("Invalid accept clause %s", getType)
}

func (o *OpenAPIService) UpdateGroupVersion(group string, specBytes []byte) (err error) {
	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()
	if _, ok := o.v3Schema[group]; !ok {
		o.v3Schema[group] = &OpenAPIV3Group{}
	}

	specPb, err := ToV3ProtoBinary(specBytes)
	if err != nil {
		return err
	}

	specPbGz := toGzip(specPb)

	specBytesETag := computeETag(specBytes)
	specPbETag := computeETag(specPb)
	specPbGzETag := computeETag(specPbGz)

	lastModified := time.Now()

	o.v3Schema[group].rwMutex.Lock()
	defer o.v3Schema[group].rwMutex.Unlock()

	o.v3Schema[group].specBytes = specBytes
	o.v3Schema[group].specPb = specPb
	o.v3Schema[group].specPbGz = specPbGz

	o.v3Schema[group].specBytesETag = specBytesETag
	o.v3Schema[group].specPbETag = specPbETag
	o.v3Schema[group].specPbGzETag = specPbGzETag

	o.v3Schema[group].lastModified = lastModified

	return nil
}

func (o *OpenAPIService) DeleteGroupVersion(group string) {
	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()
	delete(o.v3Schema, group);
}

func ToV3ProtoBinary(json []byte) ([]byte, error) {
	document, err := openapi_v3.ParseDocument(json)
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

func (o *OpenAPIService) HandleDiscovery(w http.ResponseWriter, r *http.Request) {
	data, _ := o.getGroupBytes()
	http.ServeContent(w, r, "/openapi/v3", time.Now(), bytes.NewReader(data))
	return
}

func (o *OpenAPIService) HandleGroupVersion(w http.ResponseWriter, r *http.Request) {
	url := strings.SplitAfterN(r.URL.Path, "/", 4)
	group := url[3]

	decipherableFormats := r.Header.Get("Accept")
	if decipherableFormats == "" {
		decipherableFormats = "*/*"
	}
	clauses := goautoneg.ParseAccept(decipherableFormats)
	w.Header().Add("Vary", "Accept")

	if len(clauses) == 0 {
		return
	}

	accepted := []struct {
		Type    string
		SubType string
	}{
		{"application", "json"},
		{"application", "com.github.proto-openapi.spec.v2@v1.0+protobuf"},
	}

	for _, clause := range clauses {
		for _, accepts := range accepted {
			if clause.Type != accepts.Type && clause.Type != "*" {
				continue
			}
			if clause.SubType != accepts.SubType && clause.SubType != "*" {
				continue
			}
			data, etag, lastModified, err := o.getSingleGroupBytes(accepts.SubType, group)
			if err != nil {
				return
			}
			w.Header().Set("Etag", etag)
			http.ServeContent(w, r, "", lastModified, bytes.NewReader(data))
			return
		}
	}
	w.WriteHeader(406)
	return
}

func (o *OpenAPIService) RegisterOpenAPIV3VersionedService(servePath string, handler common.PathHandlerByGroupVersion) error {
	handler.Handle(servePath, http.HandlerFunc(o.HandleDiscovery))
	handler.HandlePrefix(servePath+"/", http.HandlerFunc(o.HandleGroupVersion))
	return nil
}
