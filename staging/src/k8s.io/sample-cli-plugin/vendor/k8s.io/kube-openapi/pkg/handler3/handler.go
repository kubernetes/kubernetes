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
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	openapi_v3 "github.com/google/gnostic/openapiv3"
	"github.com/munnerz/goautoneg"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/internal/handler"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	subTypeProtobuf = "com.github.proto-openapi.spec.v3@v1.0+protobuf"
	subTypeJSON     = "json"
)

// OpenAPIV3Discovery is the format of the Discovery document for OpenAPI V3
// It maps Discovery paths to their corresponding URLs with a hash parameter included
type OpenAPIV3Discovery struct {
	Paths map[string]OpenAPIV3DiscoveryGroupVersion `json:"paths"`
}

// OpenAPIV3DiscoveryGroupVersion includes information about a group version and URL
// for accessing the OpenAPI. The URL includes a hash parameter to support client side caching
type OpenAPIV3DiscoveryGroupVersion struct {
	// Path is an absolute path of an OpenAPI V3 document in the form of /openapi/v3/apis/apps/v1?hash=014fbff9a07c
	ServerRelativeURL string `json:"serverRelativeURL"`
}

// OpenAPIService is the service responsible for serving OpenAPI spec. It has
// the ability to safely change the spec while serving it.
type OpenAPIService struct {
	// rwMutex protects All members of this service.
	rwMutex      sync.RWMutex
	lastModified time.Time
	v3Schema     map[string]*OpenAPIV3Group
}

type OpenAPIV3Group struct {
	rwMutex sync.RWMutex

	lastModified time.Time

	pbCache   handler.HandlerCache
	jsonCache handler.HandlerCache
	etagCache handler.HandlerCache
}

func computeETag(data []byte) string {
	if data == nil {
		return ""
	}
	return fmt.Sprintf("%X", sha512.Sum512(data))
}

func constructServerRelativeURL(gvString, etag string) string {
	u := url.URL{Path: path.Join("/openapi/v3", gvString)}
	query := url.Values{}
	query.Set("hash", etag)
	u.RawQuery = query.Encode()
	return u.String()
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
	discovery := &OpenAPIV3Discovery{Paths: make(map[string]OpenAPIV3DiscoveryGroupVersion)}
	for gvString, groupVersion := range o.v3Schema {
		etagBytes, err := groupVersion.etagCache.Get()
		if err != nil {
			return nil, err
		}
		discovery.Paths[gvString] = OpenAPIV3DiscoveryGroupVersion{
			ServerRelativeURL: constructServerRelativeURL(gvString, string(etagBytes)),
		}
	}
	j, err := json.Marshal(discovery)
	if err != nil {
		return nil, err
	}
	return j, nil
}

func (o *OpenAPIService) getSingleGroupBytes(getType string, group string) ([]byte, string, time.Time, error) {
	o.rwMutex.RLock()
	defer o.rwMutex.RUnlock()
	v, ok := o.v3Schema[group]
	if !ok {
		return nil, "", time.Now(), fmt.Errorf("Cannot find CRD group %s", group)
	}
	if getType == subTypeJSON {
		specBytes, err := v.jsonCache.Get()
		if err != nil {
			return nil, "", v.lastModified, err
		}
		etagBytes, err := v.etagCache.Get()
		return specBytes, string(etagBytes), v.lastModified, err
	} else if getType == subTypeProtobuf {
		specPb, err := v.pbCache.Get()
		if err != nil {
			return nil, "", v.lastModified, err
		}
		etagBytes, err := v.etagCache.Get()
		return specPb, string(etagBytes), v.lastModified, err
	}
	return nil, "", time.Now(), fmt.Errorf("Invalid accept clause %s", getType)
}

func (o *OpenAPIService) UpdateGroupVersion(group string, openapi *spec3.OpenAPI) (err error) {
	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()

	if _, ok := o.v3Schema[group]; !ok {
		o.v3Schema[group] = &OpenAPIV3Group{}
	}
	return o.v3Schema[group].UpdateSpec(openapi)
}

func (o *OpenAPIService) DeleteGroupVersion(group string) {
	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()
	delete(o.v3Schema, group)
}

func ToV3ProtoBinary(json []byte) ([]byte, error) {
	document, err := openapi_v3.ParseDocument(json)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(document)
}

func (o *OpenAPIService) HandleDiscovery(w http.ResponseWriter, r *http.Request) {
	data, _ := o.getGroupBytes()
	w.Header().Set("Etag", strconv.Quote(computeETag(data)))
	w.Header().Set("Content-Type", "application/json")
	http.ServeContent(w, r, "/openapi/v3", time.Now(), bytes.NewReader(data))
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
		{"application", subTypeJSON},
		{"application", subTypeProtobuf},
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
			// ETag must be enclosed in double quotes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
			w.Header().Set("Etag", strconv.Quote(etag))

			if hash := r.URL.Query().Get("hash"); hash != "" {
				if hash != etag {
					u := constructServerRelativeURL(group, etag)
					http.Redirect(w, r, u, 301)
					return
				}
				// The Vary header is required because the Accept header can
				// change the contents returned. This prevents clients from caching
				// protobuf as JSON and vice versa.
				w.Header().Set("Vary", "Accept")

				// Only set these headers when a hash is given.
				w.Header().Set("Cache-Control", "public, immutable")
				// Set the Expires directive to the maximum value of one year from the request,
				// effectively indicating that the cache never expires.
				w.Header().Set("Expires", time.Now().AddDate(1, 0, 0).Format(time.RFC1123))
			}
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

func (o *OpenAPIV3Group) UpdateSpec(openapi *spec3.OpenAPI) (err error) {
	o.rwMutex.Lock()
	defer o.rwMutex.Unlock()

	o.jsonCache = o.jsonCache.New(func() ([]byte, error) {
		return json.Marshal(openapi)
	})
	o.pbCache = o.pbCache.New(func() ([]byte, error) {
		json, err := o.jsonCache.Get()
		if err != nil {
			return nil, err
		}
		return ToV3ProtoBinary(json)
	})
	// TODO: This forces a json marshal of corresponding group-versions.
	// We should look to replace this with a faster hashing mechanism.
	o.etagCache = o.etagCache.New(func() ([]byte, error) {
		json, err := o.jsonCache.Get()
		if err != nil {
			return nil, err
		}
		return []byte(computeETag(json)), nil
	})
	o.lastModified = time.Now()
	return nil
}
