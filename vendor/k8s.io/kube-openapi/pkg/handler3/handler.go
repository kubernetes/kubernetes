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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	openapi_v3 "github.com/google/gnostic-models/openapiv3"
	"github.com/google/uuid"
	"github.com/munnerz/goautoneg"

	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/spec3"
)

const (
	subTypeProtobufDeprecated = "com.github.proto-openapi.spec.v3@v1.0+protobuf"
	subTypeProtobuf           = "com.github.proto-openapi.spec.v3.v1.0+protobuf"
	subTypeJSON               = "json"
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

func ToV3ProtoBinary(json []byte) ([]byte, error) {
	document, err := openapi_v3.ParseDocument(json)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(document)
}

type timedSpec struct {
	spec         []byte
	lastModified time.Time
}

// This type is protected by the lock on OpenAPIService.
type openAPIV3Group struct {
	specCache cached.LastSuccess[*spec3.OpenAPI]
	pbCache   cached.Value[timedSpec]
	jsonCache cached.Value[timedSpec]
}

func newOpenAPIV3Group() *openAPIV3Group {
	o := &openAPIV3Group{}
	o.jsonCache = cached.Transform[*spec3.OpenAPI](func(spec *spec3.OpenAPI, etag string, err error) (timedSpec, string, error) {
		if err != nil {
			return timedSpec{}, "", err
		}
		json, err := json.Marshal(spec)
		if err != nil {
			return timedSpec{}, "", err
		}
		return timedSpec{spec: json, lastModified: time.Now()}, computeETag(json), nil
	}, &o.specCache)
	o.pbCache = cached.Transform(func(ts timedSpec, etag string, err error) (timedSpec, string, error) {
		if err != nil {
			return timedSpec{}, "", err
		}
		proto, err := ToV3ProtoBinary(ts.spec)
		if err != nil {
			return timedSpec{}, "", err
		}
		return timedSpec{spec: proto, lastModified: ts.lastModified}, etag, nil
	}, o.jsonCache)
	return o
}

func (o *openAPIV3Group) UpdateSpec(openapi cached.Value[*spec3.OpenAPI]) {
	o.specCache.Store(openapi)
}

// OpenAPIService is the service responsible for serving OpenAPI spec. It has
// the ability to safely change the spec while serving it.
type OpenAPIService struct {
	// Mutex protects the schema map.
	mutex    sync.Mutex
	v3Schema map[string]*openAPIV3Group

	discoveryCache cached.LastSuccess[timedSpec]
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
func NewOpenAPIService() *OpenAPIService {
	o := &OpenAPIService{}
	o.v3Schema = make(map[string]*openAPIV3Group)
	// We're not locked because we haven't shared the structure yet.
	o.discoveryCache.Store(o.buildDiscoveryCacheLocked())
	return o
}

func (o *OpenAPIService) buildDiscoveryCacheLocked() cached.Value[timedSpec] {
	caches := make(map[string]cached.Value[timedSpec], len(o.v3Schema))
	for gvName, group := range o.v3Schema {
		caches[gvName] = group.jsonCache
	}
	return cached.Merge(func(results map[string]cached.Result[timedSpec]) (timedSpec, string, error) {
		discovery := &OpenAPIV3Discovery{Paths: make(map[string]OpenAPIV3DiscoveryGroupVersion)}
		for gvName, result := range results {
			if result.Err != nil {
				return timedSpec{}, "", result.Err
			}
			discovery.Paths[gvName] = OpenAPIV3DiscoveryGroupVersion{
				ServerRelativeURL: constructServerRelativeURL(gvName, result.Etag),
			}
		}
		j, err := json.Marshal(discovery)
		if err != nil {
			return timedSpec{}, "", err
		}
		return timedSpec{spec: j, lastModified: time.Now()}, computeETag(j), nil
	}, caches)
}

func (o *OpenAPIService) getSingleGroupBytes(getType string, group string) ([]byte, string, time.Time, error) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	v, ok := o.v3Schema[group]
	if !ok {
		return nil, "", time.Now(), fmt.Errorf("Cannot find CRD group %s", group)
	}
	switch getType {
	case subTypeJSON:
		ts, etag, err := v.jsonCache.Get()
		return ts.spec, etag, ts.lastModified, err
	case subTypeProtobuf, subTypeProtobufDeprecated:
		ts, etag, err := v.pbCache.Get()
		return ts.spec, etag, ts.lastModified, err
	default:
		return nil, "", time.Now(), fmt.Errorf("Invalid accept clause %s", getType)
	}
}

// UpdateGroupVersionLazy adds or updates an existing group with the new cached.
func (o *OpenAPIService) UpdateGroupVersionLazy(group string, openapi cached.Value[*spec3.OpenAPI]) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	if _, ok := o.v3Schema[group]; !ok {
		o.v3Schema[group] = newOpenAPIV3Group()
		// Since there is a new item, we need to re-build the cache map.
		o.discoveryCache.Store(o.buildDiscoveryCacheLocked())
	}
	o.v3Schema[group].UpdateSpec(openapi)
}

func (o *OpenAPIService) UpdateGroupVersion(group string, openapi *spec3.OpenAPI) {
	o.UpdateGroupVersionLazy(group, cached.Static(openapi, uuid.New().String()))
}

func (o *OpenAPIService) DeleteGroupVersion(group string) {
	o.mutex.Lock()
	defer o.mutex.Unlock()
	delete(o.v3Schema, group)
	// Rebuild the merge cache map since the items have changed.
	o.discoveryCache.Store(o.buildDiscoveryCacheLocked())
}

func (o *OpenAPIService) HandleDiscovery(w http.ResponseWriter, r *http.Request) {
	ts, etag, err := o.discoveryCache.Get()
	if err != nil {
		klog.Errorf("Error serving discovery: %s", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	w.Header().Set("Etag", strconv.Quote(etag))
	w.Header().Set("Content-Type", "application/json")
	http.ServeContent(w, r, "/openapi/v3", ts.lastModified, bytes.NewReader(ts.spec))
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
		Type                string
		SubType             string
		ReturnedContentType string
	}{
		{"application", subTypeJSON, "application/" + subTypeJSON},
		{"application", subTypeProtobuf, "application/" + subTypeProtobuf},
		{"application", subTypeProtobufDeprecated, "application/" + subTypeProtobuf},
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
			// Set Content-Type header in the reponse
			w.Header().Set("Content-Type", accepts.ReturnedContentType)

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
