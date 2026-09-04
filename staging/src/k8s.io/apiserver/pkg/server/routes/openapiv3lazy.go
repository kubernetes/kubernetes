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
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful/v3"
	"github.com/google/uuid"
	"github.com/munnerz/goautoneg"
	"k8s.io/klog/v2"

	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/kube-openapi/pkg/builder3"
	"k8s.io/kube-openapi/pkg/cached"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
)

// OpenAPIV3Updater is the subset of the OpenAPI v3 serving API used by
// components that publish additional group-versions (the CRD OpenAPI v3
// controller). It is implemented by both kube-openapi's
// handler3.OpenAPIService and OpenAPIV3LazyService.
type OpenAPIV3Updater interface {
	// UpdateGroupVersion adds or replaces the spec served for group (a
	// discovery path such as "apis/apps/v1").
	UpdateGroupVersion(group string, openapi *spec3.OpenAPI)
	// UpdateGroupVersionLazy adds or replaces the spec source for group. The
	// source is consulted on every request; its etag decides whether the
	// serialized form is reused.
	UpdateGroupVersionLazy(group string, openapi cached.Value[*spec3.OpenAPI])
	// DeleteGroupVersion removes group from discovery and serving.
	DeleteGroupVersion(group string)
}

var _ OpenAPIV3Updater = &handler3.OpenAPIService{}
var _ OpenAPIV3Updater = &OpenAPIV3LazyService{}

const (
	v3SubTypeProtobufDeprecated = "com.github.proto-openapi.spec.v3@v1.0+protobuf"
	v3SubTypeProtobuf           = "com.github.proto-openapi.spec.v3.v1.0+protobuf"
	v3SubTypeJSON               = "json"

	// v3ETagFormatRevision is folded into every built-in group-version etag.
	// Bump it whenever the serialized form produced by this package changes
	// for reasons not captured by the other etag inputs (for example a
	// change to filterScopedGVKs), so that clients holding immutable cached
	// copies keyed by the old hash re-fetch.
	v3ETagFormatRevision = "1"
)

// v3Entry is one served group-version. It is either a built-in entry (build
// is set, etag is fixed up front, the spec is built on first request) or a
// dynamic entry (source is set, the etag is the content hash, exactly like
// handler3). In both cases only serialized bytes are retained after a build;
// the parsed spec graph is released.
type v3Entry struct {
	mu sync.Mutex

	// Built-in entries. build is released after the first successful build.
	builtin bool
	build   func() (*spec3.OpenAPI, error)

	// Dynamic entries.
	source cached.Value[*spec3.OpenAPI]
	// sourceETag is the etag returned by source for the currently cached
	// bytes; the bytes are rebuilt when the source reports a different etag.
	sourceETag string

	etag         string
	jsonBytes    []byte
	pbBytes      []byte
	lastModified time.Time
}

// getJSON returns the JSON serialization of the entry, building it if
// necessary. For built-in entries a failed build is not cached and is retried
// on the next call. For dynamic entries a source error is hidden behind the
// last successfully serialized bytes, matching handler3's LastSuccess
// semantics.
func (e *v3Entry) getJSON() ([]byte, string, time.Time, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.builtin {
		if e.jsonBytes != nil {
			return e.jsonBytes, e.etag, e.lastModified, nil
		}
		spec, err := e.build()
		if err != nil {
			return nil, "", time.Time{}, err
		}
		j, err := json.Marshal(spec)
		if err != nil {
			return nil, "", time.Time{}, err
		}
		e.jsonBytes = j
		e.lastModified = time.Now()
		// Release the builder so that anything it captured, and the spec
		// graph, become collectable. Only the bytes are retained.
		e.build = nil
		return e.jsonBytes, e.etag, e.lastModified, nil
	}

	spec, sourceETag, err := e.source.Get()
	if err != nil {
		if e.jsonBytes != nil {
			return e.jsonBytes, e.etag, e.lastModified, nil
		}
		return nil, "", time.Time{}, err
	}
	if e.jsonBytes != nil && e.sourceETag == sourceETag {
		return e.jsonBytes, e.etag, e.lastModified, nil
	}
	j, err := json.Marshal(spec)
	if err != nil {
		if e.jsonBytes != nil {
			return e.jsonBytes, e.etag, e.lastModified, nil
		}
		return nil, "", time.Time{}, err
	}
	e.jsonBytes = j
	e.pbBytes = nil
	e.sourceETag = sourceETag
	e.etag = computeV3ETag(j)
	e.lastModified = time.Now()
	return e.jsonBytes, e.etag, e.lastModified, nil
}

// getProtobuf returns the protobuf serialization, derived from the JSON one.
func (e *v3Entry) getProtobuf() ([]byte, string, time.Time, error) {
	j, etag, lastModified, err := e.getJSON()
	if err != nil {
		return nil, "", time.Time{}, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	// getJSON may have replaced the bytes concurrently; only reuse the
	// protobuf if it corresponds to the JSON we were handed.
	if e.pbBytes != nil && bytes.Equal(j, e.jsonBytes) {
		return e.pbBytes, etag, lastModified, nil
	}
	pb, err := handler3.ToV3ProtoBinary(j)
	if err != nil {
		return nil, "", time.Time{}, err
	}
	if bytes.Equal(j, e.jsonBytes) {
		e.pbBytes = pb
	}
	return pb, etag, lastModified, nil
}

// OpenAPIV3LazyService serves /openapi/v3 with the same HTTP semantics as
// kube-openapi's handler3.OpenAPIService (discovery document, per
// group-version documents in JSON or protobuf, ETags, the hash query
// parameter with 301 redirects and immutable caching), but
//
//   - built-in group-versions are registered with a builder and a
//     precomputed etag, so nothing is built until a group-version is
//     requested — in particular the discovery document never triggers a
//     build;
//   - after a build only the serialized bytes are retained; the parsed spec
//     graph is dropped.
//
// Dynamic group-versions (CRDs, published through OpenAPIV3Updater) keep
// handler3's content-hash etags and rebuild-on-change behaviour.
type OpenAPIV3LazyService struct {
	mu      sync.RWMutex
	entries map[string]*v3Entry

	// discovery cache, keyed by the etags of all entries.
	discoveryKey          string
	discoveryBytes        []byte
	discoveryETag         string
	discoveryLastModified time.Time
}

// NewOpenAPIV3LazyService returns an empty service.
func NewOpenAPIV3LazyService() *OpenAPIV3LazyService {
	return &OpenAPIV3LazyService{entries: map[string]*v3Entry{}}
}

// AddBuiltinGroupVersion registers a built-in group-version whose spec is
// produced by build on first request. etag is served as the ETag and embedded
// in the discovery URL; it must change whenever build's output would.
func (o *OpenAPIV3LazyService) AddBuiltinGroupVersion(group, etag string, build func() (*spec3.OpenAPI, error)) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.entries[group] = &v3Entry{builtin: true, build: build, etag: etag}
}

// UpdateGroupVersionLazy implements OpenAPIV3Updater.
func (o *OpenAPIV3LazyService) UpdateGroupVersionLazy(group string, openapi cached.Value[*spec3.OpenAPI]) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.entries[group] = &v3Entry{source: openapi}
}

// UpdateGroupVersion implements OpenAPIV3Updater.
func (o *OpenAPIV3LazyService) UpdateGroupVersion(group string, openapi *spec3.OpenAPI) {
	o.UpdateGroupVersionLazy(group, cached.Static(openapi, uuid.New().String()))
}

// DeleteGroupVersion implements OpenAPIV3Updater.
func (o *OpenAPIV3LazyService) DeleteGroupVersion(group string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	delete(o.entries, group)
}

// GroupVersions returns the registered discovery paths, sorted.
func (o *OpenAPIV3LazyService) GroupVersions() []string {
	o.mu.RLock()
	defer o.mu.RUnlock()
	gvs := make([]string, 0, len(o.entries))
	for gv := range o.entries {
		gvs = append(gvs, gv)
	}
	sort.Strings(gvs)
	return gvs
}

func (o *OpenAPIV3LazyService) getDiscovery() ([]byte, string, time.Time, error) {
	o.mu.RLock()
	entries := maps.Clone(o.entries)
	o.mu.RUnlock()

	gvs := make([]string, 0, len(entries))
	for gv := range entries {
		gvs = append(gvs, gv)
	}
	sort.Strings(gvs)

	etags := make(map[string]string, len(entries))
	var key strings.Builder
	for _, gv := range gvs {
		e := entries[gv]
		var etag string
		if e.source != nil {
			// Dynamic entries: the content hash requires the source; getJSON
			// is cheap when the source etag is unchanged.
			_, sourceETag, _, err := e.getJSON()
			if err != nil {
				return nil, "", time.Time{}, err
			}
			etag = sourceETag
		} else {
			// Built-in entries: fixed etag, no build needed.
			etag = e.etag
		}
		etags[gv] = etag
		key.WriteString(gv)
		key.WriteByte(':')
		key.WriteString(etag)
		key.WriteByte('\n')
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	if o.discoveryBytes != nil && o.discoveryKey == key.String() {
		return o.discoveryBytes, o.discoveryETag, o.discoveryLastModified, nil
	}
	discovery := &handler3.OpenAPIV3Discovery{Paths: make(map[string]handler3.OpenAPIV3DiscoveryGroupVersion, len(etags))}
	for gv, etag := range etags {
		discovery.Paths[gv] = handler3.OpenAPIV3DiscoveryGroupVersion{
			ServerRelativeURL: constructV3ServerRelativeURL(gv, etag),
		}
	}
	j, err := json.Marshal(discovery)
	if err != nil {
		return nil, "", time.Time{}, err
	}
	o.discoveryKey = key.String()
	o.discoveryBytes = j
	o.discoveryETag = computeV3ETag(j)
	o.discoveryLastModified = time.Now()
	return o.discoveryBytes, o.discoveryETag, o.discoveryLastModified, nil
}

// HandleDiscovery serves the /openapi/v3 discovery document.
func (o *OpenAPIV3LazyService) HandleDiscovery(w http.ResponseWriter, r *http.Request) {
	data, etag, lastModified, err := o.getDiscovery()
	if err != nil {
		klog.Errorf("Error serving discovery: %s", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	w.Header().Set("Etag", strconv.Quote(etag))
	w.Header().Set("Content-Type", "application/json")
	http.ServeContent(w, r, "/openapi/v3", lastModified, bytes.NewReader(data))
}

func (o *OpenAPIV3LazyService) getSingleGroupBytes(getType string, group string) ([]byte, string, time.Time, error) {
	o.mu.RLock()
	e, ok := o.entries[group]
	o.mu.RUnlock()
	if !ok {
		return nil, "", time.Now(), fmt.Errorf("cannot find group %s", group)
	}
	switch getType {
	case v3SubTypeJSON:
		return e.getJSON()
	case v3SubTypeProtobuf, v3SubTypeProtobufDeprecated:
		return e.getProtobuf()
	default:
		return nil, "", time.Now(), fmt.Errorf("invalid accept clause %s", getType)
	}
}

// HandleGroupVersion serves one /openapi/v3/<group-version> document. The
// response semantics mirror handler3.OpenAPIService.HandleGroupVersion.
func (o *OpenAPIV3LazyService) HandleGroupVersion(w http.ResponseWriter, r *http.Request) {
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
		{"application", v3SubTypeJSON, "application/" + v3SubTypeJSON},
		{"application", v3SubTypeProtobuf, "application/" + v3SubTypeProtobuf},
		{"application", v3SubTypeProtobufDeprecated, "application/" + v3SubTypeProtobuf},
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
			w.Header().Set("Content-Type", accepts.ReturnedContentType)

			// ETag must be enclosed in double quotes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
			w.Header().Set("Etag", strconv.Quote(etag))

			if hash := r.URL.Query().Get("hash"); hash != "" {
				if hash != etag {
					u := constructV3ServerRelativeURL(group, etag)
					http.Redirect(w, r, u, http.StatusMovedPermanently)
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
	w.WriteHeader(http.StatusNotAcceptable)
}

// RegisterOpenAPIV3VersionedService registers the discovery and
// group-version handlers under servePath.
func (o *OpenAPIV3LazyService) RegisterOpenAPIV3VersionedService(servePath string, handler common.PathHandlerByGroupVersion) error {
	handler.Handle(servePath, http.HandlerFunc(o.HandleDiscovery))
	handler.HandlePrefix(servePath+"/", http.HandlerFunc(o.HandleGroupVersion))
	return nil
}

func computeV3ETag(data []byte) string {
	if data == nil {
		return ""
	}
	return fmt.Sprintf("%X", sha512.Sum512(data))
}

func constructV3ServerRelativeURL(gvString, etag string) string {
	u := url.URL{Path: path.Join("/openapi/v3", gvString)}
	query := url.Values{}
	query.Set("hash", etag)
	u.RawQuery = query.Encode()
	return u.String()
}

// BuiltinV3ETag computes the etag of a built-in group-version spec without
// building it. The spec produced by builder3 for a group-version is a pure
// function of the server binary (its compiled-in OpenAPI definitions and
// builder), the OpenAPI configuration, and the set of registered routes; so
// the etag hashes:
//
//   - identity: an opaque string supplied by the server, expected to include
//     the binary version and any runtime settings that influence the
//     generated definitions (feature gates);
//   - the OpenAPI config's Info block (title, version — which carries the
//     emulation version);
//   - a signature of every registered route of the group-version: method,
//     path, operation name, consumed/produced media types, parameters,
//     response codes, the Go types of the request/response samples and the
//     route metadata (the x-kubernetes-* extensions);
//   - v3ETagFormatRevision.
//
// The output has the same shape as handler3's content hash (upper-case
// SHA-512 hex) so clients cannot distinguish the two.
func BuiltinV3ETag(identity, gv string, routeContainers []common.RouteContainer, config *common.OpenAPIV3Config) string {
	var b strings.Builder
	b.WriteString("openapi-v3-builtin/")
	b.WriteString(v3ETagFormatRevision)
	b.WriteByte('\n')
	b.WriteString(identity)
	b.WriteByte('\n')
	b.WriteString(gv)
	b.WriteByte('\n')
	if config != nil && config.Info != nil {
		if info, err := json.Marshal(config.Info); err == nil {
			b.Write(info)
		}
	}
	b.WriteByte('\n')

	var sigs []string
	for _, rc := range routeContainers {
		var params []string
		for _, p := range rc.PathParameters() {
			params = append(params, parameterSignature(p))
		}
		sort.Strings(params)
		containerSig := rc.RootPath() + "\x00" + strings.Join(params, "\x01")
		for _, r := range rc.Routes() {
			sigs = append(sigs, containerSig+"\x00"+routeSignature(r))
		}
	}
	sort.Strings(sigs)
	for _, s := range sigs {
		b.WriteString(s)
		b.WriteByte('\n')
	}
	return computeV3ETag([]byte(b.String()))
}

func parameterSignature(p common.Parameter) string {
	return fmt.Sprintf("%s|%d|%s|%t|%t", p.Name(), p.Kind(), p.DataType(), p.Required(), p.AllowMultiple())
}

func routeSignature(r common.Route) string {
	var params []string
	for _, p := range r.Parameters() {
		params = append(params, parameterSignature(p))
	}
	sort.Strings(params)
	var responses []string
	for _, resp := range r.StatusCodeResponses() {
		responses = append(responses, fmt.Sprintf("%d|%T", resp.Code(), resp.Model()))
	}
	sort.Strings(responses)
	consumes := append([]string(nil), r.Consumes()...)
	sort.Strings(consumes)
	produces := append([]string(nil), r.Produces()...)
	sort.Strings(produces)
	// fmt prints maps with sorted keys, giving a deterministic rendering of
	// the metadata (x-kubernetes-group-version-kind, x-kubernetes-action, …).
	return strings.Join([]string{
		r.Method(),
		r.Path(),
		r.OperationName(),
		strings.Join(consumes, ","),
		strings.Join(produces, ","),
		strings.Join(params, "\x01"),
		strings.Join(responses, "\x01"),
		fmt.Sprintf("%T", r.RequestPayloadSample()),
		fmt.Sprintf("%T", r.ResponsePayloadSample()),
		fmt.Sprintf("%v", r.Metadata()),
	}, "\x00")
}

// InstallV3Lazy is the lazy variant of InstallV3, used when the
// OpenAPIV3LazyBuild feature gate is enabled. Every built-in group-version is
// registered with a builder and a precomputed etag (see BuiltinV3ETag, salted
// with identity) instead of being built up front. Served content is identical
// to InstallV3 once built; only serialized bytes are retained afterwards.
func (oa OpenAPI) InstallV3Lazy(c *restful.Container, mux *mux.PathRecorderMux, identity string) *OpenAPIV3LazyService {
	service := NewOpenAPIV3LazyService()
	if err := service.RegisterOpenAPIV3VersionedService("/openapi/v3", mux); err != nil {
		klog.Fatalf("Failed to register versioned open api spec for root: %v", err)
	}

	for _, t := range c.RegisteredWebServices() {
		// Strip the "/" prefix from the name
		gv := t.RootPath()[1:]
		ws := []*restful.WebService{t}
		routeContainers := restfuladapter.AdaptWebServices(ws)
		etag := BuiltinV3ETag(identity, gv, routeContainers, oa.V3Config)
		config := oa.V3Config
		service.AddBuiltinGroupVersion(gv, etag, func() (*spec3.OpenAPI, error) {
			klog.V(2).Infof("Building OpenAPI v3 spec for %s on first use", gv)
			spec, err := builder3.BuildOpenAPISpecFromRoutes(routeContainers, config)
			if err != nil {
				return nil, err
			}
			if group, version, ok := groupVersionFromPath(gv); ok {
				filterScopedGVKs(spec, group, version)
			}
			return spec, nil
		})
	}
	return service
}
