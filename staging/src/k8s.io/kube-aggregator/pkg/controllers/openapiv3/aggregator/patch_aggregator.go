package aggregator

import (
	"bytes"
	"context"
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"net/http"
	neturl "net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/munnerz/goautoneg"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog/v2"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/spec3"
)

// mergeOpenAPIV3RootPaths expects mapping of openapi v3 sub url key to multiple serverRelativeURLs
// these URLs can be backed by different ApiServices or CRDs.
//
// We expect duplicates for the following groups:
// authorization.openshift.io, security.openshift.io and quota.openshift.io
// which are backed by both CRD apiextension apiserver and openshift apiserver.
func mergeOpenAPIV3RootPaths(paths map[string][]handler3.OpenAPIV3DiscoveryGroupVersion) handler3.OpenAPIV3Discovery {
	merged := handler3.OpenAPIV3Discovery{
		Paths: make(map[string]handler3.OpenAPIV3DiscoveryGroupVersion),
	}

	for key, delegationURLs := range paths {
		// some apiservices can have duplicate paths in openapi v3 discovery (same path and hash) as they are backed by the same apiserver
		delegationUniqueURLs := sets.List(toUniqueRelativeURLs(delegationURLs))
		// we either have just one url or a special URL like a /version
		if len(delegationUniqueURLs) == 1 || (len(delegationUniqueURLs) > 1 && !hasPrefix(delegationUniqueURLs, "/openapi/v3/apis/")) {
			merged.Paths[key] = handler3.OpenAPIV3DiscoveryGroupVersion{
				ServerRelativeURL: delegationURLs[0].ServerRelativeURL, // take first found apiServiceInfo
			}
		} else {
			newMergedURL, err := mergeURLETags(delegationUniqueURLs)
			if err != nil {
				klog.Errorf("failed create merged openapi v3 URL for: %s: %s", key, err.Error())
				continue
			}
			merged.Paths[key] = handler3.OpenAPIV3DiscoveryGroupVersion{
				ServerRelativeURL: newMergedURL.String(),
			}

		}
	}
	return merged
}

// delegateAndMergeHandleGroupVersion delegates requests to eligibleURLs and merges their output
//
// We expect to delegate and merge for the following groups:
// authorization.openshift.io, security.openshift.io and quota.openshift.io
// which are backed by both CRD apiextension apiserver and openshift apiserver.
//
// The other requests will be passed to the original apiServiceInfo handler.
func delegateAndMergeHandleGroupVersion(w http.ResponseWriter, r *http.Request, eligibleURLs []string, eligibleURLsToAPIServiceInfos map[string]*openAPIV3APIServiceInfo) {
	if len(eligibleURLs) == 1 {
		// fully delegate to the handler
		eligibleURLsToAPIServiceInfos[eligibleURLs[0]].handler.ServeHTTP(w, r)
		return
	} else if len(eligibleURLs) > 1 {
		mergedURL, err := mergeURLETags(eligibleURLs)
		if err != nil {
			klog.Errorf("failed to get mergedURL: %s", err.Error())
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		if !isHashCurrent(r.URL, mergedURL.Query().Get("hash")) {
			http.Redirect(w, r, mergedURL.String(), 301)
			return

		}
		var specs []*spec3.OpenAPI
		var maxLastModified time.Time

		for eligibleURL, apiServiceInfo := range eligibleURLsToAPIServiceInfos {
			writer := newInMemoryResponseWriter()
			req, err := createNewAPIServiceRequest(r, eligibleURL)
			if err != nil {
				klog.Errorf("failed to create request: %s", err.Error())
				continue
			}
			// delegate to multiple apiService handlers
			apiServiceInfo.handler.ServeHTTP(writer, req)
			lastModified, err := time.Parse(time.RFC1123, writer.Header().Get("Last-Modified"))
			if err != nil {
				klog.Warningf("not received Last-Modified in RFC1123 format: %s", err.Error())
			} else if lastModified.After(maxLastModified) {
				maxLastModified = lastModified
			}

			spec := spec3.OpenAPI{}
			if err := json.Unmarshal(writer.data, &spec); err != nil {
				klog.Errorf("failed to unmarshal OpenAPI for openapiService %v/%v: %s", apiServiceInfo.apiService.Namespace, apiServiceInfo.apiService.Name, err.Error())
				continue
			}
			specs = append(specs, &spec)
		}

		// prefer info and version from external apiServices (will result in openshift title and description)
		sort.Slice(specs, func(i, j int) bool {
			if info := specs[i].Info; info != nil && strings.HasPrefix(strings.ToLower(info.Title), "kubernetes") {
				return false
			}
			return true
		})
		mergedSpec, err := mergeSpecsV3(specs...)
		if err != nil {
			klog.Errorf("failed to merge spec: %s", err.Error())
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		mergedSpecJSON, _ := json.Marshal(mergedSpec)

		if maxLastModified.IsZero() {
			maxLastModified = time.Now()
		}

		openAPIHandleGroupVersion(w, r, mergedSpecJSON, mergedURL.Query().Get("hash"), maxLastModified)
	}
}

// openAPIHandleGroupVersion is mostly copied from https://github.com/kubernetes/kube-openapi/blob/3c0fae5ee9fdc4e0cb7abff6fd66784a1f0dbcf8/pkg/handler3/handler.go#L222
func openAPIHandleGroupVersion(w http.ResponseWriter, r *http.Request, data []byte, etag string, lastModified time.Time) {
	const (
		subTypeProtobufDeprecated = "com.github.proto-openapi.spec.v3@v1.0+protobuf"
		subTypeProtobuf           = "com.github.proto-openapi.spec.v3.v1.0+protobuf"
		subTypeJSON               = "json"
	)

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

			switch accepts.SubType {
			case subTypeProtobuf, subTypeProtobufDeprecated:
				var err error
				data, err = handler3.ToV3ProtoBinary(data)
				if err != nil {
					klog.Errorf("failed to convert json to proto: %v", err)
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
			}
			// Set Content-Type header in the reponse
			w.Header().Set("Content-Type", accepts.ReturnedContentType)

			// ETag must be enclosed in double quotes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
			w.Header().Set("Etag", strconv.Quote(etag))

			if hash := r.URL.Query().Get("hash"); hash != "" {
				// validity of hash checked in handleGroupVersion with isHashCurrent

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

func toUniqueRelativeURLs(relativeURLs []handler3.OpenAPIV3DiscoveryGroupVersion) sets.Set[string] {
	uniqueURLs := sets.New[string]()
	for _, url := range relativeURLs {
		uniqueURLs.Insert(url.ServerRelativeURL)
	}
	return uniqueURLs
}

func hasPrefix(urls []string, prefix string) bool {
	if len(urls) == 0 {
		return false
	}
	for _, url := range urls {
		if !strings.HasPrefix(url, prefix) {
			return false
		}
	}
	return true
}

func isHashCurrent(u *neturl.URL, currentETag string) bool {
	if hash := u.Query().Get("hash"); len(hash) > 0 {
		// check if hash is current only if requested
		return hash == currentETag
	}
	return true
}

// computeETag is copied from https://github.com/kubernetes/kubernetes/blob/2c6c4566eff972d6c1320b5f8ad795f88c822d09/staging/src/k8s.io/apiserver/pkg/endpoints/discovery/aggregated/etag.go#L76
func computeETag(data []byte) string {
	if data == nil {
		return ""
	}
	return fmt.Sprintf("%X", sha512.Sum512(data))
}

func mergeURLETags(delegationURLs []string) (*neturl.URL, error) {
	// presume all urls are the same, so take the first one
	newURL, err := neturl.Parse(delegationURLs[0])
	if err != nil {
		return nil, err
	}
	if len(delegationURLs) == 1 {
		return newURL, nil
	}
	// sorted, for consistent hash
	delegationUniqueURLs := sets.List(sets.New(delegationURLs...))
	delegationUniqueURLsBytes, err := json.Marshal(delegationUniqueURLs)
	if err != nil {
		return nil, err
	}
	etag := computeETag(delegationUniqueURLsBytes)

	newQuery := newURL.Query()
	newQuery.Set("hash", etag)
	newURL.RawQuery = newQuery.Encode()
	return newURL, nil
}

func createNewAPIServiceRequest(from *http.Request, eligibleURL string) (*http.Request, error) {
	req := from.Clone(request.WithUser(context.Background(), &user.DefaultInfo{Name: aggregatorUser}))
	req.Header.Set("Accept", "application/json")
	if hash := req.URL.Query().Get("hash"); len(hash) > 0 {
		eligibleParsedURL, err := neturl.Parse(eligibleURL)
		if err != nil {
			return nil, err
		}
		// rewrite to include the latest hash for this apiservice
		q := req.URL.Query()
		q.Set("hash", eligibleParsedURL.Query().Get("hash"))
		req.URL.RawQuery = q.Encode()
	}
	return req, nil
}
