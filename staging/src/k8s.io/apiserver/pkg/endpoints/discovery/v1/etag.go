/*
Copyright 2016 The Kubernetes Authors.

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

package v1

import (
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// This file exposes helper functions used for calculating the E-Tag header
// used in discovery endpoint responses

// Unsure if there is an existing interface for this
type Marshalable interface {
	Marshal() ([]byte, error)
}

// Attaches Cache-Busting functionality to an endpoint
//   - Immutability Response Header
//   - Expires Never
//   - Redirects when incorrect hash is provided
//   - Sets ETag haeader to provided hash
//   - Replies with 304 Not Modified, if If-None-Match header matches hash
//
// hash should be the value of CalculateETag on object. If hash is empty, then
//
//	the object is simply serialized without E-Tag functionality
func ServeHTTPWithETag(
	object runtime.Object,
	hash string,
	serializer runtime.NegotiatedSerializer,
	w http.ResponseWriter,
	req *http.Request,
) {
	reqURL := req.URL
	if reqURL == nil {
		// Can not find contract guaranteeing any non-nility of req.URL
		w.WriteHeader(500)
		return
	}

	if len(hash) > 0 {
		if providedHash := reqURL.Query().Get("hash"); len(providedHash) > 0 {
			if hash == providedHash {
				// The Vary header is required because the Accept header can
				// change the contents returned. This prevents clients from
				// caching protobuf as JSON and vice versa.
				w.Header().Set("Vary", "Accept")

				// Only set these headers when a hash is given.
				w.Header().Set("Cache-Control", "public, immutable")

				// Set the Expires directive to the maximum value of one year from the request,
				// effectively indicating that the cache never expires.
				w.Header().Set(
					"Expires", time.Now().AddDate(1, 0, 0).Format(time.RFC1123))
			} else {
				// When provided hash is incorrect, reply with redirect
				redirectURL := *reqURL
				query := redirectURL.Query()

				query.Set("hash", hash)
				redirectURL.RawQuery = query.Encode()

				http.Redirect(w, req, redirectURL.String(), http.StatusMovedPermanently)
				return
			}
		}

		// ETag must be enclosed in double quotes:
		// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
		quotedHash := strconv.Quote(hash)
		w.Header().Set("Etag", quotedHash)

		// If Request includes If-None-Match and matches hash, reply with 304
		// Otherwise, we delegate to the handler for actual content
		//
		// According to documentation, An Etag within an If-None-Match
		// header will be enclosed within doule quotes:
		// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-None-Match#directives
		if clientCachedHash := req.Header.Get("If-None-Match"); quotedHash == clientCachedHash {
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}

	responsewriters.WriteObjectNegotiated(
		serializer,
		negotiation.DefaultEndpointRestrictions,
		schema.GroupVersion{},
		w,
		req,
		http.StatusOK,
		object,
	)
}

func CalculateETag(resurces Marshalable) (string, error) {
	serialized, err := json.Marshal(resurces)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%X", sha512.Sum512(serialized)), nil
}
