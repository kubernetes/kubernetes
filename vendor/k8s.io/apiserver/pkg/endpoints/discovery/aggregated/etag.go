/*
Copyright 2022 The Kubernetes Authors.

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

package aggregated

import (
	"crypto/sha512"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
)

// This file exposes helper functions used for calculating the E-Tag header
// used in discovery endpoint responses

// Attaches Cache-Busting functionality to an endpoint
//   - Sets ETag header to provided hash
//   - Replies with 304 Not Modified, if If-None-Match header matches hash
//
// hash should be the value of calculateETag on object. If hash is empty, then
// the object is simply serialized without E-Tag functionality
func ServeHTTPWithETag(
	object runtime.Object,
	hash string,
	serializer runtime.NegotiatedSerializer,
	w http.ResponseWriter,
	req *http.Request,
) {
	// ETag must be enclosed in double quotes:
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
	quotedHash := strconv.Quote(hash)
	w.Header().Set("ETag", quotedHash)
	w.Header().Set("Vary", "Accept")
	w.Header().Set("Cache-Control", "public")

	// If Request includes If-None-Match and matches hash, reply with 304
	// Otherwise, we delegate to the handler for actual content
	//
	// According to documentation, An Etag within an If-None-Match
	// header will be enclosed within double quotes:
	// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-None-Match#directives
	if clientCachedHash := req.Header.Get("If-None-Match"); quotedHash == clientCachedHash {
		w.WriteHeader(http.StatusNotModified)
		return
	}

	responsewriters.WriteObjectNegotiated(
		serializer,
		DiscoveryEndpointRestrictions,
		AggregatedDiscoveryGV,
		w,
		req,
		http.StatusOK,
		object,
		true,
	)
}

func calculateETag(resources interface{}) (string, error) {
	serialized, err := json.Marshal(resources)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%X", sha512.Sum512(serialized)), nil
}
