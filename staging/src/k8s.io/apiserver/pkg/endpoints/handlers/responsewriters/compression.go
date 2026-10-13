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

package responsewriters

import (
	"compress/gzip"
	"net/http"
	"strings"
	"sync"

	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

const (
	// gzipContentEncodingLevel is set to 1 which uses least CPU compared to higher levels, yet offers
	// similar compression ratios (off by at most 1.5x, but typically within 1.1x-1.3x). For further details see -
	// https://github.com/kubernetes/kubernetes/issues/112296
	gzipContentEncodingLevel = 1
)

// NewGzipWriterPoolOrDie returns a sync.Pool of gzip.Writers configured with gzipContentEncodingLevel.
func NewGzipWriterPoolOrDie() *sync.Pool {
	return &sync.Pool{
		New: func() interface{} {
			gw, err := gzip.NewWriterLevel(nil, gzipContentEncodingLevel)
			if err != nil {
				panic(err)
			}
			return gw
		},
	}
}

// responseContentEncodingSupported returns a supported client-requested content encoding for the
// provided request. It will return the empty string if no supported content encoding was
// found or if the APIResponseCompression feature gate is disabled.
func responseContentEncodingSupported(req *http.Request) string {
	return ContentEncodingSupported(req, features.APIResponseCompression)
}

// ContentEncodingSupported returns a supported client-requested content encoding for the
// provided request. It will return the empty string if no supported content encoding was
// found or if the provided feature gate is disabled.
func ContentEncodingSupported(req *http.Request, feature featuregate.Feature) string {
	encoding := req.Header.Get("Accept-Encoding")
	if len(encoding) == 0 {
		return ""
	}
	if !utilfeature.DefaultFeatureGate.Enabled(feature) {
		return ""
	}
	for len(encoding) > 0 {
		var token string
		if next := strings.Index(encoding, ","); next != -1 {
			token = encoding[:next]
			encoding = encoding[next+1:]
		} else {
			token = encoding
			encoding = ""
		}
		if strings.TrimSpace(token) == "gzip" {
			return "gzip"
		}
	}
	return ""
}
