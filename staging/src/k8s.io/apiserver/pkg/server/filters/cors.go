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

package filters

import (
	"net/http"
	"regexp"
	"strings"

	"k8s.io/klog"
)

// TODO: use restful.CrossOriginResourceSharing
// See github.com/emicklei/go-restful/blob/master/examples/restful-CORS-filter.go, and
// github.com/emicklei/go-restful/blob/master/examples/restful-basic-authentication.go
// Or, for a more detailed implementation use https://github.com/martini-contrib/cors
// or implement CORS at your proxy layer.

// WithCORS is a simple CORS implementation that wraps an http Handler.
// Pass nil for allowedMethods and allowedHeaders to use the defaults. If allowedOriginPatterns
// is empty or nil, no CORS support is installed.
func WithCORS(handler http.Handler, allowedOriginPatterns []string, allowedMethods []string, allowedHeaders []string, exposedHeaders []string, allowCredentials string) http.Handler {
	if len(allowedOriginPatterns) == 0 {
		return handler
	}
	allowedOriginPatternsREs := allowedOriginRegexps(allowedOriginPatterns)
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		origin := req.Header.Get("Origin")
		if origin != "" {
			allowed := false
			for _, re := range allowedOriginPatternsREs {
				if allowed = re.MatchString(origin); allowed {
					break
				}
			}
			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				// Set defaults for methods and headers if nothing was passed
				if allowedMethods == nil {
					allowedMethods = []string{"POST", "GET", "OPTIONS", "PUT", "DELETE", "PATCH"}
				}
				if allowedHeaders == nil {
					allowedHeaders = []string{"Content-Type", "Content-Length", "Accept-Encoding", "X-CSRF-Token", "Authorization", "X-Requested-With", "If-Modified-Since"}
				}
				if exposedHeaders == nil {
					exposedHeaders = []string{"Date"}
				}
				w.Header().Set("Access-Control-Allow-Methods", strings.Join(allowedMethods, ", "))
				w.Header().Set("Access-Control-Allow-Headers", strings.Join(allowedHeaders, ", "))
				w.Header().Set("Access-Control-Expose-Headers", strings.Join(exposedHeaders, ", "))
				w.Header().Set("Access-Control-Allow-Credentials", allowCredentials)

				// Stop here if its a preflight OPTIONS request
				if req.Method == "OPTIONS" {
					w.WriteHeader(http.StatusNoContent)
					return
				}
			}
		}
		// Dispatch to the next handler
		handler.ServeHTTP(w, req)
	})
}

func allowedOriginRegexps(allowedOrigins []string) []*regexp.Regexp {
	res, err := compileRegexps(allowedOrigins)
	if err != nil {
		klog.Fatalf("Invalid CORS allowed origin, --cors-allowed-origins flag was set to %v - %v", strings.Join(allowedOrigins, ","), err)
	}
	return res
}

// Takes a list of strings and compiles them into a list of regular expressions
func compileRegexps(regexpStrings []string) ([]*regexp.Regexp, error) {
	regexps := []*regexp.Regexp{}
	for _, regexpStr := range regexpStrings {
		r, err := regexp.Compile(regexpStr)
		if err != nil {
			return []*regexp.Regexp{}, err
		}
		regexps = append(regexps, r)
	}
	return regexps, nil
}
