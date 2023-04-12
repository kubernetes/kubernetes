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

	"k8s.io/klog/v2"
)

// TODO: use restful.CrossOriginResourceSharing
// See github.com/emicklei/go-restful/blob/master/examples/cors/restful-CORS-filter.go, and
// github.com/emicklei/go-restful/blob/master/examples/basicauth/restful-basic-authentication.go
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

	// Set defaults for methods and headers if nothing was passed
	if allowedMethods == nil {
		allowedMethods = []string{"POST", "GET", "OPTIONS", "PUT", "DELETE", "PATCH"}
	}
	allowMethodsResponseHeader := strings.Join(allowedMethods, ", ")

	if allowedHeaders == nil {
		allowedHeaders = []string{"Content-Type", "Content-Length", "Accept-Encoding", "X-CSRF-Token", "Authorization", "X-Requested-With", "If-Modified-Since"}
	}
	allowHeadersResponseHeader := strings.Join(allowedHeaders, ", ")

	if exposedHeaders == nil {
		exposedHeaders = []string{"Date"}
	}
	exposeHeadersResponseHeader := strings.Join(exposedHeaders, ", ")

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		origin := req.Header.Get("Origin")
		if origin == "" {
			handler.ServeHTTP(w, req)
			return
		}
		if !isOriginAllowed(origin, allowedOriginPatternsREs) {
			handler.ServeHTTP(w, req)
			return
		}

		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Access-Control-Allow-Methods", allowMethodsResponseHeader)
		w.Header().Set("Access-Control-Allow-Headers", allowHeadersResponseHeader)
		w.Header().Set("Access-Control-Expose-Headers", exposeHeadersResponseHeader)
		w.Header().Set("Access-Control-Allow-Credentials", allowCredentials)

		// Stop here if its a preflight OPTIONS request
		if req.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		// Dispatch to the next handler
		handler.ServeHTTP(w, req)
	})
}

// isOriginAllowed returns true if the given origin header in the
// request is allowed CORS.
//
// From https://www.rfc-editor.org/rfc/rfc6454#page-13
//
//	 a) The origin header can contain host and/or port
//			serialized-origin   = scheme "://" host [ ":" port ]
//
//	 b) In some cases, a number of origins contribute to causing the user
//	 agents to issue an HTTP request.  In those cases, the user agent MAY
//	 list all the origins in the Origin header field. For example, if the
//	 HTTP request was initially issued by one origin but then later
//	 redirected by another origin, the user agent MAY inform the server
//	 that two origins were involved in causing the user agent to issue the
//	 request
//			origin-list = serialized-origin *( SP serialized-origin )
func isOriginAllowed(originHeader string, allowedOriginPatternsREs []*regexp.Regexp) bool {
	for _, re := range allowedOriginPatternsREs {
		if re.MatchString(originHeader) {
			return true
		}
	}
	return false
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
