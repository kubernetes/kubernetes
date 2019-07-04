package restful

import (
	"bytes"
	"strings"
)

// Copyright 2018 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

// PathProcessor is extra behaviour that a Router can provide to extract path parameters from the path.
// If a Router does not implement this interface then the default behaviour will be used.
type PathProcessor interface {
	// ExtractParameters gets the path parameters defined in the route and webService from the urlPath
	ExtractParameters(route *Route, webService *WebService, urlPath string) map[string]string
}

type defaultPathProcessor struct{}

// Extract the parameters from the request url path
func (d defaultPathProcessor) ExtractParameters(r *Route, _ *WebService, urlPath string) map[string]string {
	urlParts := tokenizePath(urlPath)
	pathParameters := map[string]string{}
	for i, key := range r.pathParts {
		var value string
		if i >= len(urlParts) {
			value = ""
		} else {
			value = urlParts[i]
		}
		if strings.HasPrefix(key, "{") { // path-parameter
			if colon := strings.Index(key, ":"); colon != -1 {
				// extract by regex
				regPart := key[colon+1 : len(key)-1]
				keyPart := key[1:colon]
				if regPart == "*" {
					pathParameters[keyPart] = untokenizePath(i, urlParts)
					break
				} else {
					pathParameters[keyPart] = value
				}
			} else {
				// without enclosing {}
				pathParameters[key[1:len(key)-1]] = value
			}
		}
	}
	return pathParameters
}

// Untokenize back into an URL path using the slash separator
func untokenizePath(offset int, parts []string) string {
	var buffer bytes.Buffer
	for p := offset; p < len(parts); p++ {
		buffer.WriteString(parts[p])
		// do not end
		if p < len(parts)-1 {
			buffer.WriteString("/")
		}
	}
	return buffer.String()
}
