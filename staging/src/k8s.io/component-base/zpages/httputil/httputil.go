/*
Copyright 2024 The Kubernetes Authors.

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

package httputil

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/munnerz/goautoneg"
)

const (
	MediaTypeTextPlain       = "text/plain"
	MediaTypeApplicationJSON = "application/json"
)

// ErrUnsupportedMediaType is the error returned when the request's
// Accept header does not contain "text/plain".
var ErrUnsupportedMediaType = fmt.Errorf("media type not acceptable, must be: text/plain")

// ErrNotAcceptable is the error returned when the request's
// Accept header does not contain the required v, g, and as parameters.
var ErrNotAcceptable = fmt.Errorf("not acceptable")

// AcceptableMediaType checks if the request's Accept header contains
// a supported media type with optional "charset=utf-8" parameter.
func AcceptableMediaType(r *http.Request) bool {
	accepts := goautoneg.ParseAccept(r.Header.Get("Accept"))
	for _, accept := range accepts {
		if !mediaTypeMatches(accept) {
			continue
		}
		if len(accept.Params) == 0 {
			return true
		}
		if len(accept.Params) == 1 {
			if charset, ok := accept.Params["charset"]; ok && strings.EqualFold(charset, "utf-8") {
				return true
			}
		}
	}
	return false
}

func mediaTypeMatches(a goautoneg.Accept) bool {
	return (a.Type == "text" || a.Type == "*") &&
		(a.SubType == "plain" || a.SubType == "*")
}

// NegotiateMediaType negotiates the media type from the request Accept header.
// It returns the best matching media type or an error if no match is found.
func NegotiateMediaType(r *http.Request, supportedTypes []string) (string, error) {
	acceptHeader := r.Header.Get("Accept")
	if acceptHeader == "" && len(supportedTypes) > 0 {
		return MediaTypeTextPlain, nil
	}
	if acceptHeader == "*/*" {
		return MediaTypeTextPlain, nil
	}
	negotiated := goautoneg.Negotiate(acceptHeader, supportedTypes)
	if negotiated == "" {
		return "", ErrNotAcceptable
	}
	return negotiated, nil
}

// NegotiateMediaTypeWithVersion negotiates the media type from the request Accept header
// and also checks for the v, g, and as parameters.
func NegotiateMediaTypeWithVersion(r *http.Request, supportedMediaTypes []string, version, group, kind string) (string, error) {
	mediaType, err := NegotiateMediaType(r, supportedMediaTypes)
	if err != nil {
		return "", err
	}
	if mediaType == MediaTypeApplicationJSON {
		acceptParams := goautoneg.ParseAccept(r.Header.Get("Accept"))
		for _, a := range acceptParams {
			if a.Type == "application" && a.SubType == "json" {
				params := a.Params
				if v, ok := params["v"]; !ok || v != version {
					return "", ErrNotAcceptable
				}
				if g, ok := params["g"]; !ok || g != group {
					return "", ErrNotAcceptable
				}
				if as, ok := params["as"]; !ok || as != kind {
					return "", ErrNotAcceptable
				}
				return mediaType, nil
			}
		}
		return "", ErrNotAcceptable
	}
	return mediaType, nil
}
