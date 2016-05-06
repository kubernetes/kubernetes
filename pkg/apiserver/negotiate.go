/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"mime"
	"net/http"
	"strconv"
	"strings"

	"bitbucket.org/ww/goautoneg"

	"k8s.io/kubernetes/pkg/runtime"
)

func negotiateOutput(req *http.Request, supported []string) (string, map[string]string, error) {
	acceptHeader := req.Header.Get("Accept")
	if len(acceptHeader) == 0 && len(supported) > 0 {
		acceptHeader = supported[0]
	}
	accept, ok := negotiate(acceptHeader, supported)
	if !ok {
		return "", nil, errNotAcceptable{supported}
	}

	pretty := isPrettyPrint(req)
	if _, ok := accept.Params["pretty"]; !ok && pretty {
		accept.Params["pretty"] = "1"
	}

	mediaType := accept.Type
	if len(accept.SubType) > 0 {
		mediaType += "/" + accept.SubType
	}

	return mediaType, accept.Params, nil
}

func negotiateOutputSerializer(req *http.Request, ns runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	supported := ns.SupportedMediaTypes()
	mediaType, params, err := negotiateOutput(req, supported)
	if err != nil {
		return runtime.SerializerInfo{}, err
	}
	if s, ok := ns.SerializerForMediaType(mediaType, params); ok {
		return s, nil
	}
	return runtime.SerializerInfo{}, errNotAcceptable{supported}
}

func negotiateOutputStreamSerializer(req *http.Request, ns runtime.NegotiatedSerializer) (runtime.StreamSerializerInfo, error) {
	supported := ns.SupportedMediaTypes()
	mediaType, params, err := negotiateOutput(req, supported)
	if err != nil {
		return runtime.StreamSerializerInfo{}, err
	}
	if s, ok := ns.StreamingSerializerForMediaType(mediaType, params); ok {
		return s, nil
	}
	return runtime.StreamSerializerInfo{}, errNotAcceptable{supported}
}

func negotiateInputSerializer(req *http.Request, s runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	supported := s.SupportedMediaTypes()
	mediaType := req.Header.Get("Content-Type")
	if len(mediaType) == 0 {
		mediaType = supported[0]
	}
	mediaType, options, err := mime.ParseMediaType(mediaType)
	if err != nil {
		return runtime.SerializerInfo{}, errUnsupportedMediaType{supported}
	}
	out, ok := s.SerializerForMediaType(mediaType, options)
	if !ok {
		return runtime.SerializerInfo{}, errUnsupportedMediaType{supported}
	}
	return out, nil
}

// isPrettyPrint returns true if the "pretty" query parameter is true or if the User-Agent
// matches known "human" clients.
func isPrettyPrint(req *http.Request) bool {
	// DEPRECATED: should be part of the content type
	if req.URL != nil {
		pp := req.URL.Query().Get("pretty")
		if len(pp) > 0 {
			pretty, _ := strconv.ParseBool(pp)
			return pretty
		}
	}
	userAgent := req.UserAgent()
	// This covers basic all browers and cli http tools
	if strings.HasPrefix(userAgent, "curl") || strings.HasPrefix(userAgent, "Wget") || strings.HasPrefix(userAgent, "Mozilla/5.0") {
		return true
	}
	return false
}

// negotiate the most appropriate content type given the accept header and a list of
// alternatives.
func negotiate(header string, alternatives []string) (goautoneg.Accept, bool) {
	alternates := make([][]string, 0, len(alternatives))
	for _, alternate := range alternatives {
		alternates = append(alternates, strings.SplitN(alternate, "/", 2))
	}
	for _, clause := range goautoneg.ParseAccept(header) {
		for _, alternate := range alternates {
			if clause.Type == alternate[0] && clause.SubType == alternate[1] {
				return clause, true
			}
			if clause.Type == alternate[0] && clause.SubType == "*" {
				clause.SubType = alternate[1]
				return clause, true
			}
			if clause.Type == "*" && clause.SubType == "*" {
				clause.Type = alternate[0]
				clause.SubType = alternate[1]
				return clause, true
			}
		}
	}
	return goautoneg.Accept{}, false
}
