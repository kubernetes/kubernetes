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
	"fmt"
	"mime"
	"net/http"
	"strconv"
	"strings"

	"bitbucket.org/ww/goautoneg"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

type NegotiatedSerializer interface {
	SupportedMediaTypes() []string
	SerializerForMediaType(mediaType string, options map[string]string) (runtime.Serializer, bool)
	EncoderForVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Encoder
	DecoderToVersion(serializer runtime.Serializer, gv unversioned.GroupVersion) runtime.Decoder
}

func negotiateOutputSerializer(req *http.Request, s NegotiatedSerializer) (runtime.Serializer, string, error) {
	accept := goautoneg.ParseAccept(req.Header.Get("Accept"))
	pretty := isPrettyPrint(req)
	for _, t := range accept {
		if _, ok := t.Params["pretty"]; !ok && pretty {
			t.Params["pretty"] = "1"
		}
		mediaType := t.Type + "/" + t.SubType
		if s, ok := s.SerializerForMediaType(mediaType, t.Params); ok {
			return s, mediaType, nil
		}
	}
	// TODO: return the correct HTTP error
	return nil, "", errors.NewInternalError(fmt.Errorf("no supported content type"))
}

func negotiateInputSerializer(req *http.Request, s NegotiatedSerializer) (runtime.Serializer, error) {
	mediaType := req.Header.Get("Content-Type")
	if len(mediaType) == 0 {
		// TODO: return the correct http error
		return nil, errors.NewInternalError(fmt.Errorf("content type is required"))
	}
	mediaType, options, err := mime.ParseMediaType(mediaType)
	if err != nil {
		// TODO: return the correct http error
		return nil, errors.NewInternalError(fmt.Errorf("invalid content type: %v", err))
	}
	out, ok := s.SerializerForMediaType(mediaType, options)
	if !ok {
		return nil, errors.NewInternalError(fmt.Errorf("content type is not recognized"))
	}
	return out, nil
}

// isPrettyPrint returns true if the "pretty" query parameter is true or if the User-Agent
// matches known "human" clients.
func isPrettyPrint(req *http.Request) bool {
	// DEPRECATED: should be part of the content type
	pp := req.URL.Query().Get("pretty")
	if len(pp) > 0 {
		pretty, _ := strconv.ParseBool(pp)
		return pretty
	}
	userAgent := req.UserAgent()
	// This covers basic all browers and cli http tools
	if strings.HasPrefix(userAgent, "curl") || strings.HasPrefix(userAgent, "Wget") || strings.HasPrefix(userAgent, "Mozilla/5.0") {
		return true
	}
	return false
}
