/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// mediaTypesForSerializer returns a list of media and stream media types for the server.
func mediaTypesForSerializer(ns runtime.NegotiatedSerializer) (mediaTypes, streamMediaTypes []string) {
	for _, info := range ns.SupportedMediaTypes() {
		mediaTypes = append(mediaTypes, info.MediaType)
		if info.StreamSerializer != nil {
			// stream=watch is the existing mime-type parameter for watch
			streamMediaTypes = append(streamMediaTypes, info.MediaType+";stream=watch")
		}
	}
	return mediaTypes, streamMediaTypes
}

func negotiateOutputSerializer(req *http.Request, ns runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	mediaType, ok := negotiateMediaTypeOptions(req.Header.Get("Accept"), acceptedMediaTypesForEndpoint(ns), defaultEndpointRestrictions)
	if !ok {
		supported, _ := mediaTypesForSerializer(ns)
		return runtime.SerializerInfo{}, errNotAcceptable{supported}
	}
	// TODO: move into resthandler
	info := mediaType.accepted.Serializer
	if (mediaType.pretty || isPrettyPrint(req)) && info.PrettySerializer != nil {
		info.Serializer = info.PrettySerializer
	}
	return info, nil
}

func negotiateOutputStreamSerializer(req *http.Request, ns runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	mediaType, ok := negotiateMediaTypeOptions(req.Header.Get("Accept"), acceptedMediaTypesForEndpoint(ns), defaultEndpointRestrictions)
	if !ok || mediaType.accepted.Serializer.StreamSerializer == nil {
		_, supported := mediaTypesForSerializer(ns)
		return runtime.SerializerInfo{}, errNotAcceptable{supported}
	}
	return mediaType.accepted.Serializer, nil
}

func negotiateInputSerializer(req *http.Request, ns runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	mediaTypes := ns.SupportedMediaTypes()
	mediaType := req.Header.Get("Content-Type")
	if len(mediaType) == 0 {
		mediaType = mediaTypes[0].MediaType
	}
	mediaType, _, err := mime.ParseMediaType(mediaType)
	if err != nil {
		_, supported := mediaTypesForSerializer(ns)
		return runtime.SerializerInfo{}, errUnsupportedMediaType{supported}
	}

	for _, info := range mediaTypes {
		if info.MediaType != mediaType {
			continue
		}
		return info, nil
	}

	_, supported := mediaTypesForSerializer(ns)
	return runtime.SerializerInfo{}, errUnsupportedMediaType{supported}
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

// endpointRestrictions is an interface that allows content-type negotiation
// to verify server support for specific options
type endpointRestrictions interface {
	// AllowsConversion should return true if the specified group version kind
	// is an allowed target object.
	AllowsConversion(unversioned.GroupVersionKind) bool
	// AllowsServerVersion should return true if the specified version is valid
	// for the server group.
	AllowsServerVersion(version string) bool
	// AllowsStreamSchema should return true if the specified stream schema is
	// valid for the server group.
	AllowsStreamSchema(schema string) bool
}

var defaultEndpointRestrictions = emptyEndpointRestrictions{}

type emptyEndpointRestrictions struct{}

func (emptyEndpointRestrictions) AllowsConversion(unversioned.GroupVersionKind) bool { return false }
func (emptyEndpointRestrictions) AllowsServerVersion(string) bool                    { return false }
func (emptyEndpointRestrictions) AllowsStreamSchema(s string) bool                   { return s == "watch" }

// acceptedMediaType contains information about a valid media type that the
// server can serialize.
type acceptedMediaType struct {
	// Type is the first part of the media type ("application")
	Type string
	// SubType is the second part of the media type ("json")
	SubType string
	// Serializer is the serialization info this object accepts
	Serializer runtime.SerializerInfo
}

// mediaTypeOptions describes information for a given media type that may alter
// the server response
type mediaTypeOptions struct {
	// pretty is true if the requested representation should be formatted for human
	// viewing
	pretty bool

	// stream, if set, indicates that a streaming protocol variant of this encoding
	// is desired. The only currently supported value is watch which returns versioned
	// events. In the future, this may refer to other stream protocols.
	stream string

	// convert is a request to alter the type of object returned by the server from the
	// normal response
	convert *unversioned.GroupVersionKind
	// useServerVersion is an optional version for the server group
	useServerVersion string

	// export is true if the representation requested should exclude fields the server
	// has set
	export bool

	// unrecognized is a list of all unrecognized keys
	unrecognized []string

	// the accepted media type from the client
	accepted *acceptedMediaType
}

// acceptMediaTypeOptions returns an options object that matches the provided media type params. If
// it returns false, the provided options are not allowed and the media type must be skipped.  These
// parameters are unversioned and may not be changed.
func acceptMediaTypeOptions(params map[string]string, accepts *acceptedMediaType, endpoint endpointRestrictions) (mediaTypeOptions, bool) {
	var options mediaTypeOptions

	// extract all known parameters
	for k, v := range params {
		switch k {

		// controls transformation of the object when returned
		case "as":
			if options.convert == nil {
				options.convert = &unversioned.GroupVersionKind{}
			}
			options.convert.Kind = v
		case "g":
			if options.convert == nil {
				options.convert = &unversioned.GroupVersionKind{}
			}
			options.convert.Group = v
		case "v":
			if options.convert == nil {
				options.convert = &unversioned.GroupVersionKind{}
			}
			options.convert.Version = v

		// controls the streaming schema
		case "stream":
			if len(v) > 0 && (accepts.Serializer.StreamSerializer == nil || !endpoint.AllowsStreamSchema(v)) {
				return mediaTypeOptions{}, false
			}
			options.stream = v

		// controls the version of the server API group used
		// for generic output
		case "sv":
			if len(v) > 0 && !endpoint.AllowsServerVersion(v) {
				return mediaTypeOptions{}, false
			}
			options.useServerVersion = v

		// if specified, the server should transform the returned
		// output and remove fields that are always server specified,
		// or which fit the default behavior.
		case "export":
			options.export = v == "1"

		// if specified, the pretty serializer will be used
		case "pretty":
			options.pretty = v == "1"

		default:
			options.unrecognized = append(options.unrecognized, k)
		}
	}

	if options.convert != nil && !endpoint.AllowsConversion(*options.convert) {
		return mediaTypeOptions{}, false
	}

	options.accepted = accepts

	return options, true
}

// negotiateMediaTypeOptions returns the most appropriate content type given the accept header and
// a list of alternatives along with the accepted media type parameters.
func negotiateMediaTypeOptions(header string, accepted []acceptedMediaType, endpoint endpointRestrictions) (mediaTypeOptions, bool) {
	if len(header) == 0 && len(accepted) > 0 {
		return mediaTypeOptions{
			accepted: &accepted[0],
		}, true
	}

	clauses := goautoneg.ParseAccept(header)
	for _, clause := range clauses {
		for i := range accepted {
			accepts := &accepted[i]
			switch {
			case clause.Type == accepts.Type && clause.SubType == accepts.SubType,
				clause.Type == accepts.Type && clause.SubType == "*",
				clause.Type == "*" && clause.SubType == "*":
				// TODO: should we prefer the first type with no unrecognized options?  Do we need to ignore unrecognized
				// parameters.
				return acceptMediaTypeOptions(clause.Params, accepts, endpoint)
			}
		}
	}
	return mediaTypeOptions{}, false
}

// acceptedMediaTypesForEndpoint returns an array of structs that are used to efficiently check which
// allowed media types the server exposes.
func acceptedMediaTypesForEndpoint(ns runtime.NegotiatedSerializer) []acceptedMediaType {
	var acceptedMediaTypes []acceptedMediaType
	for _, info := range ns.SupportedMediaTypes() {
		segments := strings.SplitN(info.MediaType, "/", 2)
		if len(segments) == 1 {
			segments = append(segments, "*")
		}
		t := acceptedMediaType{
			Type:       segments[0],
			SubType:    segments[1],
			Serializer: info,
		}
		acceptedMediaTypes = append(acceptedMediaTypes, t)
	}
	return acceptedMediaTypes
}
