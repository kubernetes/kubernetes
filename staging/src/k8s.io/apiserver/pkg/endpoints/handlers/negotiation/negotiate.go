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

package negotiation

import (
	"mime"
	"net/http"
	"strconv"
	"strings"

	"github.com/munnerz/goautoneg"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// MediaTypesForSerializer returns a list of media and stream media types for the server.
func MediaTypesForSerializer(ns runtime.NegotiatedSerializer) (mediaTypes, streamMediaTypes []string) {
	for _, info := range ns.SupportedMediaTypes() {
		mediaTypes = append(mediaTypes, info.MediaType)
		if info.StreamSerializer != nil {
			if utilfeature.DefaultFeatureGate.Enabled(features.CBORServingAndStorage) && info.MediaType == runtime.ContentTypeCBOR {
				streamMediaTypes = append(streamMediaTypes, runtime.ContentTypeCBORSequence)
				continue
			}
			// stream=watch is the existing mime-type parameter for watch
			streamMediaTypes = append(streamMediaTypes, info.MediaType+";stream=watch")
		}
	}
	return mediaTypes, streamMediaTypes
}

// NegotiateOutputMediaType negotiates the output structured media type and a serializer, or
// returns an error.
func NegotiateOutputMediaType(req *http.Request, ns runtime.NegotiatedSerializer, restrictions EndpointRestrictions) (MediaTypeOptions, runtime.SerializerInfo, error) {
	mediaType, ok := NegotiateMediaTypeOptions(req.Header.Get("Accept"), ns.SupportedMediaTypes(), restrictions)
	if !ok {
		supported, _ := MediaTypesForSerializer(ns)
		return mediaType, runtime.SerializerInfo{}, NewNotAcceptableError(supported)
	}
	// TODO: move into resthandler
	info := mediaType.Accepted
	if (mediaType.Pretty || isPrettyPrint(req)) && info.PrettySerializer != nil {
		info.Serializer = info.PrettySerializer
	}
	return mediaType, info, nil
}

// NegotiateOutputMediaTypes returns all acceptable media type matches from the Accept
// header in quality order, or NewNotAcceptableError if none match. Callers that want to
// fall through to the next candidate when an encoder cannot represent the object should
// use this instead of NegotiateOutputMediaType. The pretty serializer override is folded
// into each result's Accepted serializer.
func NegotiateOutputMediaTypes(req *http.Request, ns runtime.NegotiatedSerializer, restrictions EndpointRestrictions) ([]MediaTypeOptions, error) {
	mediaTypes := negotiateAllMediaTypeOptions(req.Header.Get("Accept"), ns.SupportedMediaTypes(), restrictions)
	if len(mediaTypes) == 0 {
		supported, _ := MediaTypesForSerializer(ns)
		return nil, NewNotAcceptableError(supported)
	}
	pretty := isPrettyPrint(req)
	for i := range mediaTypes {
		accepted := &mediaTypes[i].Accepted
		if (mediaTypes[i].Pretty || pretty) && accepted.PrettySerializer != nil {
			accepted.Serializer = accepted.PrettySerializer
		}
	}
	return mediaTypes, nil
}

// NegotiateOutputMediaTypeStream returns a stream serializer for the given request.
func NegotiateOutputMediaTypeStream(req *http.Request, ns runtime.NegotiatedSerializer, restrictions EndpointRestrictions) (runtime.SerializerInfo, error) {
	mediaType, ok := NegotiateMediaTypeOptions(req.Header.Get("Accept"), ns.SupportedMediaTypes(), restrictions)
	if !ok || mediaType.Accepted.StreamSerializer == nil {
		_, supported := MediaTypesForSerializer(ns)
		return runtime.SerializerInfo{}, NewNotAcceptableError(supported)
	}
	return mediaType.Accepted, nil
}

// NegotiateInputSerializer returns the input serializer for the provided request.
func NegotiateInputSerializer(req *http.Request, streaming bool, ns runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	mediaType := req.Header.Get("Content-Type")
	return NegotiateInputSerializerForMediaType(mediaType, streaming, ns)
}

// NegotiateInputSerializerForMediaType returns the appropriate serializer for the given media type or an error.
func NegotiateInputSerializerForMediaType(mediaType string, streaming bool, ns runtime.NegotiatedSerializer) (runtime.SerializerInfo, error) {
	mediaTypes := ns.SupportedMediaTypes()
	if len(mediaType) == 0 {
		mediaType = mediaTypes[0].MediaType
	}
	if mediaType, _, err := mime.ParseMediaType(mediaType); err == nil {
		if info, ok := runtime.SerializerInfoForMediaType(mediaTypes, mediaType); ok {
			return info, nil
		}
	}

	supported, streamingSupported := MediaTypesForSerializer(ns)
	if streaming {
		return runtime.SerializerInfo{}, NewUnsupportedMediaTypeError(streamingSupported)
	}
	return runtime.SerializerInfo{}, NewUnsupportedMediaTypeError(supported)
}

// isPrettyPrint returns true if the "pretty" query parameter is true or if the User-Agent
// matches known "human" clients.
func isPrettyPrint(req *http.Request) bool {
	// DEPRECATED: should be part of the content type
	if req.URL != nil {
		// avoid an allocation caused by parsing the URL query
		if strings.Contains(req.URL.RawQuery, "pretty") {
			pp := req.URL.Query().Get("pretty")
			if len(pp) > 0 {
				pretty, _ := strconv.ParseBool(pp)
				return pretty
			}
		}
	}
	userAgent := req.UserAgent()
	// This covers basic all browsers and cli http tools
	if strings.HasPrefix(userAgent, "curl") || strings.HasPrefix(userAgent, "Wget") || strings.HasPrefix(userAgent, "Mozilla/5.0") {
		return true
	}
	return false
}

// EndpointRestrictions is an interface that allows content-type negotiation
// to verify server support for specific options
type EndpointRestrictions interface {
	// AllowsMediaTypeTransform returns true if the endpoint allows either the requested mime type
	// or the requested transformation. If false, the caller should ignore this mime type. If the
	// target is nil, the client is not requesting a transformation.
	AllowsMediaTypeTransform(mimeType, mimeSubType string, target *schema.GroupVersionKind) bool
	// AllowsServerVersion should return true if the specified version is valid
	// for the server group.
	AllowsServerVersion(version string) bool
	// AllowsStreamSchema should return true if the specified stream schema is
	// valid for the server group.
	AllowsStreamSchema(schema string) bool
}

// DefaultEndpointRestrictions is the default EndpointRestrictions which allows
// content-type negotiation to verify server support for specific options
var DefaultEndpointRestrictions = emptyEndpointRestrictions{}

type emptyEndpointRestrictions struct{}

func (emptyEndpointRestrictions) AllowsMediaTypeTransform(mimeType string, mimeSubType string, gvk *schema.GroupVersionKind) bool {
	return gvk == nil
}
func (emptyEndpointRestrictions) AllowsServerVersion(string) bool  { return false }
func (emptyEndpointRestrictions) AllowsStreamSchema(s string) bool { return s == "watch" }

// MediaTypeOptions describes information for a given media type that may alter
// the server response
type MediaTypeOptions struct {
	// pretty is true if the requested representation should be formatted for human
	// viewing
	Pretty bool

	// stream, if set, indicates that a streaming protocol variant of this encoding
	// is desired. The only currently supported value is watch which returns versioned
	// events. In the future, this may refer to other stream protocols.
	Stream string

	// convert is a request to alter the type of object returned by the server from the
	// normal response
	Convert *schema.GroupVersionKind
	// useServerVersion is an optional version for the server group
	UseServerVersion string

	// export is true if the representation requested should exclude fields the server
	// has set
	Export bool

	// profile controls the discovery profile (e.g., "local" for local (non peer-aggregated) discovery)
	Profile string

	// unrecognized is a list of all unrecognized keys
	Unrecognized []string

	// the accepted media type from the client
	Accepted runtime.SerializerInfo
}

// acceptMediaTypeOptions returns an options object that matches the provided media type params. If
// it returns false, the provided options are not allowed and the media type must be skipped.  These
// parameters are unversioned and may not be changed.
func acceptMediaTypeOptions(params map[string]string, accepts *runtime.SerializerInfo, endpoint EndpointRestrictions) (MediaTypeOptions, bool) {
	var options MediaTypeOptions

	// extract all known parameters
	for k, v := range params {
		switch k {

		// controls transformation of the object when returned
		case "as":
			if options.Convert == nil {
				options.Convert = &schema.GroupVersionKind{}
			}
			options.Convert.Kind = v
		case "g":
			if options.Convert == nil {
				options.Convert = &schema.GroupVersionKind{}
			}
			options.Convert.Group = v
		case "v":
			if options.Convert == nil {
				options.Convert = &schema.GroupVersionKind{}
			}
			options.Convert.Version = v

		// controls the streaming schema
		case "stream":
			if len(v) > 0 && (accepts.StreamSerializer == nil || !endpoint.AllowsStreamSchema(v)) {
				return MediaTypeOptions{}, false
			}
			options.Stream = v

		// controls the version of the server API group used
		// for generic output
		case "sv":
			if len(v) > 0 && !endpoint.AllowsServerVersion(v) {
				return MediaTypeOptions{}, false
			}
			options.UseServerVersion = v

		// if specified, the server should transform the returned
		// output and remove fields that are always server specified,
		// or which fit the default behavior.
		case "export":
			options.Export = v == "1"

		// if specified, the pretty serializer will be used
		case "pretty":
			options.Pretty = v == "1"

		// controls the discovery profile (eg local vs peer-aggregated)
		case "profile":
			options.Profile = v

		default:
			options.Unrecognized = append(options.Unrecognized, k)
		}
	}

	if !endpoint.AllowsMediaTypeTransform(accepts.MediaTypeType, accepts.MediaTypeSubType, options.Convert) {
		return MediaTypeOptions{}, false
	}

	options.Accepted = *accepts
	return options, true
}

// matchedMediaTypeOptions reports whether the serializer satisfies the Accept clause and,
// if so, returns the negotiated options. Shared by the singular and plural negotiators.
func matchedMediaTypeOptions(clause *goautoneg.Accept, accepts *runtime.SerializerInfo, endpoint EndpointRestrictions) (MediaTypeOptions, bool) {
	// q=0 means "not acceptable" per RFC 9110 §12.5.1, so the client is rejecting this media type.
	if clause.Q <= 0 {
		return MediaTypeOptions{}, false
	}
	switch {
	case clause.Type == accepts.MediaTypeType && clause.SubType == accepts.MediaTypeSubType,
		clause.Type == accepts.MediaTypeType && clause.SubType == "*",
		clause.Type == "*" && clause.SubType == "*":
		return acceptMediaTypeOptions(clause.Params, accepts, endpoint)
	}
	return MediaTypeOptions{}, false
}

// NegotiateMediaTypeOptions returns the most appropriate content type given the accept header and
// a list of alternatives along with the accepted media type parameters.
func NegotiateMediaTypeOptions(header string, accepted []runtime.SerializerInfo, endpoint EndpointRestrictions) (MediaTypeOptions, bool) {
	if len(header) == 0 && len(accepted) > 0 {
		return MediaTypeOptions{
			Accepted: accepted[0],
		}, true
	}

	clauses := goautoneg.ParseAccept(header)
	rejected := rejectedMediaTypes(clauses, accepted)
	for i := range clauses {
		for j := range accepted {
			if rejected[j] {
				continue
			}
			if retVal, ok := matchedMediaTypeOptions(&clauses[i], &accepted[j], endpoint); ok {
				return retVal, true
			}
		}
	}

	return MediaTypeOptions{}, false
}

// negotiateAllMediaTypeOptions is the plural counterpart to NegotiateMediaTypeOptions:
// it returns every accepted serializer in quality order, deduplicated so a wildcard clause
// does not re-emit a serializer an earlier clause already matched.
func negotiateAllMediaTypeOptions(header string, accepted []runtime.SerializerInfo, endpoint EndpointRestrictions) []MediaTypeOptions {
	if len(header) == 0 && len(accepted) > 0 {
		return []MediaTypeOptions{{Accepted: accepted[0]}}
	}

	clauses := goautoneg.ParseAccept(header)
	rejected := rejectedMediaTypes(clauses, accepted)

	var results []MediaTypeOptions
	seen := make(map[string]struct{})
	for i := range clauses {
		for j := range accepted {
			if rejected[j] {
				continue
			}
			retVal, ok := matchedMediaTypeOptions(&clauses[i], &accepted[j], endpoint)
			if !ok {
				continue
			}
			// An earlier, higher-priority clause may have already emitted this serializer.
			key := accepted[j].MediaType
			if _, dup := seen[key]; dup {
				continue
			}
			seen[key] = struct{}{}
			results = append(results, retVal)
		}
	}

	return results
}

// rejectedMediaTypes reports, per serializer, whether the Accept clauses reject it: a
// serializer is rejected when its most specific matching clause has q<=0 ("not acceptable",
// RFC 9110 §12.5.1). Clauses are quality-sorted, so a specificity tie keeps the highest q.
// This lets application/json;q=0 override a */* that would accept it, while */* still serves
// as a fallback for types that are not rejected. Both negotiators use it so they agree on
// acceptability.
func rejectedMediaTypes(clauses []goautoneg.Accept, accepted []runtime.SerializerInfo) []bool {
	rejected := make([]bool, len(accepted))
	for j := range accepted {
		best := -1
		for i := range clauses {
			if spec := clauseSpecificity(&clauses[i], &accepted[j]); spec > best {
				best = spec
				rejected[j] = clauses[i].Q <= 0
			}
		}
	}
	return rejected
}

// clauseSpecificity ranks how specifically a clause matches a serializer: 2 for an exact
// match, 1 for type/*, 0 for */*, -1 for no match. Quality is ignored so a specific q=0
// rejection still outranks a wildcard.
func clauseSpecificity(clause *goautoneg.Accept, accepts *runtime.SerializerInfo) int {
	switch {
	case clause.Type == accepts.MediaTypeType && clause.SubType == accepts.MediaTypeSubType:
		return 2
	case clause.Type == accepts.MediaTypeType && clause.SubType == "*":
		return 1
	case clause.Type == "*" && clause.SubType == "*":
		return 0
	}
	return -1
}
