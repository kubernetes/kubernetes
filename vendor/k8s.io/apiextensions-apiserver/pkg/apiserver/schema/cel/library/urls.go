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

package library

import (
	"net/url"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/interpreter/functions"
	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"

	"k8s.io/apiextensions-apiserver/third_party/forked/celopenapi/model"
)

// URLs provides a CEL function library extension of URL parsing functions.
//
// url
//
// Converts a string to a URL or results in an error if the string is not a valid URL. The URL must be an absolute URI
// or an absolute path.
//
//     url(<string>) <URL>
//
// Examples:
//
//     url('https://user:pass@example.com:80/path?query=val#fragment') // returns a URL
//     url('/absolute-path') // returns a URL
//     url('https://a:b:c/') // error
//     url('../relative-path') // error
//
// isURL
//
// Returns true if a string is a valid URL. The URL must be an absolute URI or an absolute path.
//
//    isURL( <string>) <bool>
//
// Examples:
//
//     isURL('https://user:pass@example.com:80/path?query=val#fragment') // returns true
//     isURL('/absolute-path') // returns true
//     isURL('https://a:b:c/') // returns false
//     isURL('../relative-path') // returns false
//
//
// getScheme / getHost / getHostname / getPort / getEscapedPath / getQuery
//
// Return the parsed components of a URL.
// - getScheme: If absent in the URL, returns an empty string.
// - getHostname: IPv6 addresses are returned with braces, e.g. "[::1]". If absent in the URL, returns an empty string.
// - getHost: IPv6 addresses are returned without braces, e.g. "::1". If absent in the URL, returns an empty string.
// - getEscapedPath: The string returned by getEscapedPath is URL escaped, e.g. "with space" becomes "with%20space".
//                   If absent in the URL, returns an empty string.
// - getPort: If absent in the URL, returns an empty string.
// - getQuery: Returns the query parameters in "matrix" form where a repeated query key is interpreted to
//             mean that there are multiple values for that key. The keys and values are returned unescaped.
//             If absent in the URL, returns an empty map.
//
//     <URL>.getScheme() <string>
//     <URL>.getHost() <string>
//     <URL>.getHostname() <string>
//     <URL>.getPort() <string>
//     <URL>.getEscapedPath() <string>
//     <URL>.getQuery() <map <string>, <list <string>>
//
// Examples:
//
//     url('/path').getScheme() // returns ''
//     url('https://example.com/').getScheme() // returns 'https'
//     url('https://example.com:80/').getHost() // returns 'example.com:80'
//     url('https://example.com/').getHost() // returns 'example.com'
//     url('https://[::1]:80/').getHost() // returns '[::1]:80'
//     url('https://[::1]/').getHost() // returns '[::1]'
//     url('/path').getHost() // returns ''
//     url('https://example.com:80/').getHostname() // returns 'example.com'
//     url('https://127.0.0.1:80/').getHostname() // returns '127.0.0.1'
//     url('https://[::1]:80/').getHostname() // returns '::1'
//     url('/path').getHostname() // returns ''
//     url('https://example.com:80/').getPort() // returns '80'
//     url('https://example.com/').getPort() // returns ''
//     url('/path').getPort() // returns ''
//     url('https://example.com/path').getEscapedPath() // returns '/path'
//     url('https://example.com/path with spaces/').getEscapedPath() // returns '/path%20with%20spaces/'
//     url('https://example.com').getEscapedPath() // returns ''
//     url('https://example.com/path?k1=a&k2=b&k2=c').getQuery() // returns { 'k1': ['a'], 'k2': ['b', 'c']}
//     url('https://example.com/path?key with spaces=value with spaces').getQuery() // returns { 'key with spaces': ['value with spaces']}
//     url('https://example.com/path?').getQuery() // returns {}
//     url('https://example.com/path').getQuery() // returns {}
//
func URLs() cel.EnvOption {
	return cel.Lib(urlsLib)
}

var urlsLib = &urls{}

type urls struct{}

var urlLibraryDecls = []*exprpb.Decl{
	decls.NewFunction("url",
		decls.NewOverload("string_to_url",
			[]*exprpb.Type{decls.String},
			model.URLObject),
	),
	decls.NewFunction("getScheme",
		decls.NewInstanceOverload("url_get_scheme",
			[]*exprpb.Type{model.URLObject},
			decls.String),
	),
	decls.NewFunction("getHost",
		decls.NewInstanceOverload("url_get_host",
			[]*exprpb.Type{model.URLObject},
			decls.String),
	),
	decls.NewFunction("getHostname",
		decls.NewInstanceOverload("url_get_hostname",
			[]*exprpb.Type{model.URLObject},
			decls.String),
	),
	decls.NewFunction("getPort",
		decls.NewInstanceOverload("url_get_port",
			[]*exprpb.Type{model.URLObject},
			decls.String),
	),
	decls.NewFunction("getEscapedPath",
		decls.NewInstanceOverload("url_get_escaped_path",
			[]*exprpb.Type{model.URLObject},
			decls.String),
	),
	decls.NewFunction("getQuery",
		decls.NewInstanceOverload("url_get_query",
			[]*exprpb.Type{model.URLObject},
			decls.NewMapType(decls.String, decls.NewListType(decls.String))),
	),
	decls.NewFunction("isURL",
		decls.NewOverload("is_url_string",
			[]*exprpb.Type{decls.String},
			decls.Bool),
	),
}

func (*urls) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
		cel.Declarations(urlLibraryDecls...),
	}
}

func (*urls) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{
		cel.Functions(
			&functions.Overload{
				Operator: "url",
				Unary:    stringToUrl,
			},
			&functions.Overload{
				Operator: "string_to_url",
				Unary:    stringToUrl,
			},
			&functions.Overload{
				Operator: "getScheme",
				Unary:    getScheme,
			},
			&functions.Overload{
				Operator: "url_get_scheme",
				Unary:    getScheme,
			},
			&functions.Overload{
				Operator: "getHost",
				Unary:    getHost,
			},
			&functions.Overload{
				Operator: "url_get_host",
				Unary:    getHost,
			},
			&functions.Overload{
				Operator: "getHostname",
				Unary:    getHostname,
			},
			&functions.Overload{
				Operator: "url_get_hostname",
				Unary:    getHostname,
			},
			&functions.Overload{
				Operator: "getPort",
				Unary:    getPort,
			},
			&functions.Overload{
				Operator: "url_get_port",
				Unary:    getPort,
			},
			&functions.Overload{
				Operator: "getEscapedPath",
				Unary:    getEscapedPath,
			},
			&functions.Overload{
				Operator: "url_get_escaped_path",
				Unary:    getEscapedPath,
			},
			&functions.Overload{
				Operator: "getQuery",
				Unary:    getQuery,
			},
			&functions.Overload{
				Operator: "url_get_query",
				Unary:    getQuery,
			},
			&functions.Overload{
				Operator: "isURL",
				Unary:    isURL,
			},
			&functions.Overload{
				Operator: "is_url_string",
				Unary:    isURL,
			},
		),
	}
}

func stringToUrl(arg ref.Val) ref.Val {
	s, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	// Use ParseRequestURI to check the URL before conversion.
	// ParseRequestURI requires absolute URLs and is used by the OpenAPIv3 'uri' format.
	_, err := url.ParseRequestURI(s)
	if err != nil {
		return types.NewErr("URL parse error during conversion from string: %v", err)
	}
	// We must parse again with Parse since ParseRequestURI incorrectly parses URLs that contain a fragment
	// part and will incorrectly append the fragment to either the path or the query, depending on which it was adjacent to.
	u, err := url.Parse(s)
	if err != nil {
		// Errors are not expected here since Parse is a more lenient parser than ParseRequestURI.
		return types.NewErr("URL parse error during conversion from string: %v", err)
	}
	return model.URL{URL: u}
}

func getScheme(arg ref.Val) ref.Val {
	u, ok := arg.Value().(*url.URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(u.Scheme)
}

func getHost(arg ref.Val) ref.Val {
	u, ok := arg.Value().(*url.URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(u.Host)
}

func getHostname(arg ref.Val) ref.Val {
	u, ok := arg.Value().(*url.URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(u.Hostname())
}

func getPort(arg ref.Val) ref.Val {
	u, ok := arg.Value().(*url.URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(u.Port())
}

func getEscapedPath(arg ref.Val) ref.Val {
	u, ok := arg.Value().(*url.URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	return types.String(u.EscapedPath())
}

func getQuery(arg ref.Val) ref.Val {
	u, ok := arg.Value().(*url.URL)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}

	result := map[ref.Val]ref.Val{}
	for k, v := range u.Query() {
		result[types.String(k)] = types.NewStringList(types.DefaultTypeAdapter, v)
	}
	return types.NewRefValMap(types.DefaultTypeAdapter, result)
}

func isURL(arg ref.Val) ref.Val {
	s, ok := arg.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg)
	}
	_, err := url.ParseRequestURI(s)
	return types.Bool(err == nil)
}
