package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
)

const (
	mimeTypeJSON        = "application/json"
	mimeTypeOctetStream = "application/octet-stream"
	mimeTypeFormPost    = "application/x-www-form-urlencoded"

	headerAuthorization    = "Authorization"
	headerAuxAuthorization = "x-ms-authorization-auxiliary"
	headerContentType      = "Content-Type"
	headerUserAgent        = "User-Agent"
)

// used as a key type in context.WithValue()
type ctxPrepareDecorators struct{}

// WithPrepareDecorators adds the specified PrepareDecorators to the provided context.
// If no PrepareDecorators are provided the context is unchanged.
func WithPrepareDecorators(ctx context.Context, prepareDecorator []PrepareDecorator) context.Context {
	if len(prepareDecorator) == 0 {
		return ctx
	}
	return context.WithValue(ctx, ctxPrepareDecorators{}, prepareDecorator)
}

// GetPrepareDecorators returns the PrepareDecorators in the provided context or the provided default PrepareDecorators.
func GetPrepareDecorators(ctx context.Context, defaultPrepareDecorators ...PrepareDecorator) []PrepareDecorator {
	inCtx := ctx.Value(ctxPrepareDecorators{})
	if pd, ok := inCtx.([]PrepareDecorator); ok {
		return pd
	}
	return defaultPrepareDecorators
}

// Preparer is the interface that wraps the Prepare method.
//
// Prepare accepts and possibly modifies an http.Request (e.g., adding Headers). Implementations
// must ensure to not share or hold per-invocation state since Preparers may be shared and re-used.
type Preparer interface {
	Prepare(*http.Request) (*http.Request, error)
}

// PreparerFunc is a method that implements the Preparer interface.
type PreparerFunc func(*http.Request) (*http.Request, error)

// Prepare implements the Preparer interface on PreparerFunc.
func (pf PreparerFunc) Prepare(r *http.Request) (*http.Request, error) {
	return pf(r)
}

// PrepareDecorator takes and possibly decorates, by wrapping, a Preparer. Decorators may affect the
// http.Request and pass it along or, first, pass the http.Request along then affect the result.
type PrepareDecorator func(Preparer) Preparer

// CreatePreparer creates, decorates, and returns a Preparer.
// Without decorators, the returned Preparer returns the passed http.Request unmodified.
// Preparers are safe to share and re-use.
func CreatePreparer(decorators ...PrepareDecorator) Preparer {
	return DecoratePreparer(
		Preparer(PreparerFunc(func(r *http.Request) (*http.Request, error) { return r, nil })),
		decorators...)
}

// DecoratePreparer accepts a Preparer and a, possibly empty, set of PrepareDecorators, which it
// applies to the Preparer. Decorators are applied in the order received, but their affect upon the
// request depends on whether they are a pre-decorator (change the http.Request and then pass it
// along) or a post-decorator (pass the http.Request along and alter it on return).
func DecoratePreparer(p Preparer, decorators ...PrepareDecorator) Preparer {
	for _, decorate := range decorators {
		p = decorate(p)
	}
	return p
}

// Prepare accepts an http.Request and a, possibly empty, set of PrepareDecorators.
// It creates a Preparer from the decorators which it then applies to the passed http.Request.
func Prepare(r *http.Request, decorators ...PrepareDecorator) (*http.Request, error) {
	if r == nil {
		return nil, NewError("autorest", "Prepare", "Invoked without an http.Request")
	}
	return CreatePreparer(decorators...).Prepare(r)
}

// WithNothing returns a "do nothing" PrepareDecorator that makes no changes to the passed
// http.Request.
func WithNothing() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			return p.Prepare(r)
		})
	}
}

// WithHeader returns a PrepareDecorator that sets the specified HTTP header of the http.Request to
// the passed value. It canonicalizes the passed header name (via http.CanonicalHeaderKey) before
// adding the header.
func WithHeader(header string, value string) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				setHeader(r, http.CanonicalHeaderKey(header), value)
			}
			return r, err
		})
	}
}

// WithHeaders returns a PrepareDecorator that sets the specified HTTP headers of the http.Request to
// the passed value. It canonicalizes the passed headers name (via http.CanonicalHeaderKey) before
// adding them.
func WithHeaders(headers map[string]interface{}) PrepareDecorator {
	h := ensureValueStrings(headers)
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if r.Header == nil {
					r.Header = make(http.Header)
				}

				for name, value := range h {
					r.Header.Set(http.CanonicalHeaderKey(name), value)
				}
			}
			return r, err
		})
	}
}

// WithBearerAuthorization returns a PrepareDecorator that adds an HTTP Authorization header whose
// value is "Bearer " followed by the supplied token.
func WithBearerAuthorization(token string) PrepareDecorator {
	return WithHeader(headerAuthorization, fmt.Sprintf("Bearer %s", token))
}

// AsContentType returns a PrepareDecorator that adds an HTTP Content-Type header whose value
// is the passed contentType.
func AsContentType(contentType string) PrepareDecorator {
	return WithHeader(headerContentType, contentType)
}

// WithUserAgent returns a PrepareDecorator that adds an HTTP User-Agent header whose value is the
// passed string.
func WithUserAgent(ua string) PrepareDecorator {
	return WithHeader(headerUserAgent, ua)
}

// AsFormURLEncoded returns a PrepareDecorator that adds an HTTP Content-Type header whose value is
// "application/x-www-form-urlencoded".
func AsFormURLEncoded() PrepareDecorator {
	return AsContentType(mimeTypeFormPost)
}

// AsJSON returns a PrepareDecorator that adds an HTTP Content-Type header whose value is
// "application/json".
func AsJSON() PrepareDecorator {
	return AsContentType(mimeTypeJSON)
}

// AsOctetStream returns a PrepareDecorator that adds the "application/octet-stream" Content-Type header.
func AsOctetStream() PrepareDecorator {
	return AsContentType(mimeTypeOctetStream)
}

// WithMethod returns a PrepareDecorator that sets the HTTP method of the passed request. The
// decorator does not validate that the passed method string is a known HTTP method.
func WithMethod(method string) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r.Method = method
			return p.Prepare(r)
		})
	}
}

// AsDelete returns a PrepareDecorator that sets the HTTP method to DELETE.
func AsDelete() PrepareDecorator { return WithMethod("DELETE") }

// AsGet returns a PrepareDecorator that sets the HTTP method to GET.
func AsGet() PrepareDecorator { return WithMethod("GET") }

// AsHead returns a PrepareDecorator that sets the HTTP method to HEAD.
func AsHead() PrepareDecorator { return WithMethod("HEAD") }

// AsMerge returns a PrepareDecorator that sets the HTTP method to MERGE.
func AsMerge() PrepareDecorator { return WithMethod("MERGE") }

// AsOptions returns a PrepareDecorator that sets the HTTP method to OPTIONS.
func AsOptions() PrepareDecorator { return WithMethod("OPTIONS") }

// AsPatch returns a PrepareDecorator that sets the HTTP method to PATCH.
func AsPatch() PrepareDecorator { return WithMethod("PATCH") }

// AsPost returns a PrepareDecorator that sets the HTTP method to POST.
func AsPost() PrepareDecorator { return WithMethod("POST") }

// AsPut returns a PrepareDecorator that sets the HTTP method to PUT.
func AsPut() PrepareDecorator { return WithMethod("PUT") }

// WithBaseURL returns a PrepareDecorator that populates the http.Request with a url.URL constructed
// from the supplied baseUrl.  Query parameters will be encoded as required.
func WithBaseURL(baseURL string) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				var u *url.URL
				if u, err = url.Parse(baseURL); err != nil {
					return r, err
				}
				if u.Scheme == "" {
					return r, fmt.Errorf("autorest: No scheme detected in URL %s", baseURL)
				}
				if u.RawQuery != "" {
					// handle unencoded semicolons (ideally the server would send them already encoded)
					u.RawQuery = strings.Replace(u.RawQuery, ";", "%3B", -1)
					q, err := url.ParseQuery(u.RawQuery)
					if err != nil {
						return r, err
					}
					u.RawQuery = q.Encode()
				}
				r.URL = u
			}
			return r, err
		})
	}
}

// WithBytes returns a PrepareDecorator that takes a list of bytes
// which passes the bytes directly to the body
func WithBytes(input *[]byte) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if input == nil {
					return r, fmt.Errorf("Input Bytes was nil")
				}

				r.ContentLength = int64(len(*input))
				r.Body = ioutil.NopCloser(bytes.NewReader(*input))
			}
			return r, err
		})
	}
}

// WithCustomBaseURL returns a PrepareDecorator that replaces brace-enclosed keys within the
// request base URL (i.e., http.Request.URL) with the corresponding values from the passed map.
func WithCustomBaseURL(baseURL string, urlParameters map[string]interface{}) PrepareDecorator {
	parameters := ensureValueStrings(urlParameters)
	for key, value := range parameters {
		baseURL = strings.Replace(baseURL, "{"+key+"}", value, -1)
	}
	return WithBaseURL(baseURL)
}

// WithFormData returns a PrepareDecoratore that "URL encodes" (e.g., bar=baz&foo=quux) into the
// http.Request body.
func WithFormData(v url.Values) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				s := v.Encode()

				setHeader(r, http.CanonicalHeaderKey(headerContentType), mimeTypeFormPost)
				r.ContentLength = int64(len(s))
				r.Body = ioutil.NopCloser(strings.NewReader(s))
			}
			return r, err
		})
	}
}

// WithMultiPartFormData returns a PrepareDecoratore that "URL encodes" (e.g., bar=baz&foo=quux) form parameters
// into the http.Request body.
func WithMultiPartFormData(formDataParameters map[string]interface{}) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				var body bytes.Buffer
				writer := multipart.NewWriter(&body)
				for key, value := range formDataParameters {
					if rc, ok := value.(io.ReadCloser); ok {
						var fd io.Writer
						if fd, err = writer.CreateFormFile(key, key); err != nil {
							return r, err
						}
						if _, err = io.Copy(fd, rc); err != nil {
							return r, err
						}
					} else {
						if err = writer.WriteField(key, ensureValueString(value)); err != nil {
							return r, err
						}
					}
				}
				if err = writer.Close(); err != nil {
					return r, err
				}
				setHeader(r, http.CanonicalHeaderKey(headerContentType), writer.FormDataContentType())
				r.Body = ioutil.NopCloser(bytes.NewReader(body.Bytes()))
				r.ContentLength = int64(body.Len())
				return r, err
			}
			return r, err
		})
	}
}

// WithFile returns a PrepareDecorator that sends file in request body.
func WithFile(f io.ReadCloser) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				b, err := ioutil.ReadAll(f)
				if err != nil {
					return r, err
				}
				r.Body = ioutil.NopCloser(bytes.NewReader(b))
				r.ContentLength = int64(len(b))
			}
			return r, err
		})
	}
}

// WithBool returns a PrepareDecorator that encodes the passed bool into the body of the request
// and sets the Content-Length header.
func WithBool(v bool) PrepareDecorator {
	return WithString(fmt.Sprintf("%v", v))
}

// WithFloat32 returns a PrepareDecorator that encodes the passed float32 into the body of the
// request and sets the Content-Length header.
func WithFloat32(v float32) PrepareDecorator {
	return WithString(fmt.Sprintf("%v", v))
}

// WithFloat64 returns a PrepareDecorator that encodes the passed float64 into the body of the
// request and sets the Content-Length header.
func WithFloat64(v float64) PrepareDecorator {
	return WithString(fmt.Sprintf("%v", v))
}

// WithInt32 returns a PrepareDecorator that encodes the passed int32 into the body of the request
// and sets the Content-Length header.
func WithInt32(v int32) PrepareDecorator {
	return WithString(fmt.Sprintf("%v", v))
}

// WithInt64 returns a PrepareDecorator that encodes the passed int64 into the body of the request
// and sets the Content-Length header.
func WithInt64(v int64) PrepareDecorator {
	return WithString(fmt.Sprintf("%v", v))
}

// WithString returns a PrepareDecorator that encodes the passed string into the body of the request
// and sets the Content-Length header.
func WithString(v string) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				r.ContentLength = int64(len(v))
				r.Body = ioutil.NopCloser(strings.NewReader(v))
			}
			return r, err
		})
	}
}

// WithJSON returns a PrepareDecorator that encodes the data passed as JSON into the body of the
// request and sets the Content-Length header.
func WithJSON(v interface{}) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				b, err := json.Marshal(v)
				if err == nil {
					r.ContentLength = int64(len(b))
					r.Body = ioutil.NopCloser(bytes.NewReader(b))
				}
			}
			return r, err
		})
	}
}

// WithXML returns a PrepareDecorator that encodes the data passed as XML into the body of the
// request and sets the Content-Length header.
func WithXML(v interface{}) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				b, err := xml.Marshal(v)
				if err == nil {
					// we have to tack on an XML header
					withHeader := xml.Header + string(b)
					bytesWithHeader := []byte(withHeader)

					r.ContentLength = int64(len(bytesWithHeader))
					setHeader(r, headerContentLength, fmt.Sprintf("%d", len(bytesWithHeader)))
					r.Body = ioutil.NopCloser(bytes.NewReader(bytesWithHeader))
				}
			}
			return r, err
		})
	}
}

// WithPath returns a PrepareDecorator that adds the supplied path to the request URL. If the path
// is absolute (that is, it begins with a "/"), it replaces the existing path.
func WithPath(path string) PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if r.URL == nil {
					return r, NewError("autorest", "WithPath", "Invoked with a nil URL")
				}
				if r.URL, err = parseURL(r.URL, path); err != nil {
					return r, err
				}
			}
			return r, err
		})
	}
}

// WithEscapedPathParameters returns a PrepareDecorator that replaces brace-enclosed keys within the
// request path (i.e., http.Request.URL.Path) with the corresponding values from the passed map. The
// values will be escaped (aka URL encoded) before insertion into the path.
func WithEscapedPathParameters(path string, pathParameters map[string]interface{}) PrepareDecorator {
	parameters := escapeValueStrings(ensureValueStrings(pathParameters))
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if r.URL == nil {
					return r, NewError("autorest", "WithEscapedPathParameters", "Invoked with a nil URL")
				}
				for key, value := range parameters {
					path = strings.Replace(path, "{"+key+"}", value, -1)
				}
				if r.URL, err = parseURL(r.URL, path); err != nil {
					return r, err
				}
			}
			return r, err
		})
	}
}

// WithPathParameters returns a PrepareDecorator that replaces brace-enclosed keys within the
// request path (i.e., http.Request.URL.Path) with the corresponding values from the passed map.
func WithPathParameters(path string, pathParameters map[string]interface{}) PrepareDecorator {
	parameters := ensureValueStrings(pathParameters)
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if r.URL == nil {
					return r, NewError("autorest", "WithPathParameters", "Invoked with a nil URL")
				}
				for key, value := range parameters {
					path = strings.Replace(path, "{"+key+"}", value, -1)
				}

				if r.URL, err = parseURL(r.URL, path); err != nil {
					return r, err
				}
			}
			return r, err
		})
	}
}

func parseURL(u *url.URL, path string) (*url.URL, error) {
	p := strings.TrimRight(u.String(), "/")
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return url.Parse(p + path)
}

// WithQueryParameters returns a PrepareDecorators that encodes and applies the query parameters
// given in the supplied map (i.e., key=value).
func WithQueryParameters(queryParameters map[string]interface{}) PrepareDecorator {
	parameters := MapToValues(queryParameters)
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err == nil {
				if r.URL == nil {
					return r, NewError("autorest", "WithQueryParameters", "Invoked with a nil URL")
				}
				v := r.URL.Query()
				for key, value := range parameters {
					for i := range value {
						d, err := url.QueryUnescape(value[i])
						if err != nil {
							return r, err
						}
						value[i] = d
					}
					v[key] = value
				}
				r.URL.RawQuery = v.Encode()
			}
			return r, err
		})
	}
}
