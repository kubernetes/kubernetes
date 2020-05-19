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
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
)

// Responder is the interface that wraps the Respond method.
//
// Respond accepts and reacts to an http.Response. Implementations must ensure to not share or hold
// state since Responders may be shared and re-used.
type Responder interface {
	Respond(*http.Response) error
}

// ResponderFunc is a method that implements the Responder interface.
type ResponderFunc func(*http.Response) error

// Respond implements the Responder interface on ResponderFunc.
func (rf ResponderFunc) Respond(r *http.Response) error {
	return rf(r)
}

// RespondDecorator takes and possibly decorates, by wrapping, a Responder. Decorators may react to
// the http.Response and pass it along or, first, pass the http.Response along then react.
type RespondDecorator func(Responder) Responder

// CreateResponder creates, decorates, and returns a Responder. Without decorators, the returned
// Responder returns the passed http.Response unmodified. Responders may or may not be safe to share
// and re-used: It depends on the applied decorators. For example, a standard decorator that closes
// the response body is fine to share whereas a decorator that reads the body into a passed struct
// is not.
//
// To prevent memory leaks, ensure that at least one Responder closes the response body.
func CreateResponder(decorators ...RespondDecorator) Responder {
	return DecorateResponder(
		Responder(ResponderFunc(func(r *http.Response) error { return nil })),
		decorators...)
}

// DecorateResponder accepts a Responder and a, possibly empty, set of RespondDecorators, which it
// applies to the Responder. Decorators are applied in the order received, but their affect upon the
// request depends on whether they are a pre-decorator (react to the http.Response and then pass it
// along) or a post-decorator (pass the http.Response along and then react).
func DecorateResponder(r Responder, decorators ...RespondDecorator) Responder {
	for _, decorate := range decorators {
		r = decorate(r)
	}
	return r
}

// Respond accepts an http.Response and a, possibly empty, set of RespondDecorators.
// It creates a Responder from the decorators it then applies to the passed http.Response.
func Respond(r *http.Response, decorators ...RespondDecorator) error {
	if r == nil {
		return nil
	}
	return CreateResponder(decorators...).Respond(r)
}

// ByIgnoring returns a RespondDecorator that ignores the passed http.Response passing it unexamined
// to the next RespondDecorator.
func ByIgnoring() RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			return r.Respond(resp)
		})
	}
}

// ByCopying copies the contents of the http.Response Body into the passed bytes.Buffer as
// the Body is read.
func ByCopying(b *bytes.Buffer) RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil && resp != nil && resp.Body != nil {
				resp.Body = TeeReadCloser(resp.Body, b)
			}
			return err
		})
	}
}

// ByDiscardingBody returns a RespondDecorator that first invokes the passed Responder after which
// it copies the remaining bytes (if any) in the response body to ioutil.Discard. Since the passed
// Responder is invoked prior to discarding the response body, the decorator may occur anywhere
// within the set.
func ByDiscardingBody() RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil && resp != nil && resp.Body != nil {
				if _, err := io.Copy(ioutil.Discard, resp.Body); err != nil {
					return fmt.Errorf("Error discarding the response body: %v", err)
				}
			}
			return err
		})
	}
}

// ByClosing returns a RespondDecorator that first invokes the passed Responder after which it
// closes the response body. Since the passed Responder is invoked prior to closing the response
// body, the decorator may occur anywhere within the set.
func ByClosing() RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if resp != nil && resp.Body != nil {
				if err := resp.Body.Close(); err != nil {
					return fmt.Errorf("Error closing the response body: %v", err)
				}
			}
			return err
		})
	}
}

// ByClosingIfError returns a RespondDecorator that first invokes the passed Responder after which
// it closes the response if the passed Responder returns an error and the response body exists.
func ByClosingIfError() RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err != nil && resp != nil && resp.Body != nil {
				if err := resp.Body.Close(); err != nil {
					return fmt.Errorf("Error closing the response body: %v", err)
				}
			}
			return err
		})
	}
}

// ByUnmarshallingBytes returns a RespondDecorator that copies the Bytes returned in the
// response Body into the value pointed to by v.
func ByUnmarshallingBytes(v *[]byte) RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil {
				bytes, errInner := ioutil.ReadAll(resp.Body)
				if errInner != nil {
					err = fmt.Errorf("Error occurred reading http.Response#Body - Error = '%v'", errInner)
				} else {
					*v = bytes
				}
			}
			return err
		})
	}
}

// ByUnmarshallingJSON returns a RespondDecorator that decodes a JSON document returned in the
// response Body into the value pointed to by v.
func ByUnmarshallingJSON(v interface{}) RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil {
				b, errInner := ioutil.ReadAll(resp.Body)
				// Some responses might include a BOM, remove for successful unmarshalling
				b = bytes.TrimPrefix(b, []byte("\xef\xbb\xbf"))
				if errInner != nil {
					err = fmt.Errorf("Error occurred reading http.Response#Body - Error = '%v'", errInner)
				} else if len(strings.Trim(string(b), " ")) > 0 {
					errInner = json.Unmarshal(b, v)
					if errInner != nil {
						err = fmt.Errorf("Error occurred unmarshalling JSON - Error = '%v' JSON = '%s'", errInner, string(b))
					}
				}
			}
			return err
		})
	}
}

// ByUnmarshallingXML returns a RespondDecorator that decodes a XML document returned in the
// response Body into the value pointed to by v.
func ByUnmarshallingXML(v interface{}) RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil {
				b, errInner := ioutil.ReadAll(resp.Body)
				if errInner != nil {
					err = fmt.Errorf("Error occurred reading http.Response#Body - Error = '%v'", errInner)
				} else {
					errInner = xml.Unmarshal(b, v)
					if errInner != nil {
						err = fmt.Errorf("Error occurred unmarshalling Xml - Error = '%v' Xml = '%s'", errInner, string(b))
					}
				}
			}
			return err
		})
	}
}

// WithErrorUnlessStatusCode returns a RespondDecorator that emits an error unless the response
// StatusCode is among the set passed. On error, response body is fully read into a buffer and
// presented in the returned error, as well as in the response body.
func WithErrorUnlessStatusCode(codes ...int) RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err == nil && !ResponseHasStatusCode(resp, codes...) {
				derr := NewErrorWithResponse("autorest", "WithErrorUnlessStatusCode", resp, "%v %v failed with %s",
					resp.Request.Method,
					resp.Request.URL,
					resp.Status)
				if resp.Body != nil {
					defer resp.Body.Close()
					b, _ := ioutil.ReadAll(resp.Body)
					derr.ServiceError = b
					resp.Body = ioutil.NopCloser(bytes.NewReader(b))
				}
				err = derr
			}
			return err
		})
	}
}

// WithErrorUnlessOK returns a RespondDecorator that emits an error if the response StatusCode is
// anything other than HTTP 200.
func WithErrorUnlessOK() RespondDecorator {
	return WithErrorUnlessStatusCode(http.StatusOK)
}

// ExtractHeader extracts all values of the specified header from the http.Response. It returns an
// empty string slice if the passed http.Response is nil or the header does not exist.
func ExtractHeader(header string, resp *http.Response) []string {
	if resp != nil && resp.Header != nil {
		return resp.Header[http.CanonicalHeaderKey(header)]
	}
	return nil
}

// ExtractHeaderValue extracts the first value of the specified header from the http.Response. It
// returns an empty string if the passed http.Response is nil or the header does not exist.
func ExtractHeaderValue(header string, resp *http.Response) string {
	h := ExtractHeader(header, resp)
	if len(h) > 0 {
		return h[0]
	}
	return ""
}
