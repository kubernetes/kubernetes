// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package googleapi contains the common code shared by all Google API
// libraries.
package googleapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"

	"google.golang.org/api/googleapi/internal/uritemplates"
)

// ContentTyper is an interface for Readers which know (or would like
// to override) their Content-Type. If a media body doesn't implement
// ContentTyper, the type is sniffed from the content using
// http.DetectContentType.
type ContentTyper interface {
	ContentType() string
}

// A SizeReaderAt is a ReaderAt with a Size method.
// An io.SectionReader implements SizeReaderAt.
type SizeReaderAt interface {
	io.ReaderAt
	Size() int64
}

// ServerResponse is embedded in each Do response and
// provides the HTTP status code and header sent by the server.
type ServerResponse struct {
	// HTTPStatusCode is the server's response status code.
	// When using a resource method's Do call, this will always be in the 2xx range.
	HTTPStatusCode int
	// Header contains the response header fields from the server.
	Header http.Header
}

const (
	Version = "0.5"

	// UserAgent is the header string used to identify this package.
	UserAgent = "google-api-go-client/" + Version

	// The default chunk size to use for resumable uplods if not specified by the user.
	DefaultUploadChunkSize = 8 * 1024 * 1024

	// The minimum chunk size that can be used for resumable uploads.  All
	// user-specified chunk sizes must be multiple of this value.
	MinUploadChunkSize = 256 * 1024
)

// Error contains an error response from the server.
type Error struct {
	// Code is the HTTP response status code and will always be populated.
	Code int `json:"code"`
	// Message is the server response message and is only populated when
	// explicitly referenced by the JSON server response.
	Message string `json:"message"`
	// Body is the raw response returned by the server.
	// It is often but not always JSON, depending on how the request fails.
	Body string
	// Header contains the response header fields from the server.
	Header http.Header

	Errors []ErrorItem
}

// ErrorItem is a detailed error code & message from the Google API frontend.
type ErrorItem struct {
	// Reason is the typed error code. For example: "some_example".
	Reason string `json:"reason"`
	// Message is the human-readable description of the error.
	Message string `json:"message"`
}

func (e *Error) Error() string {
	if len(e.Errors) == 0 && e.Message == "" {
		return fmt.Sprintf("googleapi: got HTTP response code %d with body: %v", e.Code, e.Body)
	}
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "googleapi: Error %d: ", e.Code)
	if e.Message != "" {
		fmt.Fprintf(&buf, "%s", e.Message)
	}
	if len(e.Errors) == 0 {
		return strings.TrimSpace(buf.String())
	}
	if len(e.Errors) == 1 && e.Errors[0].Message == e.Message {
		fmt.Fprintf(&buf, ", %s", e.Errors[0].Reason)
		return buf.String()
	}
	fmt.Fprintln(&buf, "\nMore details:")
	for _, v := range e.Errors {
		fmt.Fprintf(&buf, "Reason: %s, Message: %s\n", v.Reason, v.Message)
	}
	return buf.String()
}

type errorReply struct {
	Error *Error `json:"error"`
}

// CheckResponse returns an error (of type *Error) if the response
// status code is not 2xx.
func CheckResponse(res *http.Response) error {
	if res.StatusCode >= 200 && res.StatusCode <= 299 {
		return nil
	}
	slurp, err := ioutil.ReadAll(res.Body)
	if err == nil {
		jerr := new(errorReply)
		err = json.Unmarshal(slurp, jerr)
		if err == nil && jerr.Error != nil {
			if jerr.Error.Code == 0 {
				jerr.Error.Code = res.StatusCode
			}
			jerr.Error.Body = string(slurp)
			return jerr.Error
		}
	}
	return &Error{
		Code:   res.StatusCode,
		Body:   string(slurp),
		Header: res.Header,
	}
}

// IsNotModified reports whether err is the result of the
// server replying with http.StatusNotModified.
// Such error values are sometimes returned by "Do" methods
// on calls when If-None-Match is used.
func IsNotModified(err error) bool {
	if err == nil {
		return false
	}
	ae, ok := err.(*Error)
	return ok && ae.Code == http.StatusNotModified
}

// CheckMediaResponse returns an error (of type *Error) if the response
// status code is not 2xx. Unlike CheckResponse it does not assume the
// body is a JSON error document.
// It is the caller's responsibility to close res.Body.
func CheckMediaResponse(res *http.Response) error {
	if res.StatusCode >= 200 && res.StatusCode <= 299 {
		return nil
	}
	slurp, _ := ioutil.ReadAll(io.LimitReader(res.Body, 1<<20))
	return &Error{
		Code: res.StatusCode,
		Body: string(slurp),
	}
}

type MarshalStyle bool

var WithDataWrapper = MarshalStyle(true)
var WithoutDataWrapper = MarshalStyle(false)

func (wrap MarshalStyle) JSONReader(v interface{}) (io.Reader, error) {
	buf := new(bytes.Buffer)
	if wrap {
		buf.Write([]byte(`{"data": `))
	}
	err := json.NewEncoder(buf).Encode(v)
	if err != nil {
		return nil, err
	}
	if wrap {
		buf.Write([]byte(`}`))
	}
	return buf, nil
}

// endingWithErrorReader from r until it returns an error.  If the
// final error from r is io.EOF and e is non-nil, e is used instead.
type endingWithErrorReader struct {
	r io.Reader
	e error
}

func (er endingWithErrorReader) Read(p []byte) (n int, err error) {
	n, err = er.r.Read(p)
	if err == io.EOF && er.e != nil {
		err = er.e
	}
	return
}

// countingWriter counts the number of bytes it receives to write, but
// discards them.
type countingWriter struct {
	n *int64
}

func (w countingWriter) Write(p []byte) (int, error) {
	*w.n += int64(len(p))
	return len(p), nil
}

// ProgressUpdater is a function that is called upon every progress update of a resumable upload.
// This is the only part of a resumable upload (from googleapi) that is usable by the developer.
// The remaining usable pieces of resumable uploads is exposed in each auto-generated API.
type ProgressUpdater func(current, total int64)

type MediaOption interface {
	setOptions(o *MediaOptions)
}

type contentTypeOption string

func (ct contentTypeOption) setOptions(o *MediaOptions) {
	o.ContentType = string(ct)
	if o.ContentType == "" {
		o.ForceEmptyContentType = true
	}
}

// ContentType returns a MediaOption which sets the Content-Type header for media uploads.
// If ctype is empty, the Content-Type header will be omitted.
func ContentType(ctype string) MediaOption {
	return contentTypeOption(ctype)
}

type chunkSizeOption int

func (cs chunkSizeOption) setOptions(o *MediaOptions) {
	size := int(cs)
	if size%MinUploadChunkSize != 0 {
		size += MinUploadChunkSize - (size % MinUploadChunkSize)
	}
	o.ChunkSize = size
}

// ChunkSize returns a MediaOption which sets the chunk size for media uploads.
// size will be rounded up to the nearest multiple of 256K.
// Media which contains fewer than size bytes will be uploaded in a single request.
// Media which contains size bytes or more will be uploaded in separate chunks.
// If size is zero, media will be uploaded in a single request.
func ChunkSize(size int) MediaOption {
	return chunkSizeOption(size)
}

// MediaOptions stores options for customizing media upload.  It is not used by developers directly.
type MediaOptions struct {
	ContentType           string
	ForceEmptyContentType bool

	ChunkSize int
}

// ProcessMediaOptions stores options from opts in a MediaOptions.
// It is not used by developers directly.
func ProcessMediaOptions(opts []MediaOption) *MediaOptions {
	mo := &MediaOptions{ChunkSize: DefaultUploadChunkSize}
	for _, o := range opts {
		o.setOptions(mo)
	}
	return mo
}

func ResolveRelative(basestr, relstr string) string {
	u, _ := url.Parse(basestr)
	rel, _ := url.Parse(relstr)
	u = u.ResolveReference(rel)
	us := u.String()
	us = strings.Replace(us, "%7B", "{", -1)
	us = strings.Replace(us, "%7D", "}", -1)
	return us
}

// Expand subsitutes any {encoded} strings in the URL passed in using
// the map supplied.
//
// This calls SetOpaque to avoid encoding of the parameters in the URL path.
func Expand(u *url.URL, expansions map[string]string) {
	escaped, unescaped, err := uritemplates.Expand(u.Path, expansions)
	if err == nil {
		u.Path = unescaped
		u.RawPath = escaped
	}
}

// CloseBody is used to close res.Body.
// Prior to calling Close, it also tries to Read a small amount to see an EOF.
// Not seeing an EOF can prevent HTTP Transports from reusing connections.
func CloseBody(res *http.Response) {
	if res == nil || res.Body == nil {
		return
	}
	// Justification for 3 byte reads: two for up to "\r\n" after
	// a JSON/XML document, and then 1 to see EOF if we haven't yet.
	// TODO(bradfitz): detect Go 1.3+ and skip these reads.
	// See https://codereview.appspot.com/58240043
	// and https://codereview.appspot.com/49570044
	buf := make([]byte, 1)
	for i := 0; i < 3; i++ {
		_, err := res.Body.Read(buf)
		if err != nil {
			break
		}
	}
	res.Body.Close()

}

// VariantType returns the type name of the given variant.
// If the map doesn't contain the named key or the value is not a []interface{}, "" is returned.
// This is used to support "variant" APIs that can return one of a number of different types.
func VariantType(t map[string]interface{}) string {
	s, _ := t["type"].(string)
	return s
}

// ConvertVariant uses the JSON encoder/decoder to fill in the struct 'dst' with the fields found in variant 'v'.
// This is used to support "variant" APIs that can return one of a number of different types.
// It reports whether the conversion was successful.
func ConvertVariant(v map[string]interface{}, dst interface{}) bool {
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(v)
	if err != nil {
		return false
	}
	return json.Unmarshal(buf.Bytes(), dst) == nil
}

// A Field names a field to be retrieved with a partial response.
// See https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
//
// Partial responses can dramatically reduce the amount of data that must be sent to your application.
// In order to request partial responses, you can specify the full list of fields
// that your application needs by adding the Fields option to your request.
//
// Field strings use camelCase with leading lower-case characters to identify fields within the response.
//
// For example, if your response has a "NextPageToken" and a slice of "Items" with "Id" fields,
// you could request just those fields like this:
//
//     svc.Events.List().Fields("nextPageToken", "items/id").Do()
//
// or if you were also interested in each Item's "Updated" field, you can combine them like this:
//
//     svc.Events.List().Fields("nextPageToken", "items(id,updated)").Do()
//
// More information about field formatting can be found here:
// https://developers.google.com/+/api/#fields-syntax
//
// Another way to find field names is through the Google API explorer:
// https://developers.google.com/apis-explorer/#p/
type Field string

// CombineFields combines fields into a single string.
func CombineFields(s []Field) string {
	r := make([]string, len(s))
	for i, v := range s {
		r[i] = string(v)
	}
	return strings.Join(r, ",")
}

// A CallOption is an optional argument to an API call.
// It should be treated as an opaque value by users of Google APIs.
//
// A CallOption is something that configures an API call in a way that is
// not specific to that API; for instance, controlling the quota user for
// an API call is common across many APIs, and is thus a CallOption.
type CallOption interface {
	Get() (key, value string)
}

// QuotaUser returns a CallOption that will set the quota user for a call.
// The quota user can be used by server-side applications to control accounting.
// It can be an arbitrary string up to 40 characters, and will override UserIP
// if both are provided.
func QuotaUser(u string) CallOption { return quotaUser(u) }

type quotaUser string

func (q quotaUser) Get() (string, string) { return "quotaUser", string(q) }

// UserIP returns a CallOption that will set the "userIp" parameter of a call.
// This should be the IP address of the originating request.
func UserIP(ip string) CallOption { return userIP(ip) }

type userIP string

func (i userIP) Get() (string, string) { return "userIp", string(i) }

// Trace returns a CallOption that enables diagnostic tracing for a call.
// traceToken is an ID supplied by Google support.
func Trace(traceToken string) CallOption { return traceTok(traceToken) }

type traceTok string

func (t traceTok) Get() (string, string) { return "trace", "token:" + string(t) }

// TODO: Fields too
