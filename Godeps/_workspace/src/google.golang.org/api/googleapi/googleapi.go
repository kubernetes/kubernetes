// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package googleapi contains the common code shared by all Google API
// libraries.
package googleapi // import "google.golang.org/api/googleapi"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/context/ctxhttp"
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

	// statusResumeIncomplete is the code returned by the Google uploader when the transfer is not yet complete.
	statusResumeIncomplete = 308

	// UserAgent is the header string used to identify this package.
	UserAgent = "google-api-go-client/" + Version

	// uploadPause determines the delay between failed upload attempts
	uploadPause = 1 * time.Second
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
func CheckMediaResponse(res *http.Response) error {
	if res.StatusCode >= 200 && res.StatusCode <= 299 {
		return nil
	}
	slurp, _ := ioutil.ReadAll(io.LimitReader(res.Body, 1<<20))
	res.Body.Close()
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

func getMediaType(media io.Reader) (io.Reader, string) {
	if typer, ok := media.(ContentTyper); ok {
		return media, typer.ContentType()
	}

	pr, pw := io.Pipe()
	typ := "application/octet-stream"
	buf, err := ioutil.ReadAll(io.LimitReader(media, 512))
	if err != nil {
		pw.CloseWithError(fmt.Errorf("error reading media: %v", err))
		return pr, typ
	}
	typ = http.DetectContentType(buf)
	mr := io.MultiReader(bytes.NewReader(buf), media)
	go func() {
		_, err = io.Copy(pw, mr)
		if err != nil {
			pw.CloseWithError(fmt.Errorf("error reading media: %v", err))
			return
		}
		pw.Close()
	}()
	return pr, typ
}

// DetectMediaType detects and returns the content type of the provided media.
// If the type can not be determined, "application/octet-stream" is returned.
func DetectMediaType(media io.ReaderAt) string {
	if typer, ok := media.(ContentTyper); ok {
		return typer.ContentType()
	}

	typ := "application/octet-stream"
	buf := make([]byte, 1024)
	n, err := media.ReadAt(buf, 0)
	buf = buf[:n]
	if err == nil || err == io.EOF {
		typ = http.DetectContentType(buf)
	}
	return typ
}

type Lengther interface {
	Len() int
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

func typeHeader(contentType string) textproto.MIMEHeader {
	h := make(textproto.MIMEHeader)
	h.Set("Content-Type", contentType)
	return h
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

// ConditionallyIncludeMedia does nothing if media is nil.
//
// bodyp is an in/out parameter.  It should initially point to the
// reader of the application/json (or whatever) payload to send in the
// API request.  It's updated to point to the multipart body reader.
//
// ctypep is an in/out parameter.  It should initially point to the
// content type of the bodyp, usually "application/json".  It's updated
// to the "multipart/related" content type, with random boundary.
//
// The return value is the content-length of the entire multpart body.
func ConditionallyIncludeMedia(media io.Reader, bodyp *io.Reader, ctypep *string) (cancel func(), ok bool) {
	if media == nil {
		return
	}
	// Get the media type, which might return a different reader instance.
	var mediaType string
	media, mediaType = getMediaType(media)

	body, bodyType := *bodyp, *ctypep

	pr, pw := io.Pipe()
	mpw := multipart.NewWriter(pw)
	*bodyp = pr
	*ctypep = "multipart/related; boundary=" + mpw.Boundary()
	go func() {
		w, err := mpw.CreatePart(typeHeader(bodyType))
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: body CreatePart failed: %v", err))
			return
		}
		_, err = io.Copy(w, body)
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: body Copy failed: %v", err))
			return
		}

		w, err = mpw.CreatePart(typeHeader(mediaType))
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: media CreatePart failed: %v", err))
			return
		}
		_, err = io.Copy(w, media)
		if err != nil {
			mpw.Close()
			pw.CloseWithError(fmt.Errorf("googleapi: media Copy failed: %v", err))
			return
		}
		mpw.Close()
		pw.Close()
	}()
	cancel = func() { pw.CloseWithError(errAborted) }
	return cancel, true
}

var errAborted = errors.New("googleapi: upload aborted")

// ProgressUpdater is a function that is called upon every progress update of a resumable upload.
// This is the only part of a resumable upload (from googleapi) that is usable by the developer.
// The remaining usable pieces of resumable uploads is exposed in each auto-generated API.
type ProgressUpdater func(current, total int64)

// ResumableUpload is used by the generated APIs to provide resumable uploads.
// It is not used by developers directly.
type ResumableUpload struct {
	Client *http.Client
	// URI is the resumable resource destination provided by the server after specifying "&uploadType=resumable".
	URI       string
	UserAgent string // User-Agent for header of the request
	// Media is the object being uploaded.
	Media io.ReaderAt
	// MediaType defines the media type, e.g. "image/jpeg".
	MediaType string
	// ContentLength is the full size of the object being uploaded.
	ContentLength int64

	mu       sync.Mutex // guards progress
	progress int64      // number of bytes uploaded so far
	started  bool       // whether the upload has been started

	// Callback is an optional function that will be called upon every progress update.
	Callback ProgressUpdater
}

var (
	// rangeRE matches the transfer status response from the server. $1 is the last byte index uploaded.
	rangeRE = regexp.MustCompile(`^bytes=0\-(\d+)$`)
	// chunkSize is the size of the chunks created during a resumable upload and should be a power of two.
	// 1<<18 is the minimum size supported by the Google uploader, and there is no maximum.
	chunkSize int64 = 1 << 18
)

// Progress returns the number of bytes uploaded at this point.
func (rx *ResumableUpload) Progress() int64 {
	rx.mu.Lock()
	defer rx.mu.Unlock()
	return rx.progress
}

func (rx *ResumableUpload) transferStatus(ctx context.Context) (int64, *http.Response, error) {
	req, _ := http.NewRequest("POST", rx.URI, nil)
	req.ContentLength = 0
	req.Header.Set("User-Agent", rx.UserAgent)
	req.Header.Set("Content-Range", fmt.Sprintf("bytes */%v", rx.ContentLength))
	res, err := ctxhttp.Do(ctx, rx.Client, req)
	if err != nil || res.StatusCode != statusResumeIncomplete {
		return 0, res, err
	}
	var start int64
	if m := rangeRE.FindStringSubmatch(res.Header.Get("Range")); len(m) == 2 {
		start, err = strconv.ParseInt(m[1], 10, 64)
		if err != nil {
			return 0, nil, fmt.Errorf("unable to parse range size %v", m[1])
		}
		start += 1 // Start at the next byte
	}
	return start, res, nil
}

type chunk struct {
	body io.Reader
	size int64
	err  error
}

func (rx *ResumableUpload) transferChunks(ctx context.Context) (*http.Response, error) {
	var start int64
	var err error
	res := &http.Response{}
	if rx.started {
		start, res, err = rx.transferStatus(ctx)
		if err != nil || res.StatusCode != statusResumeIncomplete {
			return res, err
		}
	}
	rx.started = true

	for {
		select { // Check for cancellation
		case <-ctx.Done():
			res.StatusCode = http.StatusRequestTimeout
			return res, ctx.Err()
		default:
		}
		reqSize := rx.ContentLength - start
		if reqSize > chunkSize {
			reqSize = chunkSize
		}
		r := io.NewSectionReader(rx.Media, start, reqSize)
		req, _ := http.NewRequest("POST", rx.URI, r)
		req.ContentLength = reqSize
		req.Header.Set("Content-Range", fmt.Sprintf("bytes %v-%v/%v", start, start+reqSize-1, rx.ContentLength))
		req.Header.Set("Content-Type", rx.MediaType)
		req.Header.Set("User-Agent", rx.UserAgent)
		res, err = ctxhttp.Do(ctx, rx.Client, req)
		start += reqSize
		if err == nil && (res.StatusCode == statusResumeIncomplete || res.StatusCode == http.StatusOK) {
			rx.mu.Lock()
			rx.progress = start // keep track of number of bytes sent so far
			rx.mu.Unlock()
			if rx.Callback != nil {
				rx.Callback(start, rx.ContentLength)
			}
		}
		if err != nil || res.StatusCode != statusResumeIncomplete {
			break
		}
	}
	return res, err
}

var sleep = time.Sleep // override in unit tests

// Upload starts the process of a resumable upload with a cancellable context.
// It retries indefinitely (with a pause of uploadPause between attempts) until cancelled.
// It is called from the auto-generated API code and is not visible to the user.
// rx is private to the auto-generated API code.
func (rx *ResumableUpload) Upload(ctx context.Context) (*http.Response, error) {
	var res *http.Response
	var err error
	for {
		res, err = rx.transferChunks(ctx)
		if err != nil || res.StatusCode == http.StatusCreated || res.StatusCode == http.StatusOK {
			return res, err
		}
		select { // Check for cancellation
		case <-ctx.Done():
			res.StatusCode = http.StatusRequestTimeout
			return res, ctx.Err()
		default:
		}
		sleep(uploadPause)
	}
	return res, err
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

// has4860Fix is whether this Go environment contains the fix for
// http://golang.org/issue/4860
var has4860Fix bool

// init initializes has4860Fix by checking the behavior of the net/http package.
func init() {
	r := http.Request{
		URL: &url.URL{
			Scheme: "http",
			Opaque: "//opaque",
		},
	}
	b := &bytes.Buffer{}
	r.Write(b)
	has4860Fix = bytes.HasPrefix(b.Bytes(), []byte("GET http"))
}

// SetOpaque sets u.Opaque from u.Path such that HTTP requests to it
// don't alter any hex-escaped characters in u.Path.
func SetOpaque(u *url.URL) {
	u.Opaque = "//" + u.Host + u.Path
	if !has4860Fix {
		u.Opaque = u.Scheme + ":" + u.Opaque
	}
}

// Expand subsitutes any {encoded} strings in the URL passed in using
// the map supplied.
//
// This calls SetOpaque to avoid encoding of the parameters in the URL path.
func Expand(u *url.URL, expansions map[string]string) {
	expanded, err := uritemplates.Expand(u.Path, expansions)
	if err == nil {
		u.Path = expanded
		SetOpaque(u)
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
