package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
	"math/rand"
)

// Request is contructed iteratively by the client and finally dispatched.
// A REST endpoint is accessed with the following convention:
// base_url/<version>/<resource>/[<instance>]
type Request struct {
	client   *http.Client
	version  string
	verb     string
	path     string
	base     *url.URL
	params   url.Values
	headers  http.Header
	resource string
	instance string
	err      error
	body     []byte
	req      *http.Request
	resp     *http.Response
	timeout  time.Duration
	authstring string
	accesstoken string
}

// Response is a representation of HTTP response received from the server.
type Response struct {
	status     string
	statusCode int
	err        error
	body       []byte
}

// Status upon error, attempts to parse the body of a response into a meaningful status.
type Status struct {
	Message   string
	ErrorCode int
}

// NewRequest instance
func NewRequest(client *http.Client, base *url.URL, verb string, version string, authstring, userAgent string) *Request {
	r := &Request{
		client:  client,
		verb:    verb,
		base:    base,
		path:    base.Path,
		version: version,
		authstring: authstring,
	}
	r.SetHeader("User-Agent", userAgent)
	return r
}

func checkExists(mustExist string, before string) error {
	if len(mustExist) == 0 {
		return fmt.Errorf("%q should be set before setting %q", mustExist, before)
	}
	return nil
}

func checkSet(name string, s *string, newval string) error {
	if len(*s) != 0 {
		return fmt.Errorf("%q already set to %q, cannot change to %q",
			name, *s, newval)
	}
	*s = newval
	return nil
}

// Resource specifies the resource to be accessed.
func (r *Request) Resource(resource string) *Request {
	if r.err == nil {
		r.err = checkSet("resource", &r.resource, resource)
	}
	return r
}

// Instance specifies the instance of the resource to be accessed.
func (r *Request) Instance(instance string) *Request {
	if r.err == nil {
		r.err = checkExists("resource", "instance")
		if r.err == nil {
			r.err = checkSet("instance", &r.instance, instance)
		}
	}
	return r
}

// UsePath use the specified path and don't build up a request.
func (r *Request) UsePath(path string) *Request {
	if r.err == nil {
		r.err = checkSet("path", &r.path, path)
	}
	return r
}

// QueryOption adds specified options to query.
func (r *Request) QueryOption(key string, value string) *Request {
	if r.err != nil {
		return r
	}
	if r.params == nil {
		r.params = make(url.Values)
	}
	r.params.Add(string(key), value)
	return r
}

// QueryOptionLabel adds specified label to query.
func (r *Request) QueryOptionLabel(key string, labels map[string]string) *Request {
	if r.err != nil {
		return r
	}
	if b, err := json.Marshal(labels); err != nil {
		r.err = err
	} else {
		if r.params == nil {
			r.params = make(url.Values)
		}
		r.params.Add(string(key), string(b))
	}
	return r
}

// SetHeader adds specified header values to query.
func (r *Request) SetHeader(key, value string) *Request {
	if r.headers == nil {
		r.headers = http.Header{}
	}
	r.headers.Set(key, value)
	return r
}

// Timeout makes the request use the given duration as a timeout. Sets the "timeout"
// parameter.
func (r *Request) Timeout(d time.Duration) *Request {
	if r.err != nil {
		return r
	}
	r.timeout = d
	return r
}

// Body sets the request Body.
func (r *Request) Body(v interface{}) *Request {
	var err error
	if r.err != nil {
		return r
	}
	r.body, err = json.Marshal(v)
	if err != nil {
		r.err = err
		return r
	}
	return r
}

// URL returns the current working URL.
func (r *Request) URL() *url.URL {
	u := *r.base
	p := r.path

	if len(r.version) != 0 {
		p = path.Join(p, strings.ToLower(r.version))
	}
	if len(r.resource) != 0 {
		p = path.Join(p, strings.ToLower(r.resource))
		if len(r.instance) != 0 {
			p = path.Join(p, r.instance)
		}
	}

	u.Path = p

	query := url.Values{}
	for key, values := range r.params {
		for _, value := range values {
			query.Add(key, value)
		}
	}
	if r.timeout != 0 {
		query.Set("timeout", r.timeout.String())
	}
	u.RawQuery = query.Encode()
	return &u
}

// headerVal for key as an int. Return false if header is not present or valid.
func headerVal(key string, resp *http.Response) (int, bool) {
	if h := resp.Header.Get(key); len(h) > 0 {
		if i, err := strconv.Atoi(h); err == nil {
			return i, true
		}
	}
	return 0, false
}

func parseHTTPStatus(resp *http.Response, body []byte) error {

	var (
		status *Status
		err    error
	)

	httpOK := resp.StatusCode >= http.StatusOK && resp.StatusCode <= http.StatusPartialContent
	hasStatus := false
	if body != nil {
		err = json.Unmarshal(body, status)
		if err == nil && status.Message != "" {
			hasStatus = true
		}
	}
	// If the status is NG, return an error regardless of HTTP status.
	if hasStatus && status.ErrorCode != 0 {
		return fmt.Errorf("Error %v : %v", status.ErrorCode, status.Message)
	}

	// Status is good and HTTP status is good, everything is good
	if httpOK {
		return nil
	}

	// If HTTP status is NG, return an error.
	return fmt.Errorf("HTTP error %d", resp.StatusCode)
}

// Do executes the request and returns a Response.
func (r *Request) Do() *Response {
	var (
		err  error
		req  *http.Request
		resp *http.Response
		url  string
		body []byte
	)
	if r.err != nil {
		return &Response{err: r.err}
	}
	url = r.URL().String()
	req, err = http.NewRequest(r.verb, url, bytes.NewBuffer(r.body))
	if err != nil {
		return &Response{err: err}
	}
	if r.headers == nil {
		r.headers = http.Header{}
	}

	req.Header = r.headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Date", time.Now().String())

	if len(r.authstring) > 0 {
		req.Header.Set("Authorization", "Basic "+ r.authstring)
	}

	if len(r.accesstoken) > 0 {
		req.Header.Set("Access-Token", r.accesstoken)
	}

	resp, err = r.client.Do(req)
	if err != nil {
		return &Response{err: err}
	}
	if resp.Body != nil {
		defer resp.Body.Close()
		body, err = ioutil.ReadAll(resp.Body)
	}
	if err != nil {
		return &Response{err: err}
	}
	return &Response{
		status:     resp.Status,
		statusCode: resp.StatusCode,
		body:       body,
		err:        parseHTTPStatus(resp, body),
	}
}

// Body return http body, valid only if there is no error
func (r Response) Body() ([]byte, error) {
	return r.body, r.err
}

// StatusCode HTTP status code returned.
func (r Response) StatusCode() int {
	return r.statusCode
}

// Unmarshal result into obj
func (r Response) Unmarshal(v interface{}) error {
	if r.err != nil {
		return r.err
	}
	return json.Unmarshal(r.body, v)
}

// Error executing the request.
func (r Response) Error() error {
	return r.err
}

// FormatError formats the error
func (r Response) FormatError() error {
	if len(r.body) == 0 {
		return fmt.Errorf("Error: %v", r.err)
	}
	return fmt.Errorf("HTTP-%d: %s", r.statusCode, string(r.body))
}

func digest(method string, path string) string {
	now := time.Now().String()

	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)

	nonce := r1.Intn(10)

	return method + "+" + path + "+" + now + "+" + strconv.Itoa(nonce)
}