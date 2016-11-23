package godo

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"
)

var (
	mux *http.ServeMux

	client *Client

	server *httptest.Server
)

func setup() {
	mux = http.NewServeMux()
	server = httptest.NewServer(mux)

	client = NewClient(nil)
	url, _ := url.Parse(server.URL)
	client.BaseURL = url
}

func teardown() {
	server.Close()
}

func testMethod(t *testing.T, r *http.Request, expected string) {
	if expected != r.Method {
		t.Errorf("Request method = %v, expected %v", r.Method, expected)
	}
}

type values map[string]string

func testFormValues(t *testing.T, r *http.Request, values values) {
	expected := url.Values{}
	for k, v := range values {
		expected.Add(k, v)
	}

	err := r.ParseForm()
	if err != nil {
		t.Fatalf("parseForm(): %v", err)
	}

	if !reflect.DeepEqual(expected, r.Form) {
		t.Errorf("Request parameters = %v, expected %v", r.Form, expected)
	}
}

func testURLParseError(t *testing.T, err error) {
	if err == nil {
		t.Errorf("Expected error to be returned")
	}
	if err, ok := err.(*url.Error); !ok || err.Op != "parse" {
		t.Errorf("Expected URL parse error, got %+v", err)
	}
}

func testClientServices(t *testing.T, c *Client) {
	services := []string{
		"Account",
		"Actions",
		"Domains",
		"Droplets",
		"DropletActions",
		"Images",
		"ImageActions",
		"Keys",
		"Regions",
		"Sizes",
		"FloatingIPs",
		"FloatingIPActions",
		"Tags",
	}

	cp := reflect.ValueOf(c)
	cv := reflect.Indirect(cp)

	for _, s := range services {
		if cv.FieldByName(s).IsNil() {
			t.Errorf("c.%s shouldn't be nil", s)
		}
	}
}

func testClientDefaultBaseURL(t *testing.T, c *Client) {
	if c.BaseURL == nil || c.BaseURL.String() != defaultBaseURL {
		t.Errorf("NewClient BaseURL = %v, expected %v", c.BaseURL, defaultBaseURL)
	}
}

func testClientDefaultUserAgent(t *testing.T, c *Client) {
	if c.UserAgent != userAgent {
		t.Errorf("NewClick UserAgent = %v, expected %v", c.UserAgent, userAgent)
	}
}

func testClientDefaults(t *testing.T, c *Client) {
	testClientDefaultBaseURL(t, c)
	testClientDefaultUserAgent(t, c)
	testClientServices(t, c)
}

func TestNewClient(t *testing.T) {
	c := NewClient(nil)
	testClientDefaults(t, c)
}

func TestNew(t *testing.T) {
	c, err := New(nil)

	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	testClientDefaults(t, c)
}

func TestNewRequest(t *testing.T) {
	c := NewClient(nil)

	inURL, outURL := "/foo", defaultBaseURL+"foo"
	inBody, outBody := &DropletCreateRequest{Name: "l"},
		`{"name":"l","region":"","size":"","image":0,`+
			`"ssh_keys":null,"backups":false,"ipv6":false,`+
			`"private_networking":false,"tags":null}`+"\n"
	req, _ := c.NewRequest("GET", inURL, inBody)

	// test relative URL was expanded
	if req.URL.String() != outURL {
		t.Errorf("NewRequest(%v) URL = %v, expected %v", inURL, req.URL, outURL)
	}

	// test body was JSON encoded
	body, _ := ioutil.ReadAll(req.Body)
	if string(body) != outBody {
		t.Errorf("NewRequest(%v)Body = %v, expected %v", inBody, string(body), outBody)
	}

	// test default user-agent is attached to the request
	userAgent := req.Header.Get("User-Agent")
	if c.UserAgent != userAgent {
		t.Errorf("NewRequest() User-Agent = %v, expected %v", userAgent, c.UserAgent)
	}
}

func TestNewRequest_withUserData(t *testing.T) {
	c := NewClient(nil)

	inURL, outURL := "/foo", defaultBaseURL+"foo"
	inBody, outBody := &DropletCreateRequest{Name: "l", UserData: "u"},
		`{"name":"l","region":"","size":"","image":0,`+
			`"ssh_keys":null,"backups":false,"ipv6":false,`+
			`"private_networking":false,"user_data":"u","tags":null}`+"\n"
	req, _ := c.NewRequest("GET", inURL, inBody)

	// test relative URL was expanded
	if req.URL.String() != outURL {
		t.Errorf("NewRequest(%v) URL = %v, expected %v", inURL, req.URL, outURL)
	}

	// test body was JSON encoded
	body, _ := ioutil.ReadAll(req.Body)
	if string(body) != outBody {
		t.Errorf("NewRequest(%v)Body = %v, expected %v", inBody, string(body), outBody)
	}

	// test default user-agent is attached to the request
	userAgent := req.Header.Get("User-Agent")
	if c.UserAgent != userAgent {
		t.Errorf("NewRequest() User-Agent = %v, expected %v", userAgent, c.UserAgent)
	}
}

func TestNewRequest_badURL(t *testing.T) {
	c := NewClient(nil)
	_, err := c.NewRequest("GET", ":", nil)
	testURLParseError(t, err)
}

func TestNewRequest_withCustomUserAgent(t *testing.T) {
	ua := "testing"
	c, err := New(nil, SetUserAgent(ua))

	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	req, _ := c.NewRequest("GET", "/foo", nil)

	expected := fmt.Sprintf("%s+%s", ua, userAgent)
	if got := req.Header.Get("User-Agent"); got != expected {
		t.Errorf("New() UserAgent = %s; expected %s", got, expected)
	}
}

func TestDo(t *testing.T) {
	setup()
	defer teardown()

	type foo struct {
		A string
	}

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if m := "GET"; m != r.Method {
			t.Errorf("Request method = %v, expected %v", r.Method, m)
		}
		fmt.Fprint(w, `{"A":"a"}`)
	})

	req, _ := client.NewRequest("GET", "/", nil)
	body := new(foo)
	_, err := client.Do(req, body)
	if err != nil {
		t.Fatalf("Do(): %v", err)
	}

	expected := &foo{"a"}
	if !reflect.DeepEqual(body, expected) {
		t.Errorf("Response body = %v, expected %v", body, expected)
	}
}

func TestDo_httpError(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Bad Request", 400)
	})

	req, _ := client.NewRequest("GET", "/", nil)
	_, err := client.Do(req, nil)

	if err == nil {
		t.Error("Expected HTTP 400 error.")
	}
}

// Test handling of an error caused by the internal http client's Do()
// function.
func TestDo_redirectLoop(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/", http.StatusFound)
	})

	req, _ := client.NewRequest("GET", "/", nil)
	_, err := client.Do(req, nil)

	if err == nil {
		t.Error("Expected error to be returned.")
	}
	if err, ok := err.(*url.Error); !ok {
		t.Errorf("Expected a URL error; got %#v.", err)
	}
}

func TestCheckResponse(t *testing.T) {
	res := &http.Response{
		Request:    &http.Request{},
		StatusCode: http.StatusBadRequest,
		Body: ioutil.NopCloser(strings.NewReader(`{"message":"m",
			"errors": [{"resource": "r", "field": "f", "code": "c"}]}`)),
	}
	err := CheckResponse(res).(*ErrorResponse)

	if err == nil {
		t.Fatalf("Expected error response.")
	}

	expected := &ErrorResponse{
		Response: res,
		Message:  "m",
	}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("Error = %#v, expected %#v", err, expected)
	}
}

// ensure that we properly handle API errors that do not contain a response
// body
func TestCheckResponse_noBody(t *testing.T) {
	res := &http.Response{
		Request:    &http.Request{},
		StatusCode: http.StatusBadRequest,
		Body:       ioutil.NopCloser(strings.NewReader("")),
	}
	err := CheckResponse(res).(*ErrorResponse)

	if err == nil {
		t.Errorf("Expected error response.")
	}

	expected := &ErrorResponse{
		Response: res,
	}
	if !reflect.DeepEqual(err, expected) {
		t.Errorf("Error = %#v, expected %#v", err, expected)
	}
}

func TestErrorResponse_Error(t *testing.T) {
	res := &http.Response{Request: &http.Request{}}
	err := ErrorResponse{Message: "m", Response: res}
	if err.Error() == "" {
		t.Errorf("Expected non-empty ErrorResponse.Error()")
	}
}

func TestDo_rateLimit(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add(headerRateLimit, "60")
		w.Header().Add(headerRateRemaining, "59")
		w.Header().Add(headerRateReset, "1372700873")
	})

	var expected int

	if expected = 0; client.Rate.Limit != expected {
		t.Errorf("Client rate limit = %v, expected %v", client.Rate.Limit, expected)
	}
	if expected = 0; client.Rate.Remaining != expected {
		t.Errorf("Client rate remaining = %v, got %v", client.Rate.Remaining, expected)
	}
	if !client.Rate.Reset.IsZero() {
		t.Errorf("Client rate reset not initialized to zero value")
	}

	req, _ := client.NewRequest("GET", "/", nil)
	_, err := client.Do(req, nil)
	if err != nil {
		t.Fatalf("Do(): %v", err)
	}

	if expected = 60; client.Rate.Limit != expected {
		t.Errorf("Client rate limit = %v, expected %v", client.Rate.Limit, expected)
	}
	if expected = 59; client.Rate.Remaining != expected {
		t.Errorf("Client rate remaining = %v, expected %v", client.Rate.Remaining, expected)
	}
	reset := time.Date(2013, 7, 1, 17, 47, 53, 0, time.UTC)
	if client.Rate.Reset.UTC() != reset {
		t.Errorf("Client rate reset = %v, expected %v", client.Rate.Reset, reset)
	}
}

func TestDo_rateLimit_errorResponse(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add(headerRateLimit, "60")
		w.Header().Add(headerRateRemaining, "59")
		w.Header().Add(headerRateReset, "1372700873")
		http.Error(w, `{"message":"bad request"}`, 400)
	})

	var expected int

	req, _ := client.NewRequest("GET", "/", nil)
	_, _ = client.Do(req, nil)

	if expected = 60; client.Rate.Limit != expected {
		t.Errorf("Client rate limit = %v, expected %v", client.Rate.Limit, expected)
	}
	if expected = 59; client.Rate.Remaining != expected {
		t.Errorf("Client rate remaining = %v, expected %v", client.Rate.Remaining, expected)
	}
	reset := time.Date(2013, 7, 1, 17, 47, 53, 0, time.UTC)
	if client.Rate.Reset.UTC() != reset {
		t.Errorf("Client rate reset = %v, expected %v", client.Rate.Reset, reset)
	}
}

func checkCurrentPage(t *testing.T, resp *Response, expectedPage int) {
	links := resp.Links
	p, err := links.CurrentPage()
	if err != nil {
		t.Fatal(err)
	}

	if p != expectedPage {
		t.Fatalf("expected current page to be '%d', was '%d'", expectedPage, p)
	}
}

func TestDo_completion_callback(t *testing.T) {
	setup()
	defer teardown()

	type foo struct {
		A string
	}

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if m := "GET"; m != r.Method {
			t.Errorf("Request method = %v, expected %v", r.Method, m)
		}
		fmt.Fprint(w, `{"A":"a"}`)
	})

	req, _ := client.NewRequest("GET", "/", nil)
	body := new(foo)
	var completedReq *http.Request
	var completedResp string
	client.OnRequestCompleted(func(req *http.Request, resp *http.Response) {
		completedReq = req
		b, err := httputil.DumpResponse(resp, true)
		if err != nil {
			t.Errorf("Failed to dump response: %s", err)
		}
		completedResp = string(b)
	})
	_, err := client.Do(req, body)
	if err != nil {
		t.Fatalf("Do(): %v", err)
	}
	if !reflect.DeepEqual(req, completedReq) {
		t.Errorf("Completed request = %v, expected %v", completedReq, req)
	}
	expected := `{"A":"a"}`
	if !strings.Contains(completedResp, expected) {
		t.Errorf("expected response to contain %v, Response = %v", expected, completedResp)
	}
}

func TestAddOptions(t *testing.T) {
	cases := []struct {
		name     string
		path     string
		expected string
		opts     *ListOptions
		isErr    bool
	}{
		{
			name:     "add options",
			path:     "/action",
			expected: "/action?page=1",
			opts:     &ListOptions{Page: 1},
			isErr:    false,
		},
		{
			name:     "add options with existing parameters",
			path:     "/action?scope=all",
			expected: "/action?page=1&scope=all",
			opts:     &ListOptions{Page: 1},
			isErr:    false,
		},
	}

	for _, c := range cases {
		got, err := addOptions(c.path, c.opts)
		if c.isErr && err == nil {
			t.Errorf("%q expected error but none was encountered", c.name)
			continue
		}

		if !c.isErr && err != nil {
			t.Errorf("%q unexpected error: %v", c.name, err)
			continue
		}

		gotURL, err := url.Parse(got)
		if err != nil {
			t.Errorf("%q unable to parse returned URL", c.name)
			continue
		}

		expectedURL, err := url.Parse(c.expected)
		if err != nil {
			t.Errorf("%q unable to parse expected URL", c.name)
			continue
		}

		if g, e := gotURL.Path, expectedURL.Path; g != e {
			t.Errorf("%q path = %q; expected %q", c.name, g, e)
			continue
		}

		if g, e := gotURL.Query(), expectedURL.Query(); !reflect.DeepEqual(g, e) {
			t.Errorf("%q query = %#v; expected %#v", c.name, g, e)
			continue
		}
	}
}

func TestCustomUserAgent(t *testing.T) {
	c, err := New(nil, SetUserAgent("testing"))

	if err != nil {
		t.Fatalf("New() unexpected error: %v", err)
	}

	expected := fmt.Sprintf("%s+%s", "testing", userAgent)
	if got := c.UserAgent; got != expected {
		t.Errorf("New() UserAgent = %s; expected %s", got, expected)
	}
}
