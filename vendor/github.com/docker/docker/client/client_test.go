package client

import (
	"bytes"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/docker/docker/api"
	"github.com/docker/docker/api/types"
	"github.com/stretchr/testify/assert"
)

func TestNewEnvClient(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping unix only test for windows")
	}
	cases := []struct {
		envs            map[string]string
		expectedError   string
		expectedVersion string
	}{
		{
			envs:            map[string]string{},
			expectedVersion: api.DefaultVersion,
		},
		{
			envs: map[string]string{
				"DOCKER_CERT_PATH": "invalid/path",
			},
			expectedError: "Could not load X509 key pair: open invalid/path/cert.pem: no such file or directory",
		},
		{
			envs: map[string]string{
				"DOCKER_CERT_PATH": "testdata/",
			},
			expectedVersion: api.DefaultVersion,
		},
		{
			envs: map[string]string{
				"DOCKER_CERT_PATH":  "testdata/",
				"DOCKER_TLS_VERIFY": "1",
			},
			expectedVersion: api.DefaultVersion,
		},
		{
			envs: map[string]string{
				"DOCKER_CERT_PATH": "testdata/",
				"DOCKER_HOST":      "https://notaunixsocket",
			},
			expectedVersion: api.DefaultVersion,
		},
		{
			envs: map[string]string{
				"DOCKER_HOST": "host",
			},
			expectedError: "unable to parse docker host `host`",
		},
		{
			envs: map[string]string{
				"DOCKER_HOST": "invalid://url",
			},
			expectedVersion: api.DefaultVersion,
		},
		{
			envs: map[string]string{
				"DOCKER_API_VERSION": "anything",
			},
			expectedVersion: "anything",
		},
		{
			envs: map[string]string{
				"DOCKER_API_VERSION": "1.22",
			},
			expectedVersion: "1.22",
		},
	}

	env := envToMap()
	defer mapToEnv(env)
	for _, c := range cases {
		mapToEnv(env)
		mapToEnv(c.envs)
		apiclient, err := NewEnvClient()
		if c.expectedError != "" {
			assert.Error(t, err)
			assert.Equal(t, c.expectedError, err.Error())
		} else {
			assert.NoError(t, err)
			version := apiclient.ClientVersion()
			assert.Equal(t, c.expectedVersion, version)
		}

		if c.envs["DOCKER_TLS_VERIFY"] != "" {
			// pedantic checking that this is handled correctly
			tr := apiclient.client.Transport.(*http.Transport)
			assert.NotNil(t, tr.TLSClientConfig)
			assert.Equal(t, tr.TLSClientConfig.InsecureSkipVerify, false)
		}
	}
}

func TestGetAPIPath(t *testing.T) {
	cases := []struct {
		v string
		p string
		q url.Values
		e string
	}{
		{"", "/containers/json", nil, "/containers/json"},
		{"", "/containers/json", url.Values{}, "/containers/json"},
		{"", "/containers/json", url.Values{"s": []string{"c"}}, "/containers/json?s=c"},
		{"1.22", "/containers/json", nil, "/v1.22/containers/json"},
		{"1.22", "/containers/json", url.Values{}, "/v1.22/containers/json"},
		{"1.22", "/containers/json", url.Values{"s": []string{"c"}}, "/v1.22/containers/json?s=c"},
		{"v1.22", "/containers/json", nil, "/v1.22/containers/json"},
		{"v1.22", "/containers/json", url.Values{}, "/v1.22/containers/json"},
		{"v1.22", "/containers/json", url.Values{"s": []string{"c"}}, "/v1.22/containers/json?s=c"},
		{"v1.22", "/networks/kiwl$%^", nil, "/v1.22/networks/kiwl$%25%5E"},
	}

	for _, cs := range cases {
		c, err := NewClient("unix:///var/run/docker.sock", cs.v, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		g := c.getAPIPath(cs.p, cs.q)
		assert.Equal(t, g, cs.e)

		err = c.Close()
		assert.NoError(t, err)
	}
}

func TestParseHost(t *testing.T) {
	cases := []struct {
		host  string
		proto string
		addr  string
		base  string
		err   bool
	}{
		{"", "", "", "", true},
		{"foobar", "", "", "", true},
		{"foo://bar", "foo", "bar", "", false},
		{"tcp://localhost:2476", "tcp", "localhost:2476", "", false},
		{"tcp://localhost:2476/path", "tcp", "localhost:2476", "/path", false},
	}

	for _, cs := range cases {
		p, a, b, e := ParseHost(cs.host)
		// if we expected an error to be returned...
		if cs.err {
			assert.Error(t, e)
		}
		assert.Equal(t, cs.proto, p)
		assert.Equal(t, cs.addr, a)
		assert.Equal(t, cs.base, b)
	}
}

func TestNewEnvClientSetsDefaultVersion(t *testing.T) {
	env := envToMap()
	defer mapToEnv(env)

	envMap := map[string]string{
		"DOCKER_HOST":        "",
		"DOCKER_API_VERSION": "",
		"DOCKER_TLS_VERIFY":  "",
		"DOCKER_CERT_PATH":   "",
	}
	mapToEnv(envMap)

	client, err := NewEnvClient()
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, client.version, api.DefaultVersion)

	expected := "1.22"
	os.Setenv("DOCKER_API_VERSION", expected)
	client, err = NewEnvClient()
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, expected, client.version)
}

// TestNegotiateAPIVersionEmpty asserts that client.Client can
// negotiate a compatible APIVersion when omitted
func TestNegotiateAPIVersionEmpty(t *testing.T) {
	env := envToMap()
	defer mapToEnv(env)

	envMap := map[string]string{
		"DOCKER_API_VERSION": "",
	}
	mapToEnv(envMap)

	client, err := NewEnvClient()
	if err != nil {
		t.Fatal(err)
	}

	ping := types.Ping{
		APIVersion:   "",
		OSType:       "linux",
		Experimental: false,
	}

	// set our version to something new
	client.version = "1.25"

	// if no version from server, expect the earliest
	// version before APIVersion was implemented
	expected := "1.24"

	// test downgrade
	client.NegotiateAPIVersionPing(ping)
	assert.Equal(t, expected, client.version)
}

// TestNegotiateAPIVersion asserts that client.Client can
// negotiate a compatible APIVersion with the server
func TestNegotiateAPIVersion(t *testing.T) {
	client, err := NewEnvClient()
	if err != nil {
		t.Fatal(err)
	}

	expected := "1.21"

	ping := types.Ping{
		APIVersion:   expected,
		OSType:       "linux",
		Experimental: false,
	}

	// set our version to something new
	client.version = "1.22"

	// test downgrade
	client.NegotiateAPIVersionPing(ping)
	assert.Equal(t, expected, client.version)
}

// TestNegotiateAPIVersionOverride asserts that we honor
// the environment variable DOCKER_API_VERSION when negotianing versions
func TestNegotiateAPVersionOverride(t *testing.T) {
	env := envToMap()
	defer mapToEnv(env)

	envMap := map[string]string{
		"DOCKER_API_VERSION": "9.99",
	}
	mapToEnv(envMap)

	client, err := NewEnvClient()
	if err != nil {
		t.Fatal(err)
	}

	ping := types.Ping{
		APIVersion:   "1.24",
		OSType:       "linux",
		Experimental: false,
	}

	expected := envMap["DOCKER_API_VERSION"]

	// test that we honored the env var
	client.NegotiateAPIVersionPing(ping)
	assert.Equal(t, expected, client.version)
}

// mapToEnv takes a map of environment variables and sets them
func mapToEnv(env map[string]string) {
	for k, v := range env {
		os.Setenv(k, v)
	}
}

// envToMap returns a map of environment variables
func envToMap() map[string]string {
	env := make(map[string]string)
	for _, e := range os.Environ() {
		kv := strings.SplitAfterN(e, "=", 2)
		env[kv[0]] = kv[1]
	}

	return env
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (rtf roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return rtf(req)
}

type bytesBufferClose struct {
	*bytes.Buffer
}

func (bbc bytesBufferClose) Close() error {
	return nil
}

func TestClientRedirect(t *testing.T) {
	client := &http.Client{
		CheckRedirect: CheckRedirect,
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			if req.URL.String() == "/bla" {
				return &http.Response{StatusCode: 404}, nil
			}
			return &http.Response{
				StatusCode: 301,
				Header:     map[string][]string{"Location": {"/bla"}},
				Body:       bytesBufferClose{bytes.NewBuffer(nil)},
			}, nil
		}),
	}

	cases := []struct {
		httpMethod  string
		expectedErr error
		statusCode  int
	}{
		{http.MethodGet, nil, 301},
		{http.MethodPost, &url.Error{Op: "Post", URL: "/bla", Err: ErrRedirect}, 301},
		{http.MethodPut, &url.Error{Op: "Put", URL: "/bla", Err: ErrRedirect}, 301},
		{http.MethodDelete, &url.Error{Op: "Delete", URL: "/bla", Err: ErrRedirect}, 301},
	}

	for _, tc := range cases {
		req, err := http.NewRequest(tc.httpMethod, "/redirectme", nil)
		assert.NoError(t, err)
		resp, err := client.Do(req)
		assert.Equal(t, tc.expectedErr, err)
		assert.Equal(t, tc.statusCode, resp.StatusCode)
	}
}
