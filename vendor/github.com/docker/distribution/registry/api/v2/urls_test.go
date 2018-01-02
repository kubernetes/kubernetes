package v2

import (
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/docker/distribution/reference"
)

type urlBuilderTestCase struct {
	description  string
	expectedPath string
	expectedErr  error
	build        func() (string, error)
}

func makeURLBuilderTestCases(urlBuilder *URLBuilder) []urlBuilderTestCase {
	fooBarRef, _ := reference.WithName("foo/bar")
	return []urlBuilderTestCase{
		{
			description:  "test base url",
			expectedPath: "/v2/",
			expectedErr:  nil,
			build:        urlBuilder.BuildBaseURL,
		},
		{
			description:  "test tags url",
			expectedPath: "/v2/foo/bar/tags/list",
			expectedErr:  nil,
			build: func() (string, error) {
				return urlBuilder.BuildTagsURL(fooBarRef)
			},
		},
		{
			description:  "test manifest url tagged ref",
			expectedPath: "/v2/foo/bar/manifests/tag",
			expectedErr:  nil,
			build: func() (string, error) {
				ref, _ := reference.WithTag(fooBarRef, "tag")
				return urlBuilder.BuildManifestURL(ref)
			},
		},
		{
			description:  "test manifest url bare ref",
			expectedPath: "",
			expectedErr:  fmt.Errorf("reference must have a tag or digest"),
			build: func() (string, error) {
				return urlBuilder.BuildManifestURL(fooBarRef)
			},
		},
		{
			description:  "build blob url",
			expectedPath: "/v2/foo/bar/blobs/sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5",
			expectedErr:  nil,
			build: func() (string, error) {
				ref, _ := reference.WithDigest(fooBarRef, "sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5")
				return urlBuilder.BuildBlobURL(ref)
			},
		},
		{
			description:  "build blob upload url",
			expectedPath: "/v2/foo/bar/blobs/uploads/",
			expectedErr:  nil,
			build: func() (string, error) {
				return urlBuilder.BuildBlobUploadURL(fooBarRef)
			},
		},
		{
			description:  "build blob upload url with digest and size",
			expectedPath: "/v2/foo/bar/blobs/uploads/?digest=sha256%3A3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5&size=10000",
			expectedErr:  nil,
			build: func() (string, error) {
				return urlBuilder.BuildBlobUploadURL(fooBarRef, url.Values{
					"size":   []string{"10000"},
					"digest": []string{"sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5"},
				})
			},
		},
		{
			description:  "build blob upload chunk url",
			expectedPath: "/v2/foo/bar/blobs/uploads/uuid-part",
			expectedErr:  nil,
			build: func() (string, error) {
				return urlBuilder.BuildBlobUploadChunkURL(fooBarRef, "uuid-part")
			},
		},
		{
			description:  "build blob upload chunk url with digest and size",
			expectedPath: "/v2/foo/bar/blobs/uploads/uuid-part?digest=sha256%3A3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5&size=10000",
			expectedErr:  nil,
			build: func() (string, error) {
				return urlBuilder.BuildBlobUploadChunkURL(fooBarRef, "uuid-part", url.Values{
					"size":   []string{"10000"},
					"digest": []string{"sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5"},
				})
			},
		},
	}
}

// TestURLBuilder tests the various url building functions, ensuring they are
// returning the expected values.
func TestURLBuilder(t *testing.T) {
	roots := []string{
		"http://example.com",
		"https://example.com",
		"http://localhost:5000",
		"https://localhost:5443",
	}

	doTest := func(relative bool) {
		for _, root := range roots {
			urlBuilder, err := NewURLBuilderFromString(root, relative)
			if err != nil {
				t.Fatalf("unexpected error creating urlbuilder: %v", err)
			}

			for _, testCase := range makeURLBuilderTestCases(urlBuilder) {
				url, err := testCase.build()
				expectedErr := testCase.expectedErr
				if !reflect.DeepEqual(expectedErr, err) {
					t.Fatalf("%s: Expecting %v but got error %v", testCase.description, expectedErr, err)
				}
				if expectedErr != nil {
					continue
				}

				expectedURL := testCase.expectedPath
				if !relative {
					expectedURL = root + expectedURL
				}

				if url != expectedURL {
					t.Fatalf("%s: %q != %q", testCase.description, url, expectedURL)
				}
			}
		}
	}
	doTest(true)
	doTest(false)
}

func TestURLBuilderWithPrefix(t *testing.T) {
	roots := []string{
		"http://example.com/prefix/",
		"https://example.com/prefix/",
		"http://localhost:5000/prefix/",
		"https://localhost:5443/prefix/",
	}

	doTest := func(relative bool) {
		for _, root := range roots {
			urlBuilder, err := NewURLBuilderFromString(root, relative)
			if err != nil {
				t.Fatalf("unexpected error creating urlbuilder: %v", err)
			}

			for _, testCase := range makeURLBuilderTestCases(urlBuilder) {
				url, err := testCase.build()
				expectedErr := testCase.expectedErr
				if !reflect.DeepEqual(expectedErr, err) {
					t.Fatalf("%s: Expecting %v but got error %v", testCase.description, expectedErr, err)
				}
				if expectedErr != nil {
					continue
				}

				expectedURL := testCase.expectedPath
				if !relative {
					expectedURL = root[0:len(root)-1] + expectedURL
				}
				if url != expectedURL {
					t.Fatalf("%s: %q != %q", testCase.description, url, expectedURL)
				}
			}
		}
	}
	doTest(true)
	doTest(false)
}

type builderFromRequestTestCase struct {
	request *http.Request
	base    string
}

func TestBuilderFromRequest(t *testing.T) {
	u, err := url.Parse("http://example.com")
	if err != nil {
		t.Fatal(err)
	}

	testRequests := []struct {
		name       string
		request    *http.Request
		base       string
		configHost url.URL
	}{
		{
			name:    "no forwarded header",
			request: &http.Request{URL: u, Host: u.Host},
			base:    "http://example.com",
		},
		{
			name: "https protocol forwarded with a non-standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Custom-Forwarded-Proto": []string{"https"},
			}},
			base: "http://example.com",
		},
		{
			name: "forwarded protocol is the same",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{"https"},
			}},
			base: "https://example.com",
		},
		{
			name: "forwarded host with a non-standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"first.example.com"},
			}},
			base: "http://first.example.com",
		},
		{
			name: "forwarded multiple hosts a with non-standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"first.example.com, proxy1.example.com"},
			}},
			base: "http://first.example.com",
		},
		{
			name: "host configured in config file takes priority",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"first.example.com, proxy1.example.com"},
			}},
			base: "https://third.example.com:5000",
			configHost: url.URL{
				Scheme: "https",
				Host:   "third.example.com:5000",
			},
		},
		{
			name: "forwarded host and port with just one non-standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"first.example.com:443"},
			}},
			base: "http://first.example.com:443",
		},
		{
			name: "forwarded port with a non-standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"example.com:5000"},
				"X-Forwarded-Port": []string{"5000"},
			}},
			base: "http://example.com:5000",
		},
		{
			name: "forwarded multiple ports with a non-standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Port": []string{"443 , 5001"},
			}},
			base: "http://example.com",
		},
		{
			name: "forwarded standard port with non-standard headers",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{"https"},
				"X-Forwarded-Host":  []string{"example.com"},
				"X-Forwarded-Port":  []string{"443"},
			}},
			base: "https://example.com",
		},
		{
			name: "forwarded standard port with non-standard headers and explicit port",
			request: &http.Request{URL: u, Host: u.Host + ":443", Header: http.Header{
				"X-Forwarded-Proto": []string{"https"},
				"X-Forwarded-Host":  []string{u.Host + ":443"},
				"X-Forwarded-Port":  []string{"443"},
			}},
			base: "https://example.com:443",
		},
		{
			name: "several non-standard headers",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{"https"},
				"X-Forwarded-Host":  []string{" first.example.com:12345 "},
			}},
			base: "https://first.example.com:12345",
		},
		{
			name: "forwarded host with port supplied takes priority",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"first.example.com:5000"},
				"X-Forwarded-Port": []string{"80"},
			}},
			base: "http://first.example.com:5000",
		},
		{
			name: "malformed forwarded port",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Host": []string{"first.example.com"},
				"X-Forwarded-Port": []string{"abcd"},
			}},
			base: "http://first.example.com",
		},
		{
			name: "forwarded protocol and addr using standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded": []string{`proto=https;host="192.168.22.30:80"`},
			}},
			base: "https://192.168.22.30:80",
		},
		{
			name: "forwarded host takes priority over for",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded": []string{`host="reg.example.com:5000";for="192.168.22.30"`},
			}},
			base: "http://reg.example.com:5000",
		},
		{
			name: "forwarded host and protocol using standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded": []string{`host=reg.example.com;proto=https`},
			}},
			base: "https://reg.example.com",
		},
		{
			name: "process just the first standard forwarded header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded": []string{`host="reg.example.com:88";proto=http`, `host=reg.example.com;proto=https`},
			}},
			base: "http://reg.example.com:88",
		},
		{
			name: "process just the first list element of standard header",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded": []string{`host="reg.example.com:443";proto=https, host="reg.example.com:80";proto=http`},
			}},
			base: "https://reg.example.com:443",
		},
		{
			name: "IPv6 address use host",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded":        []string{`for="2607:f0d0:1002:51::4";host="[2607:f0d0:1002:51::4]:5001"`},
				"X-Forwarded-Port": []string{"5002"},
			}},
			base: "http://[2607:f0d0:1002:51::4]:5001",
		},
		{
			name: "IPv6 address with port",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"Forwarded":        []string{`host="[2607:f0d0:1002:51::4]:4000"`},
				"X-Forwarded-Port": []string{"5001"},
			}},
			base: "http://[2607:f0d0:1002:51::4]:4000",
		},
		{
			name: "non-standard and standard forward headers",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{`https`},
				"X-Forwarded-Host":  []string{`first.example.com`},
				"X-Forwarded-Port":  []string{``},
				"Forwarded":         []string{`host=first.example.com; proto=https`},
			}},
			base: "https://first.example.com",
		},
		{
			name: "standard header takes precedence over non-standard headers",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{`http`},
				"Forwarded":         []string{`host=second.example.com; proto=https`},
				"X-Forwarded-Host":  []string{`first.example.com`},
				"X-Forwarded-Port":  []string{`4000`},
			}},
			base: "https://second.example.com",
		},
		{
			name: "incomplete standard header uses default",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{`https`},
				"Forwarded":         []string{`for=127.0.0.1`},
				"X-Forwarded-Host":  []string{`first.example.com`},
				"X-Forwarded-Port":  []string{`4000`},
			}},
			base: "http://" + u.Host,
		},
		{
			name: "standard with just proto",
			request: &http.Request{URL: u, Host: u.Host, Header: http.Header{
				"X-Forwarded-Proto": []string{`https`},
				"Forwarded":         []string{`proto=https`},
				"X-Forwarded-Host":  []string{`first.example.com`},
				"X-Forwarded-Port":  []string{`4000`},
			}},
			base: "https://" + u.Host,
		},
	}

	doTest := func(relative bool) {
		for _, tr := range testRequests {
			var builder *URLBuilder
			if tr.configHost.Scheme != "" && tr.configHost.Host != "" {
				builder = NewURLBuilder(&tr.configHost, relative)
			} else {
				builder = NewURLBuilderFromRequest(tr.request, relative)
			}

			for _, testCase := range makeURLBuilderTestCases(builder) {
				buildURL, err := testCase.build()
				expectedErr := testCase.expectedErr
				if !reflect.DeepEqual(expectedErr, err) {
					t.Fatalf("%s: Expecting %v but got error %v", testCase.description, expectedErr, err)
				}
				if expectedErr != nil {
					continue
				}

				expectedURL := testCase.expectedPath
				if !relative {
					expectedURL = tr.base + expectedURL
				}

				if buildURL != expectedURL {
					t.Errorf("[relative=%t, request=%q, case=%q]: %q != %q", relative, tr.name, testCase.description, buildURL, expectedURL)
				}
			}
		}
	}

	doTest(true)
	doTest(false)
}

func TestBuilderFromRequestWithPrefix(t *testing.T) {
	u, err := url.Parse("http://example.com/prefix/v2/")
	if err != nil {
		t.Fatal(err)
	}

	forwardedProtoHeader := make(http.Header, 1)
	forwardedProtoHeader.Set("X-Forwarded-Proto", "https")

	testRequests := []struct {
		request    *http.Request
		base       string
		configHost url.URL
	}{
		{
			request: &http.Request{URL: u, Host: u.Host},
			base:    "http://example.com/prefix/",
		},

		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedProtoHeader},
			base:    "http://example.com/prefix/",
		},
		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedProtoHeader},
			base:    "https://example.com/prefix/",
		},
		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedProtoHeader},
			base:    "https://subdomain.example.com/prefix/",
			configHost: url.URL{
				Scheme: "https",
				Host:   "subdomain.example.com",
				Path:   "/prefix/",
			},
		},
	}

	var relative bool
	for _, tr := range testRequests {
		var builder *URLBuilder
		if tr.configHost.Scheme != "" && tr.configHost.Host != "" {
			builder = NewURLBuilder(&tr.configHost, false)
		} else {
			builder = NewURLBuilderFromRequest(tr.request, false)
		}

		for _, testCase := range makeURLBuilderTestCases(builder) {
			buildURL, err := testCase.build()
			expectedErr := testCase.expectedErr
			if !reflect.DeepEqual(expectedErr, err) {
				t.Fatalf("%s: Expecting %v but got error %v", testCase.description, expectedErr, err)
			}
			if expectedErr != nil {
				continue
			}

			var expectedURL string
			proto, ok := tr.request.Header["X-Forwarded-Proto"]
			if !ok {
				expectedURL = testCase.expectedPath
				if !relative {
					expectedURL = tr.base[0:len(tr.base)-1] + expectedURL
				}
			} else {
				urlBase, err := url.Parse(tr.base)
				if err != nil {
					t.Fatal(err)
				}
				urlBase.Scheme = proto[0]
				expectedURL = testCase.expectedPath
				if !relative {
					expectedURL = urlBase.String()[0:len(urlBase.String())-1] + expectedURL
				}

			}

			if buildURL != expectedURL {
				t.Fatalf("%s: %q != %q", testCase.description, buildURL, expectedURL)
			}
		}
	}
}
