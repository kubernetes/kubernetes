package v2

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/docker/distribution/reference"
)

type urlBuilderTestCase struct {
	description  string
	expectedPath string
	build        func() (string, error)
}

func makeURLBuilderTestCases(urlBuilder *URLBuilder) []urlBuilderTestCase {
	fooBarRef, _ := reference.ParseNamed("foo/bar")
	return []urlBuilderTestCase{
		{
			description:  "test base url",
			expectedPath: "/v2/",
			build:        urlBuilder.BuildBaseURL,
		},
		{
			description:  "test tags url",
			expectedPath: "/v2/foo/bar/tags/list",
			build: func() (string, error) {
				return urlBuilder.BuildTagsURL(fooBarRef)
			},
		},
		{
			description:  "test manifest url",
			expectedPath: "/v2/foo/bar/manifests/tag",
			build: func() (string, error) {
				ref, _ := reference.WithTag(fooBarRef, "tag")
				return urlBuilder.BuildManifestURL(ref)
			},
		},
		{
			description:  "build blob url",
			expectedPath: "/v2/foo/bar/blobs/sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5",
			build: func() (string, error) {
				ref, _ := reference.WithDigest(fooBarRef, "sha256:3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5")
				return urlBuilder.BuildBlobURL(ref)
			},
		},
		{
			description:  "build blob upload url",
			expectedPath: "/v2/foo/bar/blobs/uploads/",
			build: func() (string, error) {
				return urlBuilder.BuildBlobUploadURL(fooBarRef)
			},
		},
		{
			description:  "build blob upload url with digest and size",
			expectedPath: "/v2/foo/bar/blobs/uploads/?digest=sha256%3A3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5&size=10000",
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
			build: func() (string, error) {
				return urlBuilder.BuildBlobUploadChunkURL(fooBarRef, "uuid-part")
			},
		},
		{
			description:  "build blob upload chunk url with digest and size",
			expectedPath: "/v2/foo/bar/blobs/uploads/uuid-part?digest=sha256%3A3b3692957d439ac1928219a83fac91e7bf96c153725526874673ae1f2023f8d5&size=10000",
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
				if err != nil {
					t.Fatalf("%s: error building url: %v", testCase.description, err)
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
				if err != nil {
					t.Fatalf("%s: error building url: %v", testCase.description, err)
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

	forwardedProtoHeader := make(http.Header, 1)
	forwardedProtoHeader.Set("X-Forwarded-Proto", "https")

	forwardedHostHeader1 := make(http.Header, 1)
	forwardedHostHeader1.Set("X-Forwarded-Host", "first.example.com")

	forwardedHostHeader2 := make(http.Header, 1)
	forwardedHostHeader2.Set("X-Forwarded-Host", "first.example.com, proxy1.example.com")

	testRequests := []struct {
		request    *http.Request
		base       string
		configHost url.URL
	}{
		{
			request: &http.Request{URL: u, Host: u.Host},
			base:    "http://example.com",
		},

		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedProtoHeader},
			base:    "http://example.com",
		},
		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedProtoHeader},
			base:    "https://example.com",
		},
		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedHostHeader1},
			base:    "http://first.example.com",
		},
		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedHostHeader2},
			base:    "http://first.example.com",
		},
		{
			request: &http.Request{URL: u, Host: u.Host, Header: forwardedHostHeader2},
			base:    "https://third.example.com:5000",
			configHost: url.URL{
				Scheme: "https",
				Host:   "third.example.com:5000",
			},
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
				if err != nil {
					t.Fatalf("%s: error building url: %v", testCase.description, err)
				}

				var expectedURL string
				proto, ok := tr.request.Header["X-Forwarded-Proto"]
				if !ok {
					expectedURL = testCase.expectedPath
					if !relative {
						expectedURL = tr.base + expectedURL
					}
				} else {
					urlBase, err := url.Parse(tr.base)
					if err != nil {
						t.Fatal(err)
					}
					urlBase.Scheme = proto[0]
					expectedURL = testCase.expectedPath
					if !relative {
						expectedURL = urlBase.String() + expectedURL
					}
				}

				if buildURL != expectedURL {
					t.Fatalf("%s: %q != %q", testCase.description, buildURL, expectedURL)
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
			if err != nil {
				t.Fatalf("%s: error building url: %v", testCase.description, err)
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
