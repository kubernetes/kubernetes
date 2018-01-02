package reference

import (
	"testing"

	digest "github.com/opencontainers/go-digest"
)

func TestReferenceParser(t *testing.T) {
	for _, testcase := range []struct {
		Skip       bool
		Name       string
		Input      string
		Normalized string
		Digest     digest.Digest
		Hostname   string
		Expected   Spec
		Err        error
	}{
		{
			Name:       "Basic",
			Input:      "docker.io/library/redis:foo?fooo=asdf",
			Normalized: "docker.io/library/redis:foo",
			Hostname:   "docker.io",
			Expected: Spec{
				Locator: "docker.io/library/redis",
				Object:  "foo",
			},
		},
		{
			Name:       "BasicWithDigest",
			Input:      "docker.io/library/redis:foo@sha256:abcdef?fooo=asdf",
			Normalized: "docker.io/library/redis:foo@sha256:abcdef",
			Hostname:   "docker.io",
			Digest:     "sha256:abcdef",
			Expected: Spec{
				Locator: "docker.io/library/redis",
				Object:  "foo@sha256:abcdef",
			},
		},

		{
			Name:  "DigestOnly",
			Input: "docker.io/library/redis@sha256:abcdef?fooo=asdf",
			Expected: Spec{
				Locator: "docker.io/library/redis",
				Object:  "@sha256:abcdef",
			},
			Hostname:   "docker.io",
			Normalized: "docker.io/library/redis@sha256:abcdef",
			Digest:     "sha256:abcdef",
		},
		{
			Name:       "AtShortDigest",
			Input:      "docker.io/library/redis:obj@abcdef?fooo=asdf",
			Normalized: "docker.io/library/redis:obj@abcdef",
			Hostname:   "docker.io",
			Digest:     "abcdef",
			Expected: Spec{
				Locator: "docker.io/library/redis",
				Object:  "obj@abcdef",
			},
		},
		{
			Name:       "HostWithPort",
			Input:      "localhost:5000/library/redis:obj@abcdef?fooo=asdf",
			Normalized: "localhost:5000/library/redis:obj@abcdef",
			Hostname:   "localhost:5000",
			Digest:     "abcdef",
			Expected: Spec{
				Locator: "localhost:5000/library/redis",
				Object:  "obj@abcdef",
			},
		},
		{
			Name:  "HostnameRequired",
			Input: "/docker.io/library/redis:obj@abcdef?fooo=asdf",
			Err:   ErrHostnameRequired,
		},
		{
			Name:       "ErrObjectRequired",
			Input:      "docker.io/library/redis?fooo=asdf",
			Hostname:   "docker.io",
			Normalized: "docker.io/library/redis",
			Expected: Spec{
				Locator: "docker.io/library/redis",
				Object:  "",
			},
		},
		{
			Name:     "Subdomain",
			Input:    "sub-dom1.foo.com/bar/baz/quux:latest",
			Hostname: "sub-dom1.foo.com",
			Expected: Spec{
				Locator: "sub-dom1.foo.com/bar/baz/quux",
				Object:  "latest",
			},
		},
		{
			Name:     "SubdomainWithLongTag",
			Input:    "sub-dom1.foo.com/bar/baz/quux:some-long-tag",
			Hostname: "sub-dom1.foo.com",
			Expected: Spec{
				Locator: "sub-dom1.foo.com/bar/baz/quux",
				Object:  "some-long-tag",
			},
		},
		{
			Name:     "AGCRAppears",
			Input:    "b.gcr.io/test.example.com/my-app:test.example.com",
			Hostname: "b.gcr.io",
			Expected: Spec{
				Locator: "b.gcr.io/test.example.com/my-app",
				Object:  "test.example.com",
			},
		},
		{
			Name:     "Punycode",
			Input:    "xn--n3h.com/myimage:xn--n3h.com", // ☃.com in punycode
			Hostname: "xn--n3h.com",
			Expected: Spec{
				Locator: "xn--n3h.com/myimage",
				Object:  "xn--n3h.com",
			},
		},
		{

			Name:     "PunycodeWithDigest",
			Input:    "xn--7o8h.com/myimage:xn--7o8h.com@sha512:fffffff",
			Hostname: "xn--7o8h.com",
			Digest:   "sha512:fffffff",
			Expected: Spec{
				Locator: "xn--7o8h.com/myimage",
				Object:  "xn--7o8h.com@sha512:fffffff",
			},
		},
		{
			Skip:     true, // TODO(stevvooe): Implement this case.
			Name:     "SchemeDefined",
			Input:    "http://xn--7o8h.com/myimage:xn--7o8h.com@sha512:fffffff",
			Hostname: "xn--7o8h.com",
			Digest:   "sha512:fffffff",
			Err:      ErrInvalid,
		},
	} {
		t.Run(testcase.Name, func(t *testing.T) {
			if testcase.Skip {
				t.Skip("testcase disabled")
				return
			}

			ref, err := Parse(testcase.Input)
			if err != testcase.Err {
				if testcase.Err != nil {
					t.Fatalf("expected error %v for %q, got %v, %#v", testcase.Err, testcase.Input, err, ref)
				} else {
					t.Fatalf("unexpected error %v", err)
				}
			} else if testcase.Err != nil {
				return
			}

			if ref != testcase.Expected {
				t.Fatalf("%#v != %#v", ref, testcase.Expected)
			}

			if testcase.Normalized == "" {
				testcase.Normalized = testcase.Input
			}

			if ref.String() != testcase.Normalized {
				t.Fatalf("normalization failed: %v != %v", ref.String(), testcase.Normalized)
			}

			if ref.Digest() != testcase.Digest {
				t.Fatalf("digest extraction failed: %v != %v", ref.Digest(), testcase.Digest)
			}

			if ref.Hostname() != testcase.Hostname {
				t.Fatalf("unexpected hostname: %v != %v", ref.Hostname(), testcase.Hostname)
			}
		})
	}
}
