package storage

import (
	"testing"

	"github.com/docker/distribution/digest"
)

func TestPathMapper(t *testing.T) {
	for _, testcase := range []struct {
		spec     pathSpec
		expected string
		err      error
	}{
		{
			spec: manifestRevisionPathSpec{
				name:     "foo/bar",
				revision: "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/revisions/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
		},
		{
			spec: manifestRevisionLinkPathSpec{
				name:     "foo/bar",
				revision: "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/revisions/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789/link",
		},
		{
			spec: manifestSignatureLinkPathSpec{
				name:      "foo/bar",
				revision:  "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
				signature: "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/revisions/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789/signatures/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789/link",
		},
		{
			spec: manifestSignaturesPathSpec{
				name:     "foo/bar",
				revision: "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/revisions/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789/signatures",
		},
		{
			spec: manifestTagsPathSpec{
				name: "foo/bar",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/tags",
		},
		{
			spec: manifestTagPathSpec{
				name: "foo/bar",
				tag:  "thetag",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/tags/thetag",
		},
		{
			spec: manifestTagCurrentPathSpec{
				name: "foo/bar",
				tag:  "thetag",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/tags/thetag/current/link",
		},
		{
			spec: manifestTagIndexPathSpec{
				name: "foo/bar",
				tag:  "thetag",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/tags/thetag/index",
		},
		{
			spec: manifestTagIndexEntryPathSpec{
				name:     "foo/bar",
				tag:      "thetag",
				revision: "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/tags/thetag/index/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
		},
		{
			spec: manifestTagIndexEntryLinkPathSpec{
				name:     "foo/bar",
				tag:      "thetag",
				revision: "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_manifests/tags/thetag/index/sha256/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789/link",
		},

		{
			spec: uploadDataPathSpec{
				name: "foo/bar",
				id:   "asdf-asdf-asdf-adsf",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_uploads/asdf-asdf-asdf-adsf/data",
		},
		{
			spec: uploadStartedAtPathSpec{
				name: "foo/bar",
				id:   "asdf-asdf-asdf-adsf",
			},
			expected: "/docker/registry/v2/repositories/foo/bar/_uploads/asdf-asdf-asdf-adsf/startedat",
		},
	} {
		p, err := pathFor(testcase.spec)
		if err != nil {
			t.Fatalf("unexpected generating path (%T): %v", testcase.spec, err)
		}

		if p != testcase.expected {
			t.Fatalf("unexpected path generated (%T): %q != %q", testcase.spec, p, testcase.expected)
		}
	}

	// Add a few test cases to ensure we cover some errors

	// Specify a path that requires a revision and get a digest validation error.
	badpath, err := pathFor(manifestSignaturesPathSpec{
		name: "foo/bar",
	})

	if err == nil {
		t.Fatalf("expected an error when mapping an invalid revision: %s", badpath)
	}

}

func TestDigestFromPath(t *testing.T) {
	for _, testcase := range []struct {
		path       string
		expected   digest.Digest
		multilevel bool
		err        error
	}{
		{
			path:       "/docker/registry/v2/blobs/sha256/99/9943fffae777400c0344c58869c4c2619c329ca3ad4df540feda74d291dd7c86/data",
			multilevel: true,
			expected:   "sha256:9943fffae777400c0344c58869c4c2619c329ca3ad4df540feda74d291dd7c86",
			err:        nil,
		},
	} {
		result, err := digestFromPath(testcase.path)
		if err != testcase.err {
			t.Fatalf("Unexpected error value %v when we wanted %v", err, testcase.err)
		}

		if result != testcase.expected {
			t.Fatalf("Unexpected result value %v when we wanted %v", result, testcase.expected)

		}
	}
}
