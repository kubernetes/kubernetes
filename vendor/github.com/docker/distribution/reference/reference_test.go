package reference

import (
	_ "crypto/sha256"
	_ "crypto/sha512"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/opencontainers/go-digest"
)

func TestReferenceParse(t *testing.T) {
	// referenceTestcases is a unified set of testcases for
	// testing the parsing of references
	referenceTestcases := []struct {
		// input is the repository name or name component testcase
		input string
		// err is the error expected from Parse, or nil
		err error
		// repository is the string representation for the reference
		repository string
		// domain is the domain expected in the reference
		domain string
		// tag is the tag for the reference
		tag string
		// digest is the digest for the reference (enforces digest reference)
		digest string
	}{
		{
			input:      "test_com",
			repository: "test_com",
		},
		{
			input:      "test.com:tag",
			repository: "test.com",
			tag:        "tag",
		},
		{
			input:      "test.com:5000",
			repository: "test.com",
			tag:        "5000",
		},
		{
			input:      "test.com/repo:tag",
			domain:     "test.com",
			repository: "test.com/repo",
			tag:        "tag",
		},
		{
			input:      "test:5000/repo",
			domain:     "test:5000",
			repository: "test:5000/repo",
		},
		{
			input:      "test:5000/repo:tag",
			domain:     "test:5000",
			repository: "test:5000/repo",
			tag:        "tag",
		},
		{
			input:      "test:5000/repo@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			domain:     "test:5000",
			repository: "test:5000/repo",
			digest:     "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		},
		{
			input:      "test:5000/repo:tag@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			domain:     "test:5000",
			repository: "test:5000/repo",
			tag:        "tag",
			digest:     "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		},
		{
			input:      "test:5000/repo",
			domain:     "test:5000",
			repository: "test:5000/repo",
		},
		{
			input: "",
			err:   ErrNameEmpty,
		},
		{
			input: ":justtag",
			err:   ErrReferenceInvalidFormat,
		},
		{
			input: "@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			err:   ErrReferenceInvalidFormat,
		},
		{
			input: "repo@sha256:ffffffffffffffffffffffffffffffffff",
			err:   digest.ErrDigestInvalidLength,
		},
		{
			input: "validname@invaliddigest:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			err:   digest.ErrDigestUnsupported,
		},
		{
			input: "Uppercase:tag",
			err:   ErrNameContainsUppercase,
		},
		// FIXME "Uppercase" is incorrectly handled as a domain-name here, therefore passes.
		// See https://github.com/docker/distribution/pull/1778, and https://github.com/docker/docker/pull/20175
		//{
		//	input: "Uppercase/lowercase:tag",
		//	err:   ErrNameContainsUppercase,
		//},
		{
			input: "test:5000/Uppercase/lowercase:tag",
			err:   ErrNameContainsUppercase,
		},
		{
			input:      "lowercase:Uppercase",
			repository: "lowercase",
			tag:        "Uppercase",
		},
		{
			input: strings.Repeat("a/", 128) + "a:tag",
			err:   ErrNameTooLong,
		},
		{
			input:      strings.Repeat("a/", 127) + "a:tag-puts-this-over-max",
			domain:     "a",
			repository: strings.Repeat("a/", 127) + "a",
			tag:        "tag-puts-this-over-max",
		},
		{
			input: "aa/asdf$$^/aa",
			err:   ErrReferenceInvalidFormat,
		},
		{
			input:      "sub-dom1.foo.com/bar/baz/quux",
			domain:     "sub-dom1.foo.com",
			repository: "sub-dom1.foo.com/bar/baz/quux",
		},
		{
			input:      "sub-dom1.foo.com/bar/baz/quux:some-long-tag",
			domain:     "sub-dom1.foo.com",
			repository: "sub-dom1.foo.com/bar/baz/quux",
			tag:        "some-long-tag",
		},
		{
			input:      "b.gcr.io/test.example.com/my-app:test.example.com",
			domain:     "b.gcr.io",
			repository: "b.gcr.io/test.example.com/my-app",
			tag:        "test.example.com",
		},
		{
			input:      "xn--n3h.com/myimage:xn--n3h.com", // ☃.com in punycode
			domain:     "xn--n3h.com",
			repository: "xn--n3h.com/myimage",
			tag:        "xn--n3h.com",
		},
		{
			input:      "xn--7o8h.com/myimage:xn--7o8h.com@sha512:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", // 🐳.com in punycode
			domain:     "xn--7o8h.com",
			repository: "xn--7o8h.com/myimage",
			tag:        "xn--7o8h.com",
			digest:     "sha512:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		},
		{
			input:      "foo_bar.com:8080",
			repository: "foo_bar.com",
			tag:        "8080",
		},
		{
			input:      "foo/foo_bar.com:8080",
			domain:     "foo",
			repository: "foo/foo_bar.com",
			tag:        "8080",
		},
	}
	for _, testcase := range referenceTestcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.input)+": "+format, v...)
			t.Fail()
		}

		repo, err := Parse(testcase.input)
		if testcase.err != nil {
			if err == nil {
				failf("missing expected error: %v", testcase.err)
			} else if testcase.err != err {
				failf("mismatched error: got %v, expected %v", err, testcase.err)
			}
			continue
		} else if err != nil {
			failf("unexpected parse error: %v", err)
			continue
		}
		if repo.String() != testcase.input {
			failf("mismatched repo: got %q, expected %q", repo.String(), testcase.input)
		}

		if named, ok := repo.(Named); ok {
			if named.Name() != testcase.repository {
				failf("unexpected repository: got %q, expected %q", named.Name(), testcase.repository)
			}
			domain, _ := SplitHostname(named)
			if domain != testcase.domain {
				failf("unexpected domain: got %q, expected %q", domain, testcase.domain)
			}
		} else if testcase.repository != "" || testcase.domain != "" {
			failf("expected named type, got %T", repo)
		}

		tagged, ok := repo.(Tagged)
		if testcase.tag != "" {
			if ok {
				if tagged.Tag() != testcase.tag {
					failf("unexpected tag: got %q, expected %q", tagged.Tag(), testcase.tag)
				}
			} else {
				failf("expected tagged type, got %T", repo)
			}
		} else if ok {
			failf("unexpected tagged type")
		}

		digested, ok := repo.(Digested)
		if testcase.digest != "" {
			if ok {
				if digested.Digest().String() != testcase.digest {
					failf("unexpected digest: got %q, expected %q", digested.Digest().String(), testcase.digest)
				}
			} else {
				failf("expected digested type, got %T", repo)
			}
		} else if ok {
			failf("unexpected digested type")
		}

	}
}

// TestWithNameFailure tests cases where WithName should fail. Cases where it
// should succeed are covered by TestSplitHostname, below.
func TestWithNameFailure(t *testing.T) {
	testcases := []struct {
		input string
		err   error
	}{
		{
			input: "",
			err:   ErrNameEmpty,
		},
		{
			input: ":justtag",
			err:   ErrReferenceInvalidFormat,
		},
		{
			input: "@sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			err:   ErrReferenceInvalidFormat,
		},
		{
			input: "validname@invaliddigest:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
			err:   ErrReferenceInvalidFormat,
		},
		{
			input: strings.Repeat("a/", 128) + "a:tag",
			err:   ErrNameTooLong,
		},
		{
			input: "aa/asdf$$^/aa",
			err:   ErrReferenceInvalidFormat,
		},
	}
	for _, testcase := range testcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.input)+": "+format, v...)
			t.Fail()
		}

		_, err := WithName(testcase.input)
		if err == nil {
			failf("no error parsing name. expected: %s", testcase.err)
		}
	}
}

func TestSplitHostname(t *testing.T) {
	testcases := []struct {
		input  string
		domain string
		name   string
	}{
		{
			input:  "test.com/foo",
			domain: "test.com",
			name:   "foo",
		},
		{
			input:  "test_com/foo",
			domain: "",
			name:   "test_com/foo",
		},
		{
			input:  "test:8080/foo",
			domain: "test:8080",
			name:   "foo",
		},
		{
			input:  "test.com:8080/foo",
			domain: "test.com:8080",
			name:   "foo",
		},
		{
			input:  "test-com:8080/foo",
			domain: "test-com:8080",
			name:   "foo",
		},
		{
			input:  "xn--n3h.com:18080/foo",
			domain: "xn--n3h.com:18080",
			name:   "foo",
		},
	}
	for _, testcase := range testcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.input)+": "+format, v...)
			t.Fail()
		}

		named, err := WithName(testcase.input)
		if err != nil {
			failf("error parsing name: %s", err)
		}
		domain, name := SplitHostname(named)
		if domain != testcase.domain {
			failf("unexpected domain: got %q, expected %q", domain, testcase.domain)
		}
		if name != testcase.name {
			failf("unexpected name: got %q, expected %q", name, testcase.name)
		}
	}
}

type serializationType struct {
	Description string
	Field       Field
}

func TestSerialization(t *testing.T) {
	testcases := []struct {
		description string
		input       string
		name        string
		tag         string
		digest      string
		err         error
	}{
		{
			description: "empty value",
			err:         ErrNameEmpty,
		},
		{
			description: "just a name",
			input:       "example.com:8000/named",
			name:        "example.com:8000/named",
		},
		{
			description: "name with a tag",
			input:       "example.com:8000/named:tagged",
			name:        "example.com:8000/named",
			tag:         "tagged",
		},
		{
			description: "name with digest",
			input:       "other.com/named@sha256:1234567890098765432112345667890098765432112345667890098765432112",
			name:        "other.com/named",
			digest:      "sha256:1234567890098765432112345667890098765432112345667890098765432112",
		},
	}
	for _, testcase := range testcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.input)+": "+format, v...)
			t.Fail()
		}

		m := map[string]string{
			"Description": testcase.description,
			"Field":       testcase.input,
		}
		b, err := json.Marshal(m)
		if err != nil {
			failf("error marshalling: %v", err)
		}
		t := serializationType{}

		if err := json.Unmarshal(b, &t); err != nil {
			if testcase.err == nil {
				failf("error unmarshalling: %v", err)
			}
			if err != testcase.err {
				failf("wrong error, expected %v, got %v", testcase.err, err)
			}

			continue
		} else if testcase.err != nil {
			failf("expected error unmarshalling: %v", testcase.err)
		}

		if t.Description != testcase.description {
			failf("wrong description, expected %q, got %q", testcase.description, t.Description)
		}

		ref := t.Field.Reference()

		if named, ok := ref.(Named); ok {
			if named.Name() != testcase.name {
				failf("unexpected repository: got %q, expected %q", named.Name(), testcase.name)
			}
		} else if testcase.name != "" {
			failf("expected named type, got %T", ref)
		}

		tagged, ok := ref.(Tagged)
		if testcase.tag != "" {
			if ok {
				if tagged.Tag() != testcase.tag {
					failf("unexpected tag: got %q, expected %q", tagged.Tag(), testcase.tag)
				}
			} else {
				failf("expected tagged type, got %T", ref)
			}
		} else if ok {
			failf("unexpected tagged type")
		}

		digested, ok := ref.(Digested)
		if testcase.digest != "" {
			if ok {
				if digested.Digest().String() != testcase.digest {
					failf("unexpected digest: got %q, expected %q", digested.Digest().String(), testcase.digest)
				}
			} else {
				failf("expected digested type, got %T", ref)
			}
		} else if ok {
			failf("unexpected digested type")
		}

		t = serializationType{
			Description: testcase.description,
			Field:       AsField(ref),
		}

		b2, err := json.Marshal(t)
		if err != nil {
			failf("error marshing serialization type: %v", err)
		}

		if string(b) != string(b2) {
			failf("unexpected serialized value: expected %q, got %q", string(b), string(b2))
		}

		// Ensure t.Field is not implementing "Reference" directly, getting
		// around the Reference type system
		var fieldInterface interface{} = t.Field
		if _, ok := fieldInterface.(Reference); ok {
			failf("field should not implement Reference interface")
		}

	}
}

func TestWithTag(t *testing.T) {
	testcases := []struct {
		name     string
		digest   digest.Digest
		tag      string
		combined string
	}{
		{
			name:     "test.com/foo",
			tag:      "tag",
			combined: "test.com/foo:tag",
		},
		{
			name:     "foo",
			tag:      "tag2",
			combined: "foo:tag2",
		},
		{
			name:     "test.com:8000/foo",
			tag:      "tag4",
			combined: "test.com:8000/foo:tag4",
		},
		{
			name:     "test.com:8000/foo",
			tag:      "TAG5",
			combined: "test.com:8000/foo:TAG5",
		},
		{
			name:     "test.com:8000/foo",
			digest:   "sha256:1234567890098765432112345667890098765",
			tag:      "TAG5",
			combined: "test.com:8000/foo:TAG5@sha256:1234567890098765432112345667890098765",
		},
	}
	for _, testcase := range testcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.name)+": "+format, v...)
			t.Fail()
		}

		named, err := WithName(testcase.name)
		if err != nil {
			failf("error parsing name: %s", err)
		}
		if testcase.digest != "" {
			canonical, err := WithDigest(named, testcase.digest)
			if err != nil {
				failf("error adding digest")
			}
			named = canonical
		}

		tagged, err := WithTag(named, testcase.tag)
		if err != nil {
			failf("WithTag failed: %s", err)
		}
		if tagged.String() != testcase.combined {
			failf("unexpected: got %q, expected %q", tagged.String(), testcase.combined)
		}
	}
}

func TestWithDigest(t *testing.T) {
	testcases := []struct {
		name     string
		digest   digest.Digest
		tag      string
		combined string
	}{
		{
			name:     "test.com/foo",
			digest:   "sha256:1234567890098765432112345667890098765",
			combined: "test.com/foo@sha256:1234567890098765432112345667890098765",
		},
		{
			name:     "foo",
			digest:   "sha256:1234567890098765432112345667890098765",
			combined: "foo@sha256:1234567890098765432112345667890098765",
		},
		{
			name:     "test.com:8000/foo",
			digest:   "sha256:1234567890098765432112345667890098765",
			combined: "test.com:8000/foo@sha256:1234567890098765432112345667890098765",
		},
		{
			name:     "test.com:8000/foo",
			digest:   "sha256:1234567890098765432112345667890098765",
			tag:      "latest",
			combined: "test.com:8000/foo:latest@sha256:1234567890098765432112345667890098765",
		},
	}
	for _, testcase := range testcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.name)+": "+format, v...)
			t.Fail()
		}

		named, err := WithName(testcase.name)
		if err != nil {
			failf("error parsing name: %s", err)
		}
		if testcase.tag != "" {
			tagged, err := WithTag(named, testcase.tag)
			if err != nil {
				failf("error adding tag")
			}
			named = tagged
		}
		digested, err := WithDigest(named, testcase.digest)
		if err != nil {
			failf("WithDigest failed: %s", err)
		}
		if digested.String() != testcase.combined {
			failf("unexpected: got %q, expected %q", digested.String(), testcase.combined)
		}
	}
}

func TestParseNamed(t *testing.T) {
	testcases := []struct {
		input  string
		domain string
		name   string
		err    error
	}{
		{
			input:  "test.com/foo",
			domain: "test.com",
			name:   "foo",
		},
		{
			input:  "test:8080/foo",
			domain: "test:8080",
			name:   "foo",
		},
		{
			input: "test_com/foo",
			err:   ErrNameNotCanonical,
		},
		{
			input: "test.com",
			err:   ErrNameNotCanonical,
		},
		{
			input: "foo",
			err:   ErrNameNotCanonical,
		},
		{
			input: "library/foo",
			err:   ErrNameNotCanonical,
		},
		{
			input:  "docker.io/library/foo",
			domain: "docker.io",
			name:   "library/foo",
		},
		// Ambiguous case, parser will add "library/" to foo
		{
			input: "docker.io/foo",
			err:   ErrNameNotCanonical,
		},
	}
	for _, testcase := range testcases {
		failf := func(format string, v ...interface{}) {
			t.Logf(strconv.Quote(testcase.input)+": "+format, v...)
			t.Fail()
		}

		named, err := ParseNamed(testcase.input)
		if err != nil && testcase.err == nil {
			failf("error parsing name: %s", err)
			continue
		} else if err == nil && testcase.err != nil {
			failf("parsing succeded: expected error %v", testcase.err)
			continue
		} else if err != testcase.err {
			failf("unexpected error %v, expected %v", err, testcase.err)
			continue
		} else if err != nil {
			continue
		}

		domain, name := SplitHostname(named)
		if domain != testcase.domain {
			failf("unexpected domain: got %q, expected %q", domain, testcase.domain)
		}
		if name != testcase.name {
			failf("unexpected name: got %q, expected %q", name, testcase.name)
		}
	}
}
