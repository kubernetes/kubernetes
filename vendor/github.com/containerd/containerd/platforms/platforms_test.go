package platforms

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"

	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

func TestParseSelector(t *testing.T) {
	var (
		defaultOS   = runtime.GOOS
		defaultArch = runtime.GOARCH
	)

	for _, testcase := range []struct {
		skip      bool
		input     string
		expected  specs.Platform
		formatted string
	}{
		// While wildcards are a valid use case for platform selection,
		// addressing these cases is outside the initial scope for this
		// package. When we do add platform wildcards, we should add in these
		// testcases to ensure that they are correctly represented.
		{
			skip:  true,
			input: "*",
			expected: specs.Platform{
				OS:           "*",
				Architecture: "*",
			},
			formatted: "*/*",
		},
		{
			skip:  true,
			input: "linux/*",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: "*",
			},
			formatted: "linux/*",
		},
		{
			skip:  true,
			input: "*/arm64",
			expected: specs.Platform{
				OS:           "*",
				Architecture: "arm64",
			},
			formatted: "*/arm64",
		},
		{
			// NOTE(stevvooe): In this case, the consumer can assume this is v7
			// but we leave the variant blank. This will represent the vast
			// majority of arm images.
			input: "linux/arm",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: "arm",
			},
			formatted: "linux/arm",
		},
		{
			input: "linux/arm/v6",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: "arm",
				Variant:      "v6",
			},
			formatted: "linux/arm/v6",
		},
		{
			input: "linux/arm/v7",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: "arm",
				Variant:      "v7",
			},
			formatted: "linux/arm/v7",
		},
		{
			input: "arm",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "arm",
			},
			formatted: "linux/arm",
		},
		{
			input: "armel",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "arm",
				Variant:      "v6",
			},
			formatted: "linux/arm/v6",
		},
		{
			input: "armhf",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "arm",
			},
			formatted: "linux/arm",
		},
		{
			input: "Aarch64",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "arm64",
			},
			formatted: joinNotEmpty(defaultOS, "arm64"),
		},
		{
			input: "x86_64",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "amd64",
			},
			formatted: joinNotEmpty(defaultOS, "amd64"),
		},
		{
			input: "Linux/x86_64",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: "amd64",
			},
			formatted: "linux/amd64",
		},
		{
			input: "i386",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "386",
			},
			formatted: joinNotEmpty(defaultOS, "386"),
		},
		{
			input: "linux",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: defaultArch,
			},
			formatted: joinNotEmpty("linux", defaultArch),
		},
		{
			input: "s390x",
			expected: specs.Platform{
				OS:           defaultOS,
				Architecture: "s390x",
			},
			formatted: joinNotEmpty(defaultOS, "s390x"),
		},
		{
			input: "linux/s390x",
			expected: specs.Platform{
				OS:           "linux",
				Architecture: "s390x",
			},
			formatted: "linux/s390x",
		},
		{
			input: "macOS",
			expected: specs.Platform{
				OS:           "darwin",
				Architecture: defaultArch,
			},
			formatted: joinNotEmpty("darwin", defaultArch),
		},
	} {
		t.Run(testcase.input, func(t *testing.T) {
			if testcase.skip {
				t.Skip("this case is not yet supported")
			}
			m, err := Parse(testcase.input)
			if err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(m.Spec(), testcase.expected) {
				t.Fatalf("platform did not match expected: %#v != %#v", m.Spec(), testcase.expected)
			}

			// ensure that match works on the input to the output.
			if ok := m.Match(testcase.expected); !ok {
				t.Fatalf("expected specifier %q matches %v", testcase.input, testcase.expected)
			}

			if fmt.Sprint(m) != testcase.formatted {
				t.Fatalf("unexpected matcher string:  %q != %q", fmt.Sprint(m), testcase.formatted)
			}

			formatted := Format(m.Spec())
			if formatted != testcase.formatted {
				t.Fatalf("unexpected format: %q != %q", formatted, testcase.formatted)
			}

			// re-parse the formatted output and ensure we are stable
			reparsed, err := Parse(formatted)
			if err != nil {
				t.Fatalf("error parsing formatted output: %v", err)
			}

			if Format(reparsed.Spec()) != formatted {
				t.Fatalf("normalized output did not survive the round trip: %v != %v", Format(reparsed.Spec()), formatted)
			}
		})
	}
}

func TestParseSelectorInvalid(t *testing.T) {
	for _, testcase := range []struct {
		input string
	}{
		{
			input: "", // empty
		},
		{
			input: "/linux/arm", // leading slash
		},
		{
			input: "linux/arm/", // trailing slash
		},
		{
			input: "linux /arm", // spaces
		},
		{
			input: "linux/&arm", // invalid character
		},
		{
			input: "linux/arm/foo/bar", // too many components
		},
	} {
		t.Run(testcase.input, func(t *testing.T) {
			if _, err := Parse(testcase.input); err == nil {
				t.Fatalf("should have received an error")
			}
		})
	}
}
