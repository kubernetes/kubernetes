package identifiers

import (
	"strings"
	"testing"

	"github.com/containerd/containerd/errdefs"
)

func TestValidIdentifiers(t *testing.T) {
	for _, input := range []string{
		"default",
		"Default",
		t.Name(),
		"default-default",
		"containerd.io",
		"foo.boo",
		"swarmkit.docker.io",
		"0912341234",
		"task.0.0123456789",
		"underscores_are_allowed",
		strings.Repeat("a", maxLength),
	} {
		t.Run(input, func(t *testing.T) {
			if err := Validate(input); err != nil {
				t.Fatalf("unexpected error: %v != nil", err)
			}
		})
	}
}

func TestInvalidIdentifiers(t *testing.T) {
	for _, input := range []string{
		"",
		".foo..foo",
		"foo/foo",
		"foo/..",
		"foo..foo",
		"foo.-boo",
		"-foo.boo",
		"foo.boo-",
		"but__only_tasteful_underscores",
		"zn--e9.org", // or something like it!
		"default--default",
		strings.Repeat("a", maxLength+1),
	} {

		t.Run(input, func(t *testing.T) {
			if err := Validate(input); err == nil {
				t.Fatal("expected invalid error")
			} else if !errdefs.IsInvalidArgument(err) {
				t.Fatal("error should be an invalid identifier error")
			}
		})
	}
}
