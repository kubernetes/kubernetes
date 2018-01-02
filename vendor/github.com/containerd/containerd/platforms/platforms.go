// Package platforms provides a toolkit for normalizing, matching and
// specifying container platforms.
//
// Centered around OCI platform specifications, we define a string-based
// specifier syntax that can be used for user input. With a specifier, users
// only need to specify the parts of the platform that are relevant to their
// context, providing an operating system or architecture or both.
//
// How do I use this package?
//
// The vast majority of use cases should simply use the match function with
// user input. The first step is to parse a specifier into a matcher:
//
//   m, err := Parse("linux")
//   if err != nil { ... }
//
// Once you have a matcher, use it to match against the platform declared by a
// component, typically from an image or runtime. Since extracting an images
// platform is a little more involved, we'll use an example against the
// platform default:
//
//   if ok := m.Match(Default()); !ok { /* doesn't match */ }
//
// This can be composed in loops for resolving runtimes or used as a filter for
// fetch and select images.
//
// More details of the specifier syntax and platform spec follow.
//
// Declaring Platform Support
//
// Components that have strict platform requirements should use the OCI
// platform specification to declare their support. Typically, this will be
// images and runtimes that should make these declaring which platform they
// support specifically. This looks roughly as follows:
//
//   type Platform struct {
//	   Architecture string
//	   OS           string
//	   Variant      string
//   }
//
// Most images and runtimes should at least set Architecture and OS, according
// to their GOARCH and GOOS values, respectively (follow the OCI image
// specification when in doubt). ARM should set variant under certain
// discussions, which are outlined below.
//
// Platform Specifiers
//
// While the OCI platform specifications provide a tool for components to
// specify structured information, user input typically doesn't need the full
// context and much can be inferred. To solve this problem, we introduced
// "specifiers". A specifier has the format
// `<os>|<arch>|<os>/<arch>[/<variant>]`.  The user can provide either the
// operating system or the architecture or both.
//
// An example of a common specifier is `linux/amd64`. If the host has a default
// of runtime that matches this, the user can simply provide the component that
// matters. For example, if a image provides amd64 and arm64 support, the
// operating system, `linux` can be inferred, so they only have to provide
// `arm64` or `amd64`. Similar behavior is implemented for operating systems,
// where the architecture may be known but a runtime may support images from
// different operating systems.
//
// Normalization
//
// Because not all users are familiar with the way the Go runtime represents
// platforms, several normalizations have been provided to make this package
// easier to user.
//
// The following are performed for architectures:
//
//   Value    Normalized
//   aarch64  arm64
//   armhf    arm
//   armel    arm/v6
//   i386     386
//   x86_64   amd64
//   x86-64   amd64
//
// We also normalize the operating system `macos` to `darwin`.
//
// ARM Support
//
// To qualify ARM architecture, the Variant field is used to qualify the arm
// version. The most common arm version, v7, is represented without the variant
// unless it is explicitly provided. This is treated as equivalent to armhf. A
// previous architecture, armel, will be normalized to arm/v6.
//
// While these normalizations are provided, their support on arm platforms has
// not yet been fully implemented and tested.
package platforms

import (
	"regexp"
	"runtime"
	"strings"

	"github.com/containerd/containerd/errdefs"
	specs "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

var (
	specifierRe = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)
)

// Matcher matches platforms specifications, provided by an image or runtime.
type Matcher interface {
	Spec() specs.Platform
	Match(platform specs.Platform) bool
}

type matcher struct {
	specs.Platform
}

func (m *matcher) Spec() specs.Platform {
	return m.Platform
}

func (m *matcher) Match(platform specs.Platform) bool {
	normalized := Normalize(platform)
	return m.OS == normalized.OS &&
		m.Architecture == normalized.Architecture &&
		m.Variant == normalized.Variant
}

func (m *matcher) String() string {
	return Format(m.Platform)
}

// Parse parses the platform specifier syntax into a platform declaration.
//
// Platform specifiers are in the format `<os>|<arch>|<os>/<arch>[/<variant>]`.
// The minimum required information for a platform specifier is the operating
// system or architecture. If there is only a single string (no slashes), the
// value will be matched against the known set of operating systems, then fall
// back to the known set of architectures. The missing component will be
// inferred based on the local environment.
//
// Applications should opt to use `Match` over directly parsing specifiers.
func Parse(specifier string) (Matcher, error) {
	if strings.Contains(specifier, "*") {
		// TODO(stevvooe): need to work out exact wildcard handling
		return nil, errors.Wrapf(errdefs.ErrInvalidArgument, "%q: wildcards not yet supported", specifier)
	}

	parts := strings.Split(specifier, "/")

	for _, part := range parts {
		if !specifierRe.MatchString(part) {
			return nil, errors.Wrapf(errdefs.ErrInvalidArgument, "%q is an invalid component of %q: platform specifier component must match %q", part, specifier, specifierRe.String())
		}
	}

	var p specs.Platform
	switch len(parts) {
	case 1:
		// in this case, we will test that the value might be an OS, then look
		// it up. If it is not known, we'll treat it as an architecture. Since
		// we have very little information about the platform here, we are
		// going to be a little more strict if we don't know about the argument
		// value.
		p.OS = normalizeOS(parts[0])
		if isKnownOS(p.OS) {
			// picks a default architecture
			p.Architecture = runtime.GOARCH
			if p.Architecture == "arm" {
				// TODO(stevvooe): Resolve arm variant, if not v6 (default)
				return nil, errors.Wrapf(errdefs.ErrNotImplemented, "arm support not fully implemented")
			}

			return &matcher{p}, nil
		}

		p.Architecture, p.Variant = normalizeArch(parts[0], "")
		if isKnownArch(p.Architecture) {
			p.OS = runtime.GOOS
			return &matcher{p}, nil
		}

		return nil, errors.Wrapf(errdefs.ErrInvalidArgument, "%q: unknown operating system or architecture", specifier)
	case 2:
		// In this case, we treat as a regular os/arch pair. We don't care
		// about whether or not we know of the platform.
		p.OS = normalizeOS(parts[0])
		p.Architecture, p.Variant = normalizeArch(parts[1], "")

		return &matcher{p}, nil
	case 3:
		// we have a fully specified variant, this is rare
		p.OS = normalizeOS(parts[0])
		p.Architecture, p.Variant = normalizeArch(parts[1], parts[2])

		return &matcher{p}, nil
	}

	return nil, errors.Wrapf(errdefs.ErrInvalidArgument, "%q: cannot parse platform specifier", specifier)
}

// Format returns a string specifier from the provided platform specification.
func Format(platform specs.Platform) string {
	if platform.OS == "" {
		return "unknown"
	}

	return joinNotEmpty(platform.OS, platform.Architecture, platform.Variant)
}

func joinNotEmpty(s ...string) string {
	var ss []string
	for _, s := range s {
		if s == "" {
			continue
		}

		ss = append(ss, s)
	}

	return strings.Join(ss, "/")
}

// Normalize validates and translate the platform to the canonical value.
//
// For example, if "Aarch64" is encountered, we change it to "arm64" or if
// "x86_64" is encountered, it becomes "amd64".
func Normalize(platform specs.Platform) specs.Platform {
	platform.OS = normalizeOS(platform.OS)
	platform.Architecture, platform.Variant = normalizeArch(platform.Architecture, platform.Variant)

	// these fields are deprecated, remove them
	platform.OSFeatures = nil
	platform.OSVersion = ""

	return platform
}
