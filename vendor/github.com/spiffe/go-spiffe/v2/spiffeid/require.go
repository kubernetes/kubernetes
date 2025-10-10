package spiffeid

import (
	"net/url"
)

// RequireFromPath is similar to FromPath except that instead of returning an
// error on malformed input, it panics. It should only be used when the input
// is statically verifiable.
func RequireFromPath(td TrustDomain, path string) ID {
	id, err := FromPath(td, path)
	panicOnErr(err)
	return id
}

// RequireFromPathf is similar to FromPathf except that instead of returning an
// error on malformed input, it panics. It should only be used when the input
// is statically verifiable.
func RequireFromPathf(td TrustDomain, format string, args ...interface{}) ID {
	id, err := FromPathf(td, format, args...)
	panicOnErr(err)
	return id
}

// RequireFromSegments is similar to FromSegments except that instead of
// returning an error on malformed input, it panics. It should only be used
// when the input is statically verifiable.
func RequireFromSegments(td TrustDomain, segments ...string) ID {
	id, err := FromSegments(td, segments...)
	panicOnErr(err)
	return id
}

// RequireFromString is similar to FromString except that instead of returning
// an error on malformed input, it panics. It should only be used when the
// input is statically verifiable.
func RequireFromString(s string) ID {
	id, err := FromString(s)
	panicOnErr(err)
	return id
}

// RequireFromStringf is similar to FromStringf except that instead of
// returning an error on malformed input, it panics. It should only be used
// when the input is statically verifiable.
func RequireFromStringf(format string, args ...interface{}) ID {
	id, err := FromStringf(format, args...)
	panicOnErr(err)
	return id
}

// RequireFromURI is similar to FromURI except that instead of returning an
// error on malformed input, it panics. It should only be used when the input is
// statically verifiable.
func RequireFromURI(uri *url.URL) ID {
	id, err := FromURI(uri)
	panicOnErr(err)
	return id
}

// RequireTrustDomainFromString is similar to TrustDomainFromString except that
// instead of returning an error on malformed input, it panics. It should only
// be used when the input is statically verifiable.
func RequireTrustDomainFromString(s string) TrustDomain {
	td, err := TrustDomainFromString(s)
	panicOnErr(err)
	return td
}

// RequireTrustDomainFromURI is similar to TrustDomainFromURI except that
// instead of returning an error on malformed input, it panics. It should only
// be used when the input is statically verifiable.
func RequireTrustDomainFromURI(uri *url.URL) TrustDomain {
	td, err := TrustDomainFromURI(uri)
	panicOnErr(err)
	return td
}

// RequireFormatPath builds a path by formatting the given formatting string
// with the given args (i.e. fmt.Sprintf). The resulting path must be valid or
// the function panics. It should only be used when the input is statically
// verifiable.
func RequireFormatPath(format string, args ...interface{}) string {
	path, err := FormatPath(format, args...)
	panicOnErr(err)
	return path
}

// RequireJoinPathSegments joins one or more path segments into a slash separated
// path. Segments cannot contain slashes. The resulting path must be valid or
// the function panics. It should only be used when the input is statically
// verifiable.
func RequireJoinPathSegments(segments ...string) string {
	path, err := JoinPathSegments(segments...)
	panicOnErr(err)
	return path
}

func panicOnErr(err error) {
	if err != nil {
		panic(err)
	}
}
