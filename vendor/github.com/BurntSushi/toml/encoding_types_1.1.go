// +build !go1.2

package toml

// These interfaces were introduced in Go 1.2, so we add them manually when
// compiling for Go 1.1.

// TextMarshaler is a synonym for encoding.TextMarshaler. It is defined here
// so that Go 1.1 can be supported.
type TextMarshaler interface {
	MarshalText() (text []byte, err error)
}

// TextUnmarshaler is a synonym for encoding.TextUnmarshaler. It is defined
// here so that Go 1.1 can be supported.
type TextUnmarshaler interface {
	UnmarshalText(text []byte) error
}
