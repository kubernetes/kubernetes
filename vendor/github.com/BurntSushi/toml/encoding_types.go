// +build go1.2

package toml

// In order to support Go 1.1, we define our own TextMarshaler and
// TextUnmarshaler types. For Go 1.2+, we just alias them with the
// standard library interfaces.

import (
	"encoding"
)

// TextMarshaler is a synonym for encoding.TextMarshaler. It is defined here
// so that Go 1.1 can be supported.
type TextMarshaler encoding.TextMarshaler

// TextUnmarshaler is a synonym for encoding.TextUnmarshaler. It is defined
// here so that Go 1.1 can be supported.
type TextUnmarshaler encoding.TextUnmarshaler
