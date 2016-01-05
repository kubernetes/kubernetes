// +build !go1.2

package gcfg

type textUnmarshaler interface {
	UnmarshalText(text []byte) error
}
