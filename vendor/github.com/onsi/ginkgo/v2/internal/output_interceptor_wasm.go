//go:build wasm

package internal

import "os"

// dupStdout returns nil on WASM since output interception is not supported.
func dupStdout() *os.File {
	return nil
}

func NewOutputInterceptor() OutputInterceptor {
	return &NoopOutputInterceptor{}
}
