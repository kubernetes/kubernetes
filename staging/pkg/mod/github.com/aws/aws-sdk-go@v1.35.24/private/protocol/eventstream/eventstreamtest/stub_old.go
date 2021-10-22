// +build !go1.9

package eventstreamtest

import "testing"

var getHelper = func(t testing.TB) func() {
	return nopHelper
}

func nopHelper() {}
