package etcd

import (
	"testing"
)

type Foo struct{}
type Bar struct {
	one string
	two int
}

// Tests that logs don't panic with arbitrary interfaces
func TestDebug(t *testing.T) {
	f := &Foo{}
	b := &Bar{"asfd", 3}
	for _, test := range []interface{}{
		1234,
		"asdf",
		f,
		b,
	} {
		logger.Debug(test)
		logger.Debugf("something, %s", test)
		logger.Warning(test)
		logger.Warningf("something, %s", test)
	}
}
