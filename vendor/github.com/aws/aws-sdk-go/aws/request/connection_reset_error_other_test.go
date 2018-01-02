// +build appengine plan9

package request_test

import (
	"errors"
)

var stubConnectionResetError = errors.New("connection reset")
