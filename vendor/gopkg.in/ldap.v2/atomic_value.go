// +build go1.4

package ldap

import (
	"sync/atomic"
)

// For compilers that support it, we just use the underlying sync/atomic.Value
// type.
type atomicValue struct {
	atomic.Value
}
