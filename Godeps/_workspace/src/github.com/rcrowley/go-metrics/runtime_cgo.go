// +build cgo
// +build !appengine

package metrics

import "runtime"

func numCgoCall() int64 {
	return runtime.NumCgoCall()
}
