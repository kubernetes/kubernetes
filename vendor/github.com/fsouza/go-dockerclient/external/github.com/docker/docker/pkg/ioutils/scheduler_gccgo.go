// +build gccgo

package ioutils

import (
	"runtime"
)

func callSchedulerIfNecessary() {
	//allow or force Go scheduler to switch context, without explicitly
	//forcing this will make it hang when using gccgo implementation
	runtime.Gosched()
}
