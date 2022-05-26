package object

import (
	"bufio"
	"sync"
)

var bufPool = sync.Pool{
	New: func() interface{} {
		return bufio.NewReader(nil)
	},
}
