package logrus

import (
	"bytes"
	"sync"
)

var bufferPool BufferPool = &defaultPool{
	pool: &sync.Pool{
		New: func() any {
			return new(bytes.Buffer)
		},
	},
}

type BufferPool interface {
	Put(*bytes.Buffer)
	Get() *bytes.Buffer
}

type defaultPool struct {
	pool *sync.Pool
}

func (p *defaultPool) Put(buf *bytes.Buffer) {
	p.pool.Put(buf)
}

func (p *defaultPool) Get() *bytes.Buffer {
	return p.pool.Get().(*bytes.Buffer)
}

// SetBufferPool allows to replace the default logrus buffer pool
// to better meet the specific needs of an application.
func SetBufferPool(bp BufferPool) {
	bufferPool = bp
}
