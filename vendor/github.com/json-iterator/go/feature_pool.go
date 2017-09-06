package jsoniter

import (
	"io"
)

// IteratorPool a thread safe pool of iterators with same configuration
type IteratorPool interface {
	BorrowIterator(data []byte) *Iterator
	ReturnIterator(iter *Iterator)
}

// StreamPool a thread safe pool of streams with same configuration
type StreamPool interface {
	BorrowStream(writer io.Writer) *Stream
	ReturnStream(stream *Stream)
}

func (cfg *frozenConfig) BorrowStream(writer io.Writer) *Stream {
	select {
	case stream := <-cfg.streamPool:
		stream.Reset(writer)
		return stream
	default:
		return NewStream(cfg, writer, 512)
	}
}

func (cfg *frozenConfig) ReturnStream(stream *Stream) {
	stream.Error = nil
	select {
	case cfg.streamPool <- stream:
		return
	default:
		return
	}
}

func (cfg *frozenConfig) BorrowIterator(data []byte) *Iterator {
	select {
	case iter := <-cfg.iteratorPool:
		iter.ResetBytes(data)
		return iter
	default:
		return ParseBytes(cfg, data)
	}
}

func (cfg *frozenConfig) ReturnIterator(iter *Iterator) {
	iter.Error = nil
	select {
	case cfg.iteratorPool <- iter:
		return
	default:
		return
	}
}
