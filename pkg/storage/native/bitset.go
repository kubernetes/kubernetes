package native

import (
	"fmt"
	"github.com/gogo/protobuf/io"
	"sync"
)

// If we change, need to implement Unmarshal support for variables sizes
const chunkSize = 8192

type BitSet struct {
	mutex sync.Mutex

	chunks map[uint64]*bitsetChunk
}

type BitSetSnapshot struct {
	bitset *BitSet
	chunks map[uint64]*bitsetChunk
}

type bitsetChunk struct {
	bits          [chunkSize / 8]byte
	snapshotCount uint32
}

func NewBitSet() *BitSet {
	b := &BitSet{}
	return b
}

func (b *BitSet) Get(k uint64) bool {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	base := k / chunkSize
	chunk := b.chunks[base]
	if chunk == nil {
		return false
	}
	offset := (k - base) / 8
	mask := byte(1) << ((k - base) % 8)
	return (chunk.bits[offset] & mask) != byte(0)
}

func (b *BitSet) Put(k uint64, v bool) {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	base := k / chunkSize

	chunk := b.chunks[base]
	if chunk == nil {
		if v == false {
			return
		}
		if b.chunks == nil {
			b.chunks = make(map[uint64]*bitsetChunk)
		}
		chunk = &bitsetChunk{}
		b.chunks[base] = chunk
	} else {
		snapshotCount := chunk.snapshotCount
		if snapshotCount != 0 {
			mutableChunk := &bitsetChunk{}
			mutableChunk.bits = chunk.bits
			mutableChunk.snapshotCount = 0

			chunk = mutableChunk
			b.chunks[base] = chunk
		}
	}

	offset := (k - base) / 8
	mask := byte(1) << ((k - base) % 8)
	if v {
		chunk.bits[offset] |= mask
	} else {
		chunk.bits[offset] &= ^mask
	}
}

func (b *BitSet) Snapshot() *BitSetSnapshot {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	snapshot := &BitSetSnapshot{
		bitset: b,
		chunks: make(map[uint64]*bitsetChunk),
	}

	for k, v := range b.chunks {
		snapshot.chunks[k] = v
		v.snapshotCount++
	}

	return snapshot
}

func (b *BitSetSnapshot) Marshal(w io.WriteCloser) error {
	info := &BitSetInfo{
		ChunkCount: uint32(len(b.chunks)),
	}
	if err := w.WriteMsg(info); err != nil {
		return err
	}

	chunkCount := uint32(0)
	for min, chunk := range b.chunks {
		chunkData := &BitSetChunkInfo{
			Min:  min,
			Bits: chunk.bits[:],
		}
		if err := w.WriteMsg(chunkData); err != nil {
			return err
		}
		chunkCount++
	}

	if chunkCount != info.ChunkCount {
		return fmt.Errorf("concurent modification detected")
	}

	return nil
}

// Release is invoked when we are finished with the snapshot.
func (b *BitSetSnapshot) Release() {
	b.bitset.mutex.Lock()
	defer b.bitset.mutex.Unlock()

	for _, v := range b.chunks {
		v.snapshotCount--
	}
}

func (b *BitSet) Unmarshal(r io.ReadCloser) error {
	info := &BitSetInfo{}
	if err := r.ReadMsg(info); err != nil {
		return err
	}

	b.chunks = make(map[uint64]*bitsetChunk)
	for i := uint32(0); i < info.ChunkCount; i++ {
		chunkData := &BitSetChunkInfo{}
		if err := r.ReadMsg(chunkData); err != nil {
			return err
		}

		if len(chunkData.Bits) != (chunkSize / 8) {
			// TODO: Support this when needed
			return fmt.Errorf("chunk size change not yet implemented")
		}

		chunk := &bitsetChunk{}
		copy(chunk.bits[:], chunkData.Bits)

		b.chunks[chunkData.Min] = chunk
	}

	return nil
}
