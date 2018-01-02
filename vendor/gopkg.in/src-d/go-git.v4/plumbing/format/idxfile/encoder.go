package idxfile

import (
	"crypto/sha1"
	"hash"
	"io"
	"sort"

	"gopkg.in/src-d/go-git.v4/utils/binary"
)

// Encoder writes Idxfile structs to an output stream.
type Encoder struct {
	io.Writer
	hash hash.Hash
}

// NewEncoder returns a new stream encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	h := sha1.New()
	mw := io.MultiWriter(w, h)
	return &Encoder{mw, h}
}

// Encode encodes an Idxfile to the encoder writer.
func (e *Encoder) Encode(idx *Idxfile) (int, error) {
	idx.Entries.Sort()

	flow := []func(*Idxfile) (int, error){
		e.encodeHeader,
		e.encodeFanout,
		e.encodeHashes,
		e.encodeCRC32,
		e.encodeOffsets,
		e.encodeChecksums,
	}

	sz := 0
	for _, f := range flow {
		i, err := f(idx)
		sz += i

		if err != nil {
			return sz, err
		}
	}

	return sz, nil
}

func (e *Encoder) encodeHeader(idx *Idxfile) (int, error) {
	c, err := e.Write(idxHeader)
	if err != nil {
		return c, err
	}

	return c + 4, binary.WriteUint32(e, idx.Version)
}

func (e *Encoder) encodeFanout(idx *Idxfile) (int, error) {
	fanout := idx.calculateFanout()
	for _, c := range fanout {
		if err := binary.WriteUint32(e, c); err != nil {
			return 0, err
		}
	}

	return 1024, nil
}

func (e *Encoder) encodeHashes(idx *Idxfile) (int, error) {
	sz := 0
	for _, ent := range idx.Entries {
		i, err := e.Write(ent.Hash[:])
		sz += i

		if err != nil {
			return sz, err
		}
	}

	return sz, nil
}

func (e *Encoder) encodeCRC32(idx *Idxfile) (int, error) {
	sz := 0
	for _, ent := range idx.Entries {
		err := binary.Write(e, ent.CRC32)
		sz += 4

		if err != nil {
			return sz, err
		}
	}

	return sz, nil
}

func (e *Encoder) encodeOffsets(idx *Idxfile) (int, error) {
	sz := 0

	var o64bits []uint64
	for _, ent := range idx.Entries {
		o := ent.Offset
		if o > offsetLimit {
			o64bits = append(o64bits, o)
			o = offsetLimit + uint64(len(o64bits))
		}

		if err := binary.WriteUint32(e, uint32(o)); err != nil {
			return sz, err
		}

		sz += 4
	}

	for _, o := range o64bits {
		if err := binary.WriteUint64(e, o); err != nil {
			return sz, err
		}

		sz += 8
	}

	return sz, nil
}

func (e *Encoder) encodeChecksums(idx *Idxfile) (int, error) {
	if _, err := e.Write(idx.PackfileChecksum[:]); err != nil {
		return 0, err
	}

	copy(idx.IdxChecksum[:], e.hash.Sum(nil)[:20])
	if _, err := e.Write(idx.IdxChecksum[:]); err != nil {
		return 0, err
	}

	return 40, nil
}

// EntryList implements sort.Interface allowing sorting in increasing order.
type EntryList []*Entry

func (p EntryList) Len() int           { return len(p) }
func (p EntryList) Less(i, j int) bool { return p[i].Hash.String() < p[j].Hash.String() }
func (p EntryList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p EntryList) Sort()              { sort.Sort(p) }
