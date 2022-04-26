package packfile

import (
	"bytes"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

// See https://github.com/jelmer/dulwich/blob/master/dulwich/pack.py and
// https://github.com/tarruda/node-git-core/blob/master/src/js/delta.js
// for more info

const (
	// Standard chunk size used to generate fingerprints
	s = 16

	// https://github.com/git/git/blob/f7466e94375b3be27f229c78873f0acf8301c0a5/diff-delta.c#L428
	// Max size of a copy operation (64KB)
	maxCopySize = 64 * 1024
)

// GetDelta returns an EncodedObject of type OFSDeltaObject. Base and Target object,
// will be loaded into memory to be able to create the delta object.
// To generate target again, you will need the obtained object and "base" one.
// Error will be returned if base or target object cannot be read.
func GetDelta(base, target plumbing.EncodedObject) (plumbing.EncodedObject, error) {
	return getDelta(new(deltaIndex), base, target)
}

func getDelta(index *deltaIndex, base, target plumbing.EncodedObject) (o plumbing.EncodedObject, err error) {
	br, err := base.Reader()
	if err != nil {
		return nil, err
	}

	defer ioutil.CheckClose(br, &err)

	tr, err := target.Reader()
	if err != nil {
		return nil, err
	}

	defer ioutil.CheckClose(tr, &err)

	bb := bufPool.Get().(*bytes.Buffer)
	defer bufPool.Put(bb)
	bb.Reset()

	_, err = bb.ReadFrom(br)
	if err != nil {
		return nil, err
	}

	tb := bufPool.Get().(*bytes.Buffer)
	defer bufPool.Put(tb)
	tb.Reset()

	_, err = tb.ReadFrom(tr)
	if err != nil {
		return nil, err
	}

	db := diffDelta(index, bb.Bytes(), tb.Bytes())
	delta := &plumbing.MemoryObject{}
	_, err = delta.Write(db)
	if err != nil {
		return nil, err
	}

	delta.SetSize(int64(len(db)))
	delta.SetType(plumbing.OFSDeltaObject)

	return delta, nil
}

// DiffDelta returns the delta that transforms src into tgt.
func DiffDelta(src, tgt []byte) []byte {
	return diffDelta(new(deltaIndex), src, tgt)
}

func diffDelta(index *deltaIndex, src []byte, tgt []byte) []byte {
	buf := bufPool.Get().(*bytes.Buffer)
	defer bufPool.Put(buf)
	buf.Reset()
	buf.Write(deltaEncodeSize(len(src)))
	buf.Write(deltaEncodeSize(len(tgt)))

	if len(index.entries) == 0 {
		index.init(src)
	}

	ibuf := bufPool.Get().(*bytes.Buffer)
	defer bufPool.Put(ibuf)
	ibuf.Reset()
	for i := 0; i < len(tgt); i++ {
		offset, l := index.findMatch(src, tgt, i)

		if l == 0 {
			// couldn't find a match, just write the current byte and continue
			ibuf.WriteByte(tgt[i])
		} else if l < 0 {
			// src is less than blksz, copy the rest of the target to avoid
			// calls to findMatch
			for ; i < len(tgt); i++ {
				ibuf.WriteByte(tgt[i])
			}
		} else if l < s {
			// remaining target is less than blksz, copy what's left of it
			// and avoid calls to findMatch
			for j := i; j < i+l; j++ {
				ibuf.WriteByte(tgt[j])
			}
			i += l - 1
		} else {
			encodeInsertOperation(ibuf, buf)

			rl := l
			aOffset := offset
			for rl > 0 {
				if rl < maxCopySize {
					buf.Write(encodeCopyOperation(aOffset, rl))
					break
				}

				buf.Write(encodeCopyOperation(aOffset, maxCopySize))
				rl -= maxCopySize
				aOffset += maxCopySize
			}

			i += l - 1
		}
	}

	encodeInsertOperation(ibuf, buf)

	// buf.Bytes() is only valid until the next modifying operation on the buffer. Copy it.
	return append([]byte{}, buf.Bytes()...)
}

func encodeInsertOperation(ibuf, buf *bytes.Buffer) {
	if ibuf.Len() == 0 {
		return
	}

	b := ibuf.Bytes()
	s := ibuf.Len()
	o := 0
	for {
		if s <= 127 {
			break
		}
		buf.WriteByte(byte(127))
		buf.Write(b[o : o+127])
		s -= 127
		o += 127
	}
	buf.WriteByte(byte(s))
	buf.Write(b[o : o+s])

	ibuf.Reset()
}

func deltaEncodeSize(size int) []byte {
	var ret []byte
	c := size & 0x7f
	size >>= 7
	for {
		if size == 0 {
			break
		}

		ret = append(ret, byte(c|0x80))
		c = size & 0x7f
		size >>= 7
	}
	ret = append(ret, byte(c))

	return ret
}

func encodeCopyOperation(offset, length int) []byte {
	code := 0x80
	var opcodes []byte

	var i uint
	for i = 0; i < 4; i++ {
		f := 0xff << (i * 8)
		if offset&f != 0 {
			opcodes = append(opcodes, byte(offset&f>>(i*8)))
			code |= 0x01 << i
		}
	}

	for i = 0; i < 3; i++ {
		f := 0xff << (i * 8)
		if length&f != 0 {
			opcodes = append(opcodes, byte(length&f>>(i*8)))
			code |= 0x10 << i
		}
	}

	return append([]byte{byte(code)}, opcodes...)
}
