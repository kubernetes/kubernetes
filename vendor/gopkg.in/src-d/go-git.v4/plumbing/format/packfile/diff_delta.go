package packfile

import (
	"bytes"
	"hash/adler32"
	"io/ioutil"

	"gopkg.in/src-d/go-git.v4/plumbing"
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
	br, err := base.Reader()
	if err != nil {
		return nil, err
	}
	tr, err := target.Reader()
	if err != nil {
		return nil, err
	}

	bb, err := ioutil.ReadAll(br)
	if err != nil {
		return nil, err
	}

	tb, err := ioutil.ReadAll(tr)
	if err != nil {
		return nil, err
	}

	db := DiffDelta(bb, tb)
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
func DiffDelta(src []byte, tgt []byte) []byte {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	buf.Write(deltaEncodeSize(len(src)))
	buf.Write(deltaEncodeSize(len(tgt)))

	sindex := initMatch(src)

	ibuf := bufPool.Get().(*bytes.Buffer)
	ibuf.Reset()
	for i := 0; i < len(tgt); i++ {
		offset, l := findMatch(src, tgt, sindex, i)

		if l < s {
			ibuf.WriteByte(tgt[i])
		} else {
			encodeInsertOperation(ibuf, buf)

			rl := l
			aOffset := offset
			for {
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
	bytes := buf.Bytes()

	bufPool.Put(buf)
	bufPool.Put(ibuf)

	return bytes
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

func initMatch(src []byte) map[uint32]int {
	i := 0
	index := make(map[uint32]int)
	for {
		if i+s > len(src) {
			break
		}

		ch := adler32.Checksum(src[i : i+s])
		index[ch] = i
		i += s
	}

	return index
}

func findMatch(src, tgt []byte, sindex map[uint32]int, tgtOffset int) (srcOffset, l int) {
	if len(tgt) >= tgtOffset+s {
		ch := adler32.Checksum(tgt[tgtOffset : tgtOffset+s])
		var ok bool
		srcOffset, ok = sindex[ch]
		if !ok {
			return
		}

		l = matchLength(src, tgt, tgtOffset, srcOffset)
	}

	return
}

func matchLength(src, tgt []byte, otgt, osrc int) int {
	l := 0
	for {
		if (osrc >= len(src) || otgt >= len(tgt)) || src[osrc] != tgt[otgt] {
			break
		}

		l++
		osrc++
		otgt++
	}

	return l
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
