package roaring

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/tinylib/msgp/msgp"
)

// writeTo for runContainer16 follows this
// spec: https://github.com/RoaringBitmap/RoaringFormatSpec
//
func (b *runContainer16) writeTo(stream io.Writer) (int, error) {
	buf := make([]byte, 2+4*len(b.iv))
	binary.LittleEndian.PutUint16(buf[0:], uint16(len(b.iv)))
	for i, v := range b.iv {
		binary.LittleEndian.PutUint16(buf[2+i*4:], v.start)
		binary.LittleEndian.PutUint16(buf[2+2+i*4:], v.last-v.start)
	}
	return stream.Write(buf)
}

func (b *runContainer32) writeToMsgpack(stream io.Writer) (int, error) {
	bts, err := b.MarshalMsg(nil)
	if err != nil {
		return 0, err
	}
	return stream.Write(bts)
}

func (b *runContainer16) writeToMsgpack(stream io.Writer) (int, error) {
	bts, err := b.MarshalMsg(nil)
	if err != nil {
		return 0, err
	}
	return stream.Write(bts)
}

func (b *runContainer32) readFromMsgpack(stream io.Reader) (int, error) {
	err := msgp.Decode(stream, b)
	return 0, err
}

func (b *runContainer16) readFromMsgpack(stream io.Reader) (int, error) {
	err := msgp.Decode(stream, b)
	return 0, err
}

func (b *runContainer16) readFrom(stream io.Reader) (int, error) {
	b.iv = b.iv[:0]
	b.card = 0
	var numRuns uint16
	err := binary.Read(stream, binary.LittleEndian, &numRuns)
	if err != nil {
		return 0, err
	}
	encRun := make([]uint16, 2*numRuns)
	by := make([]byte, 4*numRuns)
	err = binary.Read(stream, binary.LittleEndian, &by)
	if err != nil {
		return 0, err
	}
	for i := range encRun {
		if len(by) < 2 {
			panic("insufficient/odd number of stored bytes, corrupted stream detected")
		}
		encRun[i] = binary.LittleEndian.Uint16(by)
		by = by[2:]
	}
	nr := int(numRuns)
	for i := 0; i < nr; i++ {
		if i > 0 && b.iv[i-1].last >= encRun[i*2] {
			panic(fmt.Errorf("error: stored runContainer had runs that were not in sorted order!! (b.iv[i-1=%v].last = %v >= encRun[i=%v] = %v)", i-1, b.iv[i-1].last, i, encRun[i*2]))
		}
		b.iv = append(b.iv, interval16{start: encRun[i*2], last: encRun[i*2] + encRun[i*2+1]})
		b.card += int64(encRun[i*2+1]) + 1
	}
	return 0, err
}
