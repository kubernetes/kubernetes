package tsm1_test

import (
	"bytes"
	"io"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"github.com/dgryski/go-bitstream"
	"github.com/influxdata/influxdb/tsdb/engine/tsm1"
)

func TestBitStreamEOF(t *testing.T) {
	br := tsm1.NewBitReader([]byte("0"))

	b, err := br.ReadBits(8)
	if b != '0' {
		t.Error("ReadBits(8) didn't return first byte")
	}

	if _, err := br.ReadBits(8); err != io.EOF {
		t.Error("ReadBits(8) on empty string didn't return EOF")
	}

	// 0 = 0b00110000
	br = tsm1.NewBitReader([]byte("0"))

	buf := bytes.NewBuffer(nil)
	bw := bitstream.NewWriter(buf)

	for i := 0; i < 4; i++ {
		bit, err := br.ReadBit()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Error("GetBit returned error err=", err.Error())
			return
		}
		bw.WriteBit(bitstream.Bit(bit))
	}

	bw.Flush(bitstream.One)

	err = bw.WriteByte(0xAA)
	if err != nil {
		t.Error("unable to WriteByte")
	}

	c := buf.Bytes()

	if len(c) != 2 || c[1] != 0xAA || c[0] != 0x3f {
		t.Error("bad return from 4 read bytes")
	}

	_, err = tsm1.NewBitReader([]byte("")).ReadBit()
	if err != io.EOF {
		t.Error("ReadBit on empty string didn't return EOF")
	}
}

func TestBitStream(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	br := tsm1.NewBitReader([]byte("hello"))
	bw := bitstream.NewWriter(buf)

	for {
		bit, err := br.ReadBit()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Error("GetBit returned error err=", err.Error())
			return
		}
		bw.WriteBit(bitstream.Bit(bit))
	}

	s := buf.String()

	if s != "hello" {
		t.Error("expected 'hello', got=", []byte(s))
	}
}

func TestByteStream(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	br := tsm1.NewBitReader([]byte("hello"))
	bw := bitstream.NewWriter(buf)

	for i := 0; i < 3; i++ {
		bit, err := br.ReadBit()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Error("GetBit returned error err=", err.Error())
			return
		}
		bw.WriteBit(bitstream.Bit(bit))
	}

	for i := 0; i < 3; i++ {
		byt, err := br.ReadBits(8)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Error("ReadBits(8) returned error err=", err.Error())
			return
		}
		bw.WriteByte(byte(byt))
	}

	u, err := br.ReadBits(13)

	if err != nil {
		t.Error("ReadBits returned error err=", err.Error())
		return
	}

	bw.WriteBits(u, 13)

	bw.WriteBits(('!'<<12)|('.'<<4)|0x02, 20)
	// 0x2f == '/'
	bw.Flush(bitstream.One)

	s := buf.String()

	if s != "hello!./" {
		t.Errorf("expected 'hello!./', got=%x", []byte(s))
	}
}

// Ensure bit reader can read random bits written to a stream.
func TestBitReader_Quick(t *testing.T) {
	if err := quick.Check(func(values []uint64, nbits []uint) bool {
		// Limit nbits to 64.
		for i := 0; i < len(values) && i < len(nbits); i++ {
			nbits[i] = (nbits[i] % 64) + 1
			values[i] = values[i] & (math.MaxUint64 >> (64 - nbits[i]))
		}

		// Write bits to a buffer.
		var buf bytes.Buffer
		w := bitstream.NewWriter(&buf)
		for i := 0; i < len(values) && i < len(nbits); i++ {
			w.WriteBits(values[i], int(nbits[i]))
		}
		w.Flush(bitstream.Zero)

		// Read bits from the buffer.
		r := tsm1.NewBitReader(buf.Bytes())
		for i := 0; i < len(values) && i < len(nbits); i++ {
			v, err := r.ReadBits(nbits[i])
			if err != nil {
				t.Errorf("unexpected error(%d): %s", i, err)
				return false
			} else if v != values[i] {
				t.Errorf("value mismatch(%d): got=%d, exp=%d (nbits=%d)", i, v, values[i], nbits[i])
				return false
			}
		}

		return true
	}, &quick.Config{
		Values: func(a []reflect.Value, rand *rand.Rand) {
			a[0], _ = quick.Value(reflect.TypeOf([]uint64{}), rand)
			a[1], _ = quick.Value(reflect.TypeOf([]uint{}), rand)
		},
	}); err != nil {
		t.Fatal(err)
	}
}
