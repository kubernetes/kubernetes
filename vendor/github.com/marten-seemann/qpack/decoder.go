package qpack

import (
	"bytes"
	"errors"
	"fmt"
	"sync"

	"golang.org/x/net/http2/hpack"
)

// A decodingError is something the spec defines as a decoding error.
type decodingError struct {
	err error
}

func (de decodingError) Error() string {
	return fmt.Sprintf("decoding error: %v", de.err)
}

// An invalidIndexError is returned when an encoder references a table
// entry before the static table or after the end of the dynamic table.
type invalidIndexError int

func (e invalidIndexError) Error() string {
	return fmt.Sprintf("invalid indexed representation index %d", int(e))
}

var errNoDynamicTable = decodingError{errors.New("no dynamic table")}

// errNeedMore is an internal sentinel error value that means the
// buffer is truncated and we need to read more data before we can
// continue parsing.
var errNeedMore = errors.New("need more data")

// A Decoder is the decoding context for incremental processing of
// header blocks.
type Decoder struct {
	mutex sync.Mutex

	emitFunc func(f HeaderField)

	readRequiredInsertCount bool
	readDeltaBase           bool

	// buf is the unparsed buffer. It's only written to
	// saveBuf if it was truncated in the middle of a header
	// block. Because it's usually not owned, we can only
	// process it under Write.
	buf []byte // not owned; only valid during Write

	// saveBuf is previous data passed to Write which we weren't able
	// to fully parse before. Unlike buf, we own this data.
	saveBuf bytes.Buffer
}

// NewDecoder returns a new decoder
// The emitFunc will be called for each valid field parsed,
// in the same goroutine as calls to Write, before Write returns.
func NewDecoder(emitFunc func(f HeaderField)) *Decoder {
	return &Decoder{emitFunc: emitFunc}
}

func (d *Decoder) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}

	d.mutex.Lock()
	n, err := d.writeLocked(p)
	d.mutex.Unlock()
	return n, err
}

func (d *Decoder) writeLocked(p []byte) (int, error) {
	// Only copy the data if we have to. Optimistically assume
	// that p will contain a complete header block.
	if d.saveBuf.Len() == 0 {
		d.buf = p
	} else {
		d.saveBuf.Write(p)
		d.buf = d.saveBuf.Bytes()
		d.saveBuf.Reset()
	}

	if err := d.decode(); err != nil {
		if err != errNeedMore {
			return 0, err
		}
		// TODO: limit the size of the buffer
		d.saveBuf.Write(d.buf)
	}
	return len(p), nil
}

// DecodeFull decodes an entire block.
func (d *Decoder) DecodeFull(p []byte) ([]HeaderField, error) {
	if len(p) == 0 {
		return []HeaderField{}, nil
	}

	d.mutex.Lock()
	defer d.mutex.Unlock()

	saveFunc := d.emitFunc
	defer func() { d.emitFunc = saveFunc }()

	var hf []HeaderField
	d.emitFunc = func(f HeaderField) { hf = append(hf, f) }
	if _, err := d.writeLocked(p); err != nil {
		return nil, err
	}
	if err := d.Close(); err != nil {
		return nil, err
	}
	return hf, nil
}

// Close declares that the decoding is complete and resets the Decoder
// to be reused again for a new header block. If there is any remaining
// data in the decoder's buffer, Close returns an error.
func (d *Decoder) Close() error {
	if d.saveBuf.Len() > 0 {
		d.saveBuf.Reset()
		return decodingError{errors.New("truncated headers")}
	}
	d.readRequiredInsertCount = false
	d.readDeltaBase = false
	return nil
}

func (d *Decoder) decode() error {
	if !d.readRequiredInsertCount {
		requiredInsertCount, rest, err := readVarInt(8, d.buf)
		if err != nil {
			return err
		}
		d.readRequiredInsertCount = true
		if requiredInsertCount != 0 {
			return decodingError{errors.New("expected Required Insert Count to be zero")}
		}
		d.buf = rest
	}
	if !d.readDeltaBase {
		base, rest, err := readVarInt(7, d.buf)
		if err != nil {
			return err
		}
		d.readDeltaBase = true
		if base != 0 {
			return decodingError{errors.New("expected Base to be zero")}
		}
		d.buf = rest
	}
	if len(d.buf) == 0 {
		return errNeedMore
	}

	for len(d.buf) > 0 {
		b := d.buf[0]
		var err error
		switch {
		case b&0x80 > 0: // 1xxxxxxx
			err = d.parseIndexedHeaderField()
		case b&0xc0 == 0x40: // 01xxxxxx
			err = d.parseLiteralHeaderField()
		case b&0xe0 == 0x20: // 001xxxxx
			err = d.parseLiteralHeaderFieldWithoutNameReference()
		default:
			err = fmt.Errorf("unexpected type byte: %#x", b)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (d *Decoder) parseIndexedHeaderField() error {
	buf := d.buf
	if buf[0]&0x40 == 0 {
		return errNoDynamicTable
	}
	index, buf, err := readVarInt(6, buf)
	if err != nil {
		return err
	}
	hf, ok := d.at(index)
	if !ok {
		return decodingError{invalidIndexError(index)}
	}
	d.emitFunc(hf)
	d.buf = buf
	return nil
}

func (d *Decoder) parseLiteralHeaderField() error {
	buf := d.buf
	if buf[0]&0x20 > 0 || buf[0]&0x10 == 0 {
		return errNoDynamicTable
	}
	index, buf, err := readVarInt(4, buf)
	if err != nil {
		return err
	}
	hf, ok := d.at(index)
	if !ok {
		return decodingError{invalidIndexError(index)}
	}
	if len(buf) == 0 {
		return errNeedMore
	}
	usesHuffman := buf[0]&0x80 > 0
	val, buf, err := d.readString(buf, 7, usesHuffman)
	if err != nil {
		return err
	}
	hf.Value = val
	d.emitFunc(hf)
	d.buf = buf
	return nil
}

func (d *Decoder) parseLiteralHeaderFieldWithoutNameReference() error {
	buf := d.buf
	usesHuffmanForName := buf[0]&0x8 > 0
	name, buf, err := d.readString(buf, 3, usesHuffmanForName)
	if err != nil {
		return err
	}
	if len(buf) == 0 {
		return errNeedMore
	}
	usesHuffmanForVal := buf[0]&0x80 > 0
	val, buf, err := d.readString(buf, 7, usesHuffmanForVal)
	if err != nil {
		return err
	}
	d.emitFunc(HeaderField{Name: name, Value: val})
	d.buf = buf
	return nil
}

func (d *Decoder) readString(buf []byte, n uint8, usesHuffman bool) (string, []byte, error) {
	l, buf, err := readVarInt(n, buf)
	if err != nil {
		return "", nil, err
	}
	if uint64(len(buf)) < l {
		return "", nil, errNeedMore
	}
	var val string
	if usesHuffman {
		var err error
		val, err = hpack.HuffmanDecodeToString(buf[:l])
		if err != nil {
			return "", nil, err
		}
	} else {
		val = string(buf[:l])
	}
	buf = buf[l:]
	return val, buf, nil
}

func (d *Decoder) at(i uint64) (hf HeaderField, ok bool) {
	if i >= uint64(len(staticTableEntries)) {
		return
	}
	return staticTableEntries[i], true
}
