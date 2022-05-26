package packfile

import (
	"bytes"
	"compress/zlib"
	"io"
	"sync"

	"github.com/go-git/go-git/v5/plumbing/storer"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

var signature = []byte{'P', 'A', 'C', 'K'}

const (
	// VersionSupported is the packfile version supported by this package
	VersionSupported uint32 = 2

	firstLengthBits = uint8(4)   // the first byte into object header has 4 bits to store the length
	lengthBits      = uint8(7)   // subsequent bytes has 7 bits to store the length
	maskFirstLength = 15         // 0000 1111
	maskContinue    = 0x80       // 1000 0000
	maskLength      = uint8(127) // 0111 1111
	maskType        = uint8(112) // 0111 0000
)

// UpdateObjectStorage updates the storer with the objects in the given
// packfile.
func UpdateObjectStorage(s storer.Storer, packfile io.Reader) error {
	if pw, ok := s.(storer.PackfileWriter); ok {
		return WritePackfileToObjectStorage(pw, packfile)
	}

	p, err := NewParserWithStorage(NewScanner(packfile), s)
	if err != nil {
		return err
	}

	_, err = p.Parse()
	return err
}

// WritePackfileToObjectStorage writes all the packfile objects into the given
// object storage.
func WritePackfileToObjectStorage(
	sw storer.PackfileWriter,
	packfile io.Reader,
) (err error) {
	w, err := sw.PackfileWriter()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)

	var n int64
	n, err = io.Copy(w, packfile)
	if err == nil && n == 0 {
		return ErrEmptyPackfile
	}

	return err
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return bytes.NewBuffer(nil)
	},
}

var zlibInitBytes = []byte{0x78, 0x9c, 0x01, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01}

var zlibReaderPool = sync.Pool{
	New: func() interface{} {
		r, _ := zlib.NewReader(bytes.NewReader(zlibInitBytes))
		return r
	},
}
