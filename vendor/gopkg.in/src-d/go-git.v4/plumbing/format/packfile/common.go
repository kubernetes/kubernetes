package packfile

import (
	"bytes"
	"io"
	"sync"

	"gopkg.in/src-d/go-git.v4/plumbing/storer"
	"gopkg.in/src-d/go-git.v4/utils/ioutil"
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

// UpdateObjectStorage updates the given storer.EncodedObjectStorer with the contents of the
// packfile.
func UpdateObjectStorage(s storer.EncodedObjectStorer, packfile io.Reader) error {
	if sw, ok := s.(storer.PackfileWriter); ok {
		return writePackfileToObjectStorage(sw, packfile)
	}

	stream := NewScanner(packfile)
	d, err := NewDecoder(stream, s)
	if err != nil {
		return err
	}

	_, err = d.Decode()
	return err
}

func writePackfileToObjectStorage(sw storer.PackfileWriter, packfile io.Reader) error {
	var err error
	w, err := sw.PackfileWriter()
	if err != nil {
		return err
	}

	defer ioutil.CheckClose(w, &err)
	_, err = io.Copy(w, packfile)
	return err
}

var bufPool = sync.Pool{
	New: func() interface{} {
		return bytes.NewBuffer(nil)
	},
}
