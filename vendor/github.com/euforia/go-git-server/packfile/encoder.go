package packfile

import (
	"bufio"
	"compress/zlib"
	"crypto/sha1"
	"encoding/binary"
	"hash"
	"io"
	"log"

	"gopkg.in/src-d/go-git.v4/plumbing/storer"

	"gopkg.in/src-d/go-git.v4/plumbing"
)

const writerSize = 65519

type Encoder struct {
	cw *checksumWriter
	w  *bufio.Writer

	store storer.EncodedObjectStorer
}

func NewEncoder(w io.Writer, store storer.EncodedObjectStorer) *Encoder {
	enc := &Encoder{
		cw:    newSHA160checksumWriter(w),
		store: store,
	}
	enc.w = bufio.NewWriterSize(enc.cw, writerSize)
	return enc
}

// Encode walks all hashes collecting them all, writes the header, followed
// by the entries and then the footer.
func (enc *Encoder) Encode(hashes ...plumbing.Hash) ([]byte, error) {
	wlker := NewObjectWalker(enc.store)
	// Object map to hold uniques
	out := map[plumbing.Hash]plumbing.EncodedObject{}

	for _, h := range hashes {
		//h := plumbing.NewHash(want)
		err := wlker.Walk(h, func(obj plumbing.EncodedObject) error {
			out[obj.Hash()] = obj
			return nil
		})

		if err != nil {
			log.Println("ERR", h.String(), err)
		}
	}

	log.Printf("[upload-pack] Packfile header: objects=%d", len(out))
	if err := enc.writeHeader(len(out)); err != nil {
		return nil, err
	}

	if err := enc.writeEntries(out); err != nil {
		return nil, err
	}

	return enc.writeFooter()
}

func (enc *Encoder) writeHeader(objCount int) (err error) {
	if err = binary.Write(enc.w, binary.BigEndian, []byte("PACK")); err == nil {
		// packfile version
		if err = binary.Write(enc.w, binary.BigEndian, uint32(2)); err == nil {
			// object count
			if err = binary.Write(enc.w, binary.BigEndian, uint32(objCount)); err == nil {
				enc.w.Flush()
			}
		}
	}
	return err
}

func (enc *Encoder) writeFooter() ([]byte, error) {
	checksum := enc.cw.Sum()
	err := binary.Write(enc.w, binary.BigEndian, checksum)
	enc.w.Flush()

	return checksum, err
}

// write given objects
func (enc *Encoder) writeEntries(objs map[plumbing.Hash]plumbing.EncodedObject) error {
	// write all objects
	for _, o := range objs {
		if err := enc.writeEntry(o); err != nil {
			return err
		}
		enc.w.Flush()
	}

	return nil
}

func (enc *Encoder) writeEntry(o plumbing.EncodedObject) error {
	var t byte
	// Write type
	t |= 0x80
	switch o.Type() {
	case plumbing.CommitObject:
		t |= byte(plumbing.CommitObject) << 4
	case plumbing.TreeObject:
		t |= byte(plumbing.TreeObject) << 4
	case plumbing.BlobObject:
		t |= byte(plumbing.BlobObject) << 4
	case plumbing.TagObject:
		t |= byte(plumbing.TagObject) << 4
	}
	// Write size
	t |= byte(uint64(o.Size()) &^ 0xfffffffffffffff0)
	sz := o.Size() >> 4
	szb := make([]byte, 16)
	n := binary.PutUvarint(szb, uint64(sz))
	szb = szb[0:n]
	enc.w.Write(append([]byte{t}, szb...))

	// Compress data and write
	zw := zlib.NewWriter(enc.w)
	defer zw.Close()

	or, _ := o.Reader()
	defer or.Close()

	_, err := io.Copy(zw, or)
	if err == nil {
		zw.Flush()
	}

	return err
}

/*// WritePackFile to write with the given objects
func WritePackFile(objs map[plumbing.Hash]plumbing.EncodedObject, writer io.Writer) ([]byte, error) {

	cw := newSHA160checksumWriter(writer)
	w := bufio.NewWriterSize(cw, 65519)

	err := writePackfileHeader(uint32(len(objs)), w)
	if err != nil {
		return nil, err
	}
	w.Flush()

	// write all objects
	for _, o := range objs {
		if err = writePackfileEntry(w, o); err != nil {
			return nil, err
		}
		w.Flush()
	}
	//w.Flush()

	checksum := cw.Sum()
	err = binary.Write(w, binary.BigEndian, checksum)

	w.Flush()

	return checksum, err
}

func writePackfileHeader(objCount uint32, w io.Writer) (err error) {
	// signature
	if err = binary.Write(w, binary.BigEndian, []byte("PACK")); err == nil {
		// packfile version
		if err = binary.Write(w, binary.BigEndian, uint32(2)); err == nil {
			// object count
			err = binary.Write(w, binary.BigEndian, objCount)
		}
	}
	return err
}*/

type checksumWriter struct {
	hash   hash.Hash
	writer io.Writer
}

func newSHA160checksumWriter(w io.Writer) *checksumWriter {
	return &checksumWriter{hash: sha1.New(), writer: w}
}

func (w *checksumWriter) Write(p []byte) (n int, err error) {
	w.hash.Write(p)
	return w.writer.Write(p)
}

func (w *checksumWriter) Sum() []byte {
	return w.hash.Sum(nil)
}
