package packfile

import (
	"bytes"
	"fmt"
	"io"
	"log"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
	"gopkg.in/src-d/go-git.v4/plumbing/storer"
)

type Decoder struct {
	scanner *packfile.Scanner
	store   storer.EncodedObjectStorer
	// packfile offset to object map
	objmap map[int64]plumbing.EncodedObject
}

func NewDecoder(rd io.Reader, store storer.EncodedObjectStorer) *Decoder {
	return &Decoder{
		scanner: packfile.NewScanner(rd),
		store:   store,
		objmap:  map[int64]plumbing.EncodedObject{},
	}
}

// Decode from reader and write to object storage
func (dec *Decoder) Decode() error {

	defer dec.scanner.Close()

	version, objcount, err := dec.scanner.Header()
	if err != nil {
		return err
	}

	log.Printf("DBG [packfile] version=%d objects=%d", version, objcount)

	for i := 0; i < int(objcount); i++ {

		header, err := dec.scanner.NextObjectHeader()
		if err != nil {
			return err
		}
		//log.Printf("%d Header: type=%s length=%d offset=%d", i+1, header.Type, header.Length, header.Offset)

		var obj plumbing.EncodedObject

		switch header.Type {
		case plumbing.OFSDeltaObject:
			obj, err = dec.makeOFSDeltaObject(header)

		case plumbing.REFDeltaObject:
			obj, err = dec.makeRefDeltaObject(header)

		default:
			obj, err = dec.makeObject(header)

		}

		if err != nil {
			return err
		}

		dec.objmap[header.Offset] = obj
	}

	// Set all objects
	for _, v := range dec.objmap {
		if _, err := dec.store.SetEncodedObject(v); err != nil {
			return err
		}
	}

	return nil
}

func (dec *Decoder) makeObject(header *packfile.ObjectHeader) (plumbing.EncodedObject, error) {
	obj := &plumbing.MemoryObject{}
	obj.SetType(header.Type)
	obj.SetSize(header.Length)

	w, err := obj.Writer()
	if err == nil {
		if _, _, err = dec.scanner.NextObject(w); err == nil {
			err = obj.Close()
		}
	}

	return obj, err
}

func (dec *Decoder) makeDeltaObject(header *packfile.ObjectHeader, base plumbing.EncodedObject) (plumbing.EncodedObject, error) {
	obj := &plumbing.MemoryObject{}
	obj.SetSize(header.Length)
	obj.SetType(base.Type())

	buf := new(bytes.Buffer)
	_, _, err := dec.scanner.NextObject(buf)
	if err == nil {
		err = packfile.ApplyDelta(obj, base, buf.Bytes())
	}

	return obj, err
}

func (dec *Decoder) makeOFSDeltaObject(header *packfile.ObjectHeader) (plumbing.EncodedObject, error) {
	base, ok := dec.objmap[header.OffsetReference]
	if !ok {
		return nil, fmt.Errorf("object not found at offset: %d", header.OffsetReference)
	}

	return dec.makeDeltaObject(header, base)
}

func (dec *Decoder) makeRefDeltaObject(header *packfile.ObjectHeader) (plumbing.EncodedObject, error) {
	base, err := dec.store.EncodedObject(plumbing.AnyObject, header.Reference)
	if err != nil {
		return nil, err
	}

	return dec.makeDeltaObject(header, base)
}
