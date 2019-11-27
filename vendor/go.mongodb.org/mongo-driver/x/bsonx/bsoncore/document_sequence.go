package bsoncore

import (
	"errors"
	"io"

	"go.mongodb.org/mongo-driver/bson/bsontype"
)

// DocumentSequenceStyle is used to represent how a document sequence is laid out in a slice of
// bytes.
type DocumentSequenceStyle uint32

// These constants are the valid styles for a DocumentSequence.
const (
	_ DocumentSequenceStyle = iota
	SequenceStyle
	ArrayStyle
)

// DocumentSequence represents a sequence of documents. The Style field indicates how the documents
// are laid out inside of the Data field.
type DocumentSequence struct {
	Style DocumentSequenceStyle
	Data  []byte
	Pos   int
}

// ErrCorruptedDocument is returned when a full document couldn't be read from the sequence.
var ErrCorruptedDocument = errors.New("invalid DocumentSequence: corrupted document")

// ErrNonDocument is returned when a DocumentSequence contains a non-document BSON value.
var ErrNonDocument = errors.New("invalid DocumentSequence: a non-document value was found in sequence")

// ErrInvalidDocumentSequenceStyle is returned when an unknown DocumentSequenceStyle is set on a
// DocumentSequence.
var ErrInvalidDocumentSequenceStyle = errors.New("invalid DocumentSequenceStyle")

// DocumentCount returns the number of documents in the sequence.
func (ds *DocumentSequence) DocumentCount() int {
	if ds == nil {
		return 0
	}
	switch ds.Style {
	case SequenceStyle:
		var count int
		var ok bool
		rem := ds.Data
		for len(rem) > 0 {
			_, rem, ok = ReadDocument(rem)
			if !ok {
				return 0
			}
			count++
		}
		return count
	case ArrayStyle:
		_, rem, ok := ReadLength(ds.Data)
		if !ok {
			return 0
		}

		var count int
		for len(rem) > 1 {
			_, rem, ok = ReadElement(rem)
			if !ok {
				return 0
			}
			count++
		}
		return count
	default:
		return 0
	}
}

// Empty returns true if the sequence is empty. It always returns true for unknown sequence styles.
func (ds *DocumentSequence) Empty() bool {
	if ds == nil {
		return true
	}

	switch ds.Style {
	case SequenceStyle:
		return len(ds.Data) == 0
	case ArrayStyle:
		return len(ds.Data) <= 5
	default:
		return true
	}
}

//ResetIterator resets the iteration point for the Next method to the beginning of the document
//sequence.
func (ds *DocumentSequence) ResetIterator() {
	if ds == nil {
		return
	}
	ds.Pos = 0
}

// Documents returns a slice of the documents. If nil either the Data field is also nil or could not
// be properly read.
func (ds *DocumentSequence) Documents() ([]Document, error) {
	if ds == nil {
		return nil, nil
	}
	switch ds.Style {
	case SequenceStyle:
		rem := ds.Data
		var docs []Document
		var doc Document
		var ok bool
		for {
			doc, rem, ok = ReadDocument(rem)
			if !ok {
				if len(rem) == 0 {
					break
				}
				return nil, ErrCorruptedDocument
			}
			docs = append(docs, doc)
		}
		return docs, nil
	case ArrayStyle:
		if len(ds.Data) == 0 {
			return nil, nil
		}
		vals, err := Document(ds.Data).Values()
		if err != nil {
			return nil, ErrCorruptedDocument
		}
		docs := make([]Document, 0, len(vals))
		for _, v := range vals {
			if v.Type != bsontype.EmbeddedDocument {
				return nil, ErrNonDocument
			}
			docs = append(docs, v.Data)
		}
		return docs, nil
	default:
		return nil, ErrInvalidDocumentSequenceStyle
	}
}

// Next retrieves the next document from this sequence and returns it. This method will return
// io.EOF when it has reached the end of the sequence.
func (ds *DocumentSequence) Next() (Document, error) {
	if ds == nil || ds.Pos >= len(ds.Data) {
		return nil, io.EOF
	}
	switch ds.Style {
	case SequenceStyle:
		doc, _, ok := ReadDocument(ds.Data[ds.Pos:])
		if !ok {
			return nil, ErrCorruptedDocument
		}
		ds.Pos += len(doc)
		return doc, nil
	case ArrayStyle:
		if ds.Pos < 4 {
			if len(ds.Data) < 4 {
				return nil, ErrCorruptedDocument
			}
			ds.Pos = 4 // Skip the length of the document
		}
		if len(ds.Data[ds.Pos:]) == 1 && ds.Data[ds.Pos] == 0x00 {
			return nil, io.EOF // At the end of the document
		}
		elem, _, ok := ReadElement(ds.Data[ds.Pos:])
		if !ok {
			return nil, ErrCorruptedDocument
		}
		ds.Pos += len(elem)
		val := elem.Value()
		if val.Type != bsontype.EmbeddedDocument {
			return nil, ErrNonDocument
		}
		return val.Data, nil
	default:
		return nil, ErrInvalidDocumentSequenceStyle
	}
}
