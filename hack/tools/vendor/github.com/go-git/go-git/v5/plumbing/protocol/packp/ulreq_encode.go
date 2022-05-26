package packp

import (
	"bytes"
	"fmt"
	"io"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/format/pktline"
)

// Encode writes the UlReq encoding of u to the stream.
//
// All the payloads will end with a newline character.  Wants and
// shallows are sorted alphabetically.  A depth of 0 means no depth
// request is sent.
func (req *UploadRequest) Encode(w io.Writer) error {
	e := newUlReqEncoder(w)
	return e.Encode(req)
}

type ulReqEncoder struct {
	pe   *pktline.Encoder // where to write the encoded data
	data *UploadRequest   // the data to encode
	err  error            // sticky error
}

func newUlReqEncoder(w io.Writer) *ulReqEncoder {
	return &ulReqEncoder{
		pe: pktline.NewEncoder(w),
	}
}

func (e *ulReqEncoder) Encode(v *UploadRequest) error {
	e.data = v

	if len(v.Wants) == 0 {
		return fmt.Errorf("empty wants provided")
	}

	plumbing.HashesSort(e.data.Wants)
	for state := e.encodeFirstWant; state != nil; {
		state = state()
	}

	return e.err
}

func (e *ulReqEncoder) encodeFirstWant() stateFn {
	var err error
	if e.data.Capabilities.IsEmpty() {
		err = e.pe.Encodef("want %s\n", e.data.Wants[0])
	} else {
		err = e.pe.Encodef(
			"want %s %s\n",
			e.data.Wants[0],
			e.data.Capabilities.String(),
		)
	}

	if err != nil {
		e.err = fmt.Errorf("encoding first want line: %s", err)
		return nil
	}

	return e.encodeAdditionalWants
}

func (e *ulReqEncoder) encodeAdditionalWants() stateFn {
	last := e.data.Wants[0]
	for _, w := range e.data.Wants[1:] {
		if bytes.Equal(last[:], w[:]) {
			continue
		}

		if err := e.pe.Encodef("want %s\n", w); err != nil {
			e.err = fmt.Errorf("encoding want %q: %s", w, err)
			return nil
		}

		last = w
	}

	return e.encodeShallows
}

func (e *ulReqEncoder) encodeShallows() stateFn {
	plumbing.HashesSort(e.data.Shallows)

	var last plumbing.Hash
	for _, s := range e.data.Shallows {
		if bytes.Equal(last[:], s[:]) {
			continue
		}

		if err := e.pe.Encodef("shallow %s\n", s); err != nil {
			e.err = fmt.Errorf("encoding shallow %q: %s", s, err)
			return nil
		}

		last = s
	}

	return e.encodeDepth
}

func (e *ulReqEncoder) encodeDepth() stateFn {
	switch depth := e.data.Depth.(type) {
	case DepthCommits:
		if depth != 0 {
			commits := int(depth)
			if err := e.pe.Encodef("deepen %d\n", commits); err != nil {
				e.err = fmt.Errorf("encoding depth %d: %s", depth, err)
				return nil
			}
		}
	case DepthSince:
		when := time.Time(depth).UTC()
		if err := e.pe.Encodef("deepen-since %d\n", when.Unix()); err != nil {
			e.err = fmt.Errorf("encoding depth %s: %s", when, err)
			return nil
		}
	case DepthReference:
		reference := string(depth)
		if err := e.pe.Encodef("deepen-not %s\n", reference); err != nil {
			e.err = fmt.Errorf("encoding depth %s: %s", reference, err)
			return nil
		}
	default:
		e.err = fmt.Errorf("unsupported depth type")
		return nil
	}

	return e.encodeFlush
}

func (e *ulReqEncoder) encodeFlush() stateFn {
	if err := e.pe.Flush(); err != nil {
		e.err = fmt.Errorf("encoding flush-pkt: %s", err)
		return nil
	}

	return nil
}
