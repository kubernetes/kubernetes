package packp

import (
	"bytes"
	"fmt"
	"io"
	"sort"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/format/pktline"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
)

// Encode writes the AdvRefs encoding to a writer.
//
// All the payloads will end with a newline character. Capabilities,
// references and shallows are written in alphabetical order, except for
// peeled references that always follow their corresponding references.
func (a *AdvRefs) Encode(w io.Writer) error {
	e := newAdvRefsEncoder(w)
	return e.Encode(a)
}

type advRefsEncoder struct {
	data         *AdvRefs         // data to encode
	pe           *pktline.Encoder // where to write the encoded data
	firstRefName string           // reference name to encode in the first pkt-line (HEAD if present)
	firstRefHash plumbing.Hash    // hash referenced to encode in the first pkt-line (HEAD if present)
	sortedRefs   []string         // hash references to encode ordered by increasing order
	err          error            // sticky error

}

func newAdvRefsEncoder(w io.Writer) *advRefsEncoder {
	return &advRefsEncoder{
		pe: pktline.NewEncoder(w),
	}
}

func (e *advRefsEncoder) Encode(v *AdvRefs) error {
	e.data = v
	e.sortRefs()
	e.setFirstRef()

	for state := encodePrefix; state != nil; {
		state = state(e)
	}

	return e.err
}

func (e *advRefsEncoder) sortRefs() {
	if len(e.data.References) > 0 {
		refs := make([]string, 0, len(e.data.References))
		for refName := range e.data.References {
			refs = append(refs, refName)
		}

		sort.Strings(refs)
		e.sortedRefs = refs
	}
}

func (e *advRefsEncoder) setFirstRef() {
	if e.data.Head != nil {
		e.firstRefName = head
		e.firstRefHash = *e.data.Head
		return
	}

	if len(e.sortedRefs) > 0 {
		refName := e.sortedRefs[0]
		e.firstRefName = refName
		e.firstRefHash = e.data.References[refName]
	}
}

type encoderStateFn func(*advRefsEncoder) encoderStateFn

func encodePrefix(e *advRefsEncoder) encoderStateFn {
	for _, p := range e.data.Prefix {
		if bytes.Equal(p, pktline.Flush) {
			if e.err = e.pe.Flush(); e.err != nil {
				return nil
			}
			continue
		}
		if e.err = e.pe.Encodef("%s\n", string(p)); e.err != nil {
			return nil
		}
	}

	return encodeFirstLine
}

// Adds the first pkt-line payload: head hash, head ref and capabilities.
// If HEAD ref is not found, the first reference ordered in increasing order will be used.
// If there aren't HEAD neither refs, the first line will be "PKT-LINE(zero-id SP "capabilities^{}" NUL capability-list)".
// See: https://github.com/git/git/blob/master/Documentation/technical/pack-protocol.txt
// See: https://github.com/git/git/blob/master/Documentation/technical/protocol-common.txt
func encodeFirstLine(e *advRefsEncoder) encoderStateFn {
	const formatFirstLine = "%s %s\x00%s\n"
	var firstLine string
	capabilities := formatCaps(e.data.Capabilities)

	if e.firstRefName == "" {
		firstLine = fmt.Sprintf(formatFirstLine, plumbing.ZeroHash.String(), "capabilities^{}", capabilities)
	} else {
		firstLine = fmt.Sprintf(formatFirstLine, e.firstRefHash.String(), e.firstRefName, capabilities)

	}

	if e.err = e.pe.EncodeString(firstLine); e.err != nil {
		return nil
	}

	return encodeRefs
}

func formatCaps(c *capability.List) string {
	if c == nil {
		return ""
	}

	return c.String()
}

// Adds the (sorted) refs: hash SP refname EOL
// and their peeled refs if any.
func encodeRefs(e *advRefsEncoder) encoderStateFn {
	for _, r := range e.sortedRefs {
		if r == e.firstRefName {
			continue
		}

		hash := e.data.References[r]
		if e.err = e.pe.Encodef("%s %s\n", hash.String(), r); e.err != nil {
			return nil
		}

		if hash, ok := e.data.Peeled[r]; ok {
			if e.err = e.pe.Encodef("%s %s^{}\n", hash.String(), r); e.err != nil {
				return nil
			}
		}
	}

	return encodeShallow
}

// Adds the (sorted) shallows: "shallow" SP hash EOL
func encodeShallow(e *advRefsEncoder) encoderStateFn {
	sorted := sortShallows(e.data.Shallows)
	for _, hash := range sorted {
		if e.err = e.pe.Encodef("shallow %s\n", hash); e.err != nil {
			return nil
		}
	}

	return encodeFlush
}

func sortShallows(c []plumbing.Hash) []string {
	ret := []string{}
	for _, h := range c {
		ret = append(ret, h.String())
	}
	sort.Strings(ret)

	return ret
}

func encodeFlush(e *advRefsEncoder) encoderStateFn {
	e.err = e.pe.Flush()
	return nil
}
