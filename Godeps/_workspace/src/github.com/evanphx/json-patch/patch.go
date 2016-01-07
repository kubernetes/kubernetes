package jsonpatch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

const (
	eRaw = iota
	eDoc
	eAry
)

type lazyNode struct {
	raw   *json.RawMessage
	doc   partialDoc
	ary   partialArray
	which int
}

type operation map[string]*json.RawMessage

// Patch is an ordered collection of operations.
type Patch []operation

type partialDoc map[string]*lazyNode
type partialArray []*lazyNode

type container interface {
	get(key string) (*lazyNode, error)
	set(key string, val *lazyNode) error
	remove(key string) error
}

func newLazyNode(raw *json.RawMessage) *lazyNode {
	return &lazyNode{raw: raw, doc: nil, ary: nil, which: eRaw}
}

func (n *lazyNode) MarshalJSON() ([]byte, error) {
	switch n.which {
	case eRaw:
		return *n.raw, nil
	case eDoc:
		return json.Marshal(n.doc)
	case eAry:
		return json.Marshal(n.ary)
	default:
		return nil, fmt.Errorf("Unknown type")
	}
}

func (n *lazyNode) UnmarshalJSON(data []byte) error {
	dest := make(json.RawMessage, len(data))
	copy(dest, data)
	n.raw = &dest
	n.which = eRaw
	return nil
}

func (n *lazyNode) intoDoc() (*partialDoc, error) {
	if n.which == eDoc {
		return &n.doc, nil
	}

	err := json.Unmarshal(*n.raw, &n.doc)

	if err != nil {
		return nil, err
	}

	n.which = eDoc
	return &n.doc, nil
}

func (n *lazyNode) intoAry() (*partialArray, error) {
	if n.which == eAry {
		return &n.ary, nil
	}

	err := json.Unmarshal(*n.raw, &n.ary)

	if err != nil {
		return nil, err
	}

	n.which = eAry
	return &n.ary, nil
}

func (n *lazyNode) compact() []byte {
	buf := &bytes.Buffer{}

	err := json.Compact(buf, *n.raw)

	if err != nil {
		return *n.raw
	}

	return buf.Bytes()
}

func (n *lazyNode) tryDoc() bool {
	err := json.Unmarshal(*n.raw, &n.doc)

	if err != nil {
		return false
	}

	n.which = eDoc
	return true
}

func (n *lazyNode) tryAry() bool {
	err := json.Unmarshal(*n.raw, &n.ary)

	if err != nil {
		return false
	}

	n.which = eAry
	return true
}

func (n *lazyNode) equal(o *lazyNode) bool {
	if n.which == eRaw {
		if !n.tryDoc() && !n.tryAry() {
			if o.which != eRaw {
				return false
			}

			return bytes.Equal(n.compact(), o.compact())
		}
	}

	if n.which == eDoc {
		if o.which == eRaw {
			if !o.tryDoc() {
				return false
			}
		}

		if o.which != eDoc {
			return false
		}

		for k, v := range n.doc {
			ov, ok := o.doc[k]

			if !ok {
				return false
			}

			if v == nil && ov == nil {
				continue
			}

			if !v.equal(ov) {
				return false
			}
		}

		return true
	}

	if o.which != eAry && !o.tryAry() {
		return false
	}

	if len(n.ary) != len(o.ary) {
		return false
	}

	for idx, val := range n.ary {
		if !val.equal(o.ary[idx]) {
			return false
		}
	}

	return true
}

func (o operation) kind() string {
	if obj, ok := o["op"]; ok {
		var op string

		err := json.Unmarshal(*obj, &op)

		if err != nil {
			return "unknown"
		}

		return op
	}

	return "unknown"
}

func (o operation) path() string {
	if obj, ok := o["path"]; ok {
		var op string

		err := json.Unmarshal(*obj, &op)

		if err != nil {
			return "unknown"
		}

		return op
	}

	return "unknown"
}

func (o operation) from() string {
	if obj, ok := o["from"]; ok {
		var op string

		err := json.Unmarshal(*obj, &op)

		if err != nil {
			return "unknown"
		}

		return op
	}

	return "unknown"
}

func (o operation) value() *lazyNode {
	if obj, ok := o["value"]; ok {
		return newLazyNode(obj)
	}

	return nil
}

func isArray(buf []byte) bool {
Loop:
	for _, c := range buf {
		switch c {
		case ' ':
		case '\n':
		case '\t':
			continue
		case '[':
			return true
		default:
			break Loop
		}
	}

	return false
}

func findObject(pd *partialDoc, path string) (container, string) {
	doc := container(pd)

	split := strings.Split(path, "/")

	parts := split[1 : len(split)-1]

	key := split[len(split)-1]

	var err error

	for _, part := range parts {

		next, ok := doc.get(part)

		if next == nil || ok != nil {
			return nil, ""
		}

		if isArray(*next.raw) {
			doc, err = next.intoAry()

			if err != nil {
				return nil, ""
			}
		} else {
			doc, err = next.intoDoc()

			if err != nil {
				return nil, ""
			}
		}
	}

	return doc, key
}

func (d *partialDoc) set(key string, val *lazyNode) error {
	(*d)[key] = val
	return nil
}

func (d *partialDoc) get(key string) (*lazyNode, error) {
	return (*d)[key], nil
}

func (d *partialDoc) remove(key string) error {
	delete(*d, key)
	return nil
}

func (d *partialArray) set(key string, val *lazyNode) error {
	if key == "-" {
		*d = append(*d, val)
		return nil
	}

	idx, err := strconv.Atoi(key)

	if err != nil {
		return err
	}

	ary := make([]*lazyNode, len(*d)+1)

	cur := *d

	copy(ary[0:idx], cur[0:idx])
	ary[idx] = val
	copy(ary[idx+1:], cur[idx:])

	*d = ary
	return nil
}

func (d *partialArray) get(key string) (*lazyNode, error) {
	idx, err := strconv.Atoi(key)

	if err != nil {
		return nil, err
	}

	return (*d)[idx], nil
}

func (d *partialArray) remove(key string) error {
	idx, err := strconv.Atoi(key)

	if err != nil {
		return err
	}

	cur := *d

	ary := make([]*lazyNode, len(cur)-1)

	copy(ary[0:idx], cur[0:idx])
	copy(ary[idx:], cur[idx+1:])

	*d = ary
	return nil

}

func (p Patch) add(doc *partialDoc, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	if con == nil {
		return fmt.Errorf("Missing container: %s", path)
	}

	con.set(key, op.value())

	return nil
}

func (p Patch) remove(doc *partialDoc, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	return con.remove(key)
}

func (p Patch) replace(doc *partialDoc, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	con.set(key, op.value())

	return nil
}

func (p Patch) move(doc *partialDoc, op operation) error {
	from := op.from()

	con, key := findObject(doc, from)

	val, err := con.get(key)

	if err != nil {
		return err
	}

	con.remove(key)

	path := op.path()

	con, key = findObject(doc, path)

	con.set(key, val)

	return nil
}

func (p Patch) test(doc *partialDoc, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	val, err := con.get(key)

	if err != nil {
		return err
	}

	if val.equal(op.value()) {
		return nil
	}

	return fmt.Errorf("Testing value %s failed", path)
}

// Equal indicates if 2 JSON documents have the same structural equality.
func Equal(a, b []byte) bool {
	ra := make(json.RawMessage, len(a))
	copy(ra, a)
	la := newLazyNode(&ra)

	rb := make(json.RawMessage, len(b))
	copy(rb, b)
	lb := newLazyNode(&rb)

	return la.equal(lb)
}

// DecodePatch decodes the passed JSON document as an RFC 6902 patch.
func DecodePatch(buf []byte) (Patch, error) {
	var p Patch

	err := json.Unmarshal(buf, &p)

	if err != nil {
		return nil, err
	}

	return p, nil
}

// Apply mutates a JSON document according to the patch, and returns the new
// document.
func (p Patch) Apply(doc []byte) ([]byte, error) {
	pd := &partialDoc{}

	err := json.Unmarshal(doc, pd)

	if err != nil {
		return nil, err
	}

	err = nil

	for _, op := range p {
		switch op.kind() {
		case "add":
			err = p.add(pd, op)
		case "remove":
			err = p.remove(pd, op)
		case "replace":
			err = p.replace(pd, op)
		case "move":
			err = p.move(pd, op)
		case "test":
			err = p.test(pd, op)
		default:
			err = fmt.Errorf("Unexpected kind: %s", op.kind())
		}

		if err != nil {
			return nil, err
		}
	}

	return json.Marshal(pd)
}
