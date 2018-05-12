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
	add(key string, val *lazyNode) error
	remove(key string) error
}

func newLazyNode(raw *json.RawMessage) *lazyNode {
	return &lazyNode{raw: raw, doc: nil, ary: nil, which: eRaw}
}

func (n *lazyNode) MarshalJSON() ([]byte, error) {
	switch n.which {
	case eRaw:
		return json.Marshal(n.raw)
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

	if n.raw == nil {
		return nil, fmt.Errorf("Unable to unmarshal nil pointer as partial document")
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

	if n.raw == nil {
		return nil, fmt.Errorf("Unable to unmarshal nil pointer as partial array")
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

	if n.raw == nil {
		return nil
	}

	err := json.Compact(buf, *n.raw)

	if err != nil {
		return *n.raw
	}

	return buf.Bytes()
}

func (n *lazyNode) tryDoc() bool {
	if n.raw == nil {
		return false
	}

	err := json.Unmarshal(*n.raw, &n.doc)

	if err != nil {
		return false
	}

	n.which = eDoc
	return true
}

func (n *lazyNode) tryAry() bool {
	if n.raw == nil {
		return false
	}

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

func findObject(pd *container, path string) (container, string) {
	doc := *pd

	split := strings.Split(path, "/")

	if len(split) < 2 {
		return nil, ""
	}

	parts := split[1 : len(split)-1]

	key := split[len(split)-1]

	var err error

	for _, part := range parts {

		next, ok := doc.get(decodePatchKey(part))

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

	return doc, decodePatchKey(key)
}

func (d *partialDoc) set(key string, val *lazyNode) error {
	(*d)[key] = val
	return nil
}

func (d *partialDoc) add(key string, val *lazyNode) error {
	(*d)[key] = val
	return nil
}

func (d *partialDoc) get(key string) (*lazyNode, error) {
	return (*d)[key], nil
}

func (d *partialDoc) remove(key string) error {
	_, ok := (*d)[key]
	if !ok {
		return fmt.Errorf("Unable to remove nonexistent key: %s", key)
	}

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

	sz := len(*d)
	if idx+1 > sz {
		sz = idx + 1
	}

	ary := make([]*lazyNode, sz)

	cur := *d

	copy(ary, cur)

	if idx >= len(ary) {
		return fmt.Errorf("Unable to access invalid index: %d", idx)
	}

	ary[idx] = val

	*d = ary
	return nil
}

func (d *partialArray) add(key string, val *lazyNode) error {
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

	if idx < 0 {
		idx *= -1

		if idx > len(ary) {
			return fmt.Errorf("Unable to access invalid index: %d", idx)
		}
		idx = len(ary) - idx
	}

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

	if idx >= len(*d) {
		return nil, fmt.Errorf("Unable to access invalid index: %d", idx)
	}

	return (*d)[idx], nil
}

func (d *partialArray) remove(key string) error {
	idx, err := strconv.Atoi(key)
	if err != nil {
		return err
	}

	cur := *d

	if idx >= len(cur) {
		return fmt.Errorf("Unable to remove invalid index: %d", idx)
	}

	ary := make([]*lazyNode, len(cur)-1)

	copy(ary[0:idx], cur[0:idx])
	copy(ary[idx:], cur[idx+1:])

	*d = ary
	return nil

}

func (p Patch) add(doc *container, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	if con == nil {
		return fmt.Errorf("jsonpatch add operation does not apply: doc is missing path: %s", path)
	}

	return con.add(key, op.value())
}

func (p Patch) remove(doc *container, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	if con == nil {
		return fmt.Errorf("jsonpatch remove operation does not apply: doc is missing path: %s", path)
	}

	return con.remove(key)
}

func (p Patch) replace(doc *container, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	if con == nil {
		return fmt.Errorf("jsonpatch replace operation does not apply: doc is missing path: %s", path)
	}

	val, ok := con.get(key)
	if val == nil || ok != nil {
		return fmt.Errorf("jsonpatch replace operation does not apply: doc is missing key: %s", path)
	}

	return con.set(key, op.value())
}

func (p Patch) move(doc *container, op operation) error {
	from := op.from()

	con, key := findObject(doc, from)

	if con == nil {
		return fmt.Errorf("jsonpatch move operation does not apply: doc is missing from path: %s", from)
	}

	val, err := con.get(key)
	if err != nil {
		return err
	}

	err = con.remove(key)
	if err != nil {
		return err
	}

	path := op.path()

	con, key = findObject(doc, path)

	if con == nil {
		return fmt.Errorf("jsonpatch move operation does not apply: doc is missing destination path: %s", path)
	}

	return con.set(key, val)
}

func (p Patch) test(doc *container, op operation) error {
	path := op.path()

	con, key := findObject(doc, path)

	if con == nil {
		return fmt.Errorf("jsonpatch test operation does not apply: is missing path: %s", path)
	}

	val, err := con.get(key)

	if err != nil {
		return err
	}

	if val == nil {
		if op.value().raw == nil {
			return nil
		}
		return fmt.Errorf("Testing value %s failed", path)
	}

	if val.equal(op.value()) {
		return nil
	}

	return fmt.Errorf("Testing value %s failed", path)
}

func (p Patch) copy(doc *container, op operation) error {
	from := op.from()

	con, key := findObject(doc, from)

	if con == nil {
		return fmt.Errorf("jsonpatch copy operation does not apply: doc is missing from path: %s", from)
	}

	val, err := con.get(key)
	if err != nil {
		return err
	}

	path := op.path()

	con, key = findObject(doc, path)

	if con == nil {
		return fmt.Errorf("jsonpatch copy operation does not apply: doc is missing destination path: %s", path)
	}

	return con.set(key, val)
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
	return p.ApplyIndent(doc, "")
}

// ApplyIndent mutates a JSON document according to the patch, and returns the new
// document indented.
func (p Patch) ApplyIndent(doc []byte, indent string) ([]byte, error) {
	var pd container
	if doc[0] == '[' {
		pd = &partialArray{}
	} else {
		pd = &partialDoc{}
	}

	err := json.Unmarshal(doc, pd)

	if err != nil {
		return nil, err
	}

	err = nil

	for _, op := range p {
		switch op.kind() {
		case "add":
			err = p.add(&pd, op)
		case "remove":
			err = p.remove(&pd, op)
		case "replace":
			err = p.replace(&pd, op)
		case "move":
			err = p.move(&pd, op)
		case "test":
			err = p.test(&pd, op)
		case "copy":
			err = p.copy(&pd, op)
		default:
			err = fmt.Errorf("Unexpected kind: %s", op.kind())
		}

		if err != nil {
			return nil, err
		}
	}

	if indent != "" {
		return json.MarshalIndent(pd, "", indent)
	}

	return json.Marshal(pd)
}

// From http://tools.ietf.org/html/rfc6901#section-4 :
//
// Evaluation of each reference token begins by decoding any escaped
// character sequence.  This is performed by first transforming any
// occurrence of the sequence '~1' to '/', and then transforming any
// occurrence of the sequence '~0' to '~'.

var (
	rfc6901Decoder = strings.NewReplacer("~1", "/", "~0", "~")
)

func decodePatchKey(k string) string {
	return rfc6901Decoder.Replace(k)
}
