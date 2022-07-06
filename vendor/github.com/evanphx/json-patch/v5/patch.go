package jsonpatch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/pkg/errors"
)

const (
	eRaw = iota
	eDoc
	eAry
)

var (
	// SupportNegativeIndices decides whether to support non-standard practice of
	// allowing negative indices to mean indices starting at the end of an array.
	// Default to true.
	SupportNegativeIndices bool = true
	// AccumulatedCopySizeLimit limits the total size increase in bytes caused by
	// "copy" operations in a patch.
	AccumulatedCopySizeLimit int64 = 0
	startObject                    = json.Delim('{')
	endObject                      = json.Delim('}')
	startArray                     = json.Delim('[')
	endArray                       = json.Delim(']')
)

var (
	ErrTestFailed   = errors.New("test failed")
	ErrMissing      = errors.New("missing value")
	ErrUnknownType  = errors.New("unknown object type")
	ErrInvalid      = errors.New("invalid state detected")
	ErrInvalidIndex = errors.New("invalid index referenced")

	rawJSONArray  = []byte("[]")
	rawJSONObject = []byte("{}")
	rawJSONNull   = []byte("null")
)

type lazyNode struct {
	raw   *json.RawMessage
	doc   *partialDoc
	ary   partialArray
	which int
}

// Operation is a single JSON-Patch step, such as a single 'add' operation.
type Operation map[string]*json.RawMessage

// Patch is an ordered collection of Operations.
type Patch []Operation

type partialDoc struct {
	keys []string
	obj  map[string]*lazyNode
}

type partialArray []*lazyNode

type container interface {
	get(key string, options *ApplyOptions) (*lazyNode, error)
	set(key string, val *lazyNode, options *ApplyOptions) error
	add(key string, val *lazyNode, options *ApplyOptions) error
	remove(key string, options *ApplyOptions) error
}

// ApplyOptions specifies options for calls to ApplyWithOptions.
// Use NewApplyOptions to obtain default values for ApplyOptions.
type ApplyOptions struct {
	// SupportNegativeIndices decides whether to support non-standard practice of
	// allowing negative indices to mean indices starting at the end of an array.
	// Default to true.
	SupportNegativeIndices bool
	// AccumulatedCopySizeLimit limits the total size increase in bytes caused by
	// "copy" operations in a patch.
	AccumulatedCopySizeLimit int64
	// AllowMissingPathOnRemove indicates whether to fail "remove" operations when the target path is missing.
	// Default to false.
	AllowMissingPathOnRemove bool
	// EnsurePathExistsOnAdd instructs json-patch to recursively create the missing parts of path on "add" operation.
	// Default to false.
	EnsurePathExistsOnAdd bool
}

// NewApplyOptions creates a default set of options for calls to ApplyWithOptions.
func NewApplyOptions() *ApplyOptions {
	return &ApplyOptions{
		SupportNegativeIndices:   SupportNegativeIndices,
		AccumulatedCopySizeLimit: AccumulatedCopySizeLimit,
		AllowMissingPathOnRemove: false,
		EnsurePathExistsOnAdd:    false,
	}
}

func newLazyNode(raw *json.RawMessage) *lazyNode {
	return &lazyNode{raw: raw, doc: nil, ary: nil, which: eRaw}
}

func newRawMessage(buf []byte) *json.RawMessage {
	ra := make(json.RawMessage, len(buf))
	copy(ra, buf)
	return &ra
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
		return nil, ErrUnknownType
	}
}

func (n *lazyNode) UnmarshalJSON(data []byte) error {
	dest := make(json.RawMessage, len(data))
	copy(dest, data)
	n.raw = &dest
	n.which = eRaw
	return nil
}

func (n *partialDoc) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	if _, err := buf.WriteString("{"); err != nil {
		return nil, err
	}
	for i, k := range n.keys {
		if i > 0 {
			if _, err := buf.WriteString(", "); err != nil {
				return nil, err
			}
		}
		key, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		if _, err := buf.Write(key); err != nil {
			return nil, err
		}
		if _, err := buf.WriteString(": "); err != nil {
			return nil, err
		}
		value, err := json.Marshal(n.obj[k])
		if err != nil {
			return nil, err
		}
		if _, err := buf.Write(value); err != nil {
			return nil, err
		}
	}
	if _, err := buf.WriteString("}"); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

type syntaxError struct {
	msg string
}

func (err *syntaxError) Error() string {
	return err.msg
}

func (n *partialDoc) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &n.obj); err != nil {
		return err
	}
	buffer := bytes.NewBuffer(data)
	d := json.NewDecoder(buffer)
	if t, err := d.Token(); err != nil {
		return err
	} else if t != startObject {
		return &syntaxError{fmt.Sprintf("unexpected JSON token in document node: %s", t)}
	}
	for d.More() {
		k, err := d.Token()
		if err != nil {
			return err
		}
		key, ok := k.(string)
		if !ok {
			return &syntaxError{fmt.Sprintf("unexpected JSON token as document node key: %s", k)}
		}
		if err := skipValue(d); err != nil {
			return err
		}
		n.keys = append(n.keys, key)
	}
	return nil
}

func skipValue(d *json.Decoder) error {
	t, err := d.Token()
	if err != nil {
		return err
	}
	if t != startObject && t != startArray {
		return nil
	}
	for d.More() {
		if t == startObject {
			// consume key token
			if _, err := d.Token(); err != nil {
				return err
			}
		}
		if err := skipValue(d); err != nil {
			return err
		}
	}
	end, err := d.Token()
	if err != nil {
		return err
	}
	if t == startObject && end != endObject {
		return &syntaxError{msg: "expected close object token"}
	}
	if t == startArray && end != endArray {
		return &syntaxError{msg: "expected close object token"}
	}
	return nil
}

func deepCopy(src *lazyNode) (*lazyNode, int, error) {
	if src == nil {
		return nil, 0, nil
	}
	a, err := src.MarshalJSON()
	if err != nil {
		return nil, 0, err
	}
	sz := len(a)
	return newLazyNode(newRawMessage(a)), sz, nil
}

func (n *lazyNode) intoDoc() (*partialDoc, error) {
	if n.which == eDoc {
		return n.doc, nil
	}

	if n.raw == nil {
		return nil, ErrInvalid
	}

	err := json.Unmarshal(*n.raw, &n.doc)

	if err != nil {
		return nil, err
	}

	n.which = eDoc
	return n.doc, nil
}

func (n *lazyNode) intoAry() (*partialArray, error) {
	if n.which == eAry {
		return &n.ary, nil
	}

	if n.raw == nil {
		return nil, ErrInvalid
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

		if len(n.doc.obj) != len(o.doc.obj) {
			return false
		}

		for k, v := range n.doc.obj {
			ov, ok := o.doc.obj[k]

			if !ok {
				return false
			}

			if (v == nil) != (ov == nil) {
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

// Kind reads the "op" field of the Operation.
func (o Operation) Kind() string {
	if obj, ok := o["op"]; ok && obj != nil {
		var op string

		err := json.Unmarshal(*obj, &op)

		if err != nil {
			return "unknown"
		}

		return op
	}

	return "unknown"
}

// Path reads the "path" field of the Operation.
func (o Operation) Path() (string, error) {
	if obj, ok := o["path"]; ok && obj != nil {
		var op string

		err := json.Unmarshal(*obj, &op)

		if err != nil {
			return "unknown", err
		}

		return op, nil
	}

	return "unknown", errors.Wrapf(ErrMissing, "operation missing path field")
}

// From reads the "from" field of the Operation.
func (o Operation) From() (string, error) {
	if obj, ok := o["from"]; ok && obj != nil {
		var op string

		err := json.Unmarshal(*obj, &op)

		if err != nil {
			return "unknown", err
		}

		return op, nil
	}

	return "unknown", errors.Wrapf(ErrMissing, "operation, missing from field")
}

func (o Operation) value() *lazyNode {
	if obj, ok := o["value"]; ok {
		return newLazyNode(obj)
	}

	return nil
}

// ValueInterface decodes the operation value into an interface.
func (o Operation) ValueInterface() (interface{}, error) {
	if obj, ok := o["value"]; ok && obj != nil {
		var v interface{}

		err := json.Unmarshal(*obj, &v)

		if err != nil {
			return nil, err
		}

		return v, nil
	}

	return nil, errors.Wrapf(ErrMissing, "operation, missing value field")
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

func findObject(pd *container, path string, options *ApplyOptions) (container, string) {
	doc := *pd

	split := strings.Split(path, "/")

	if len(split) < 2 {
		return nil, ""
	}

	parts := split[1 : len(split)-1]

	key := split[len(split)-1]

	var err error

	for _, part := range parts {

		next, ok := doc.get(decodePatchKey(part), options)

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

func (d *partialDoc) set(key string, val *lazyNode, options *ApplyOptions) error {
	found := false
	for _, k := range d.keys {
		if k == key {
			found = true
			break
		}
	}
	if !found {
		d.keys = append(d.keys, key)
	}
	d.obj[key] = val
	return nil
}

func (d *partialDoc) add(key string, val *lazyNode, options *ApplyOptions) error {
	return d.set(key, val, options)
}

func (d *partialDoc) get(key string, options *ApplyOptions) (*lazyNode, error) {
	v, ok := d.obj[key]
	if !ok {
		return v, errors.Wrapf(ErrMissing, "unable to get nonexistent key: %s", key)
	}
	return v, nil
}

func (d *partialDoc) remove(key string, options *ApplyOptions) error {
	_, ok := d.obj[key]
	if !ok {
		if options.AllowMissingPathOnRemove {
			return nil
		}
		return errors.Wrapf(ErrMissing, "unable to remove nonexistent key: %s", key)
	}
	idx := -1
	for i, k := range d.keys {
		if k == key {
			idx = i
			break
		}
	}
	d.keys = append(d.keys[0:idx], d.keys[idx+1:]...)
	delete(d.obj, key)
	return nil
}

// set should only be used to implement the "replace" operation, so "key" must
// be an already existing index in "d".
func (d *partialArray) set(key string, val *lazyNode, options *ApplyOptions) error {
	idx, err := strconv.Atoi(key)
	if err != nil {
		return err
	}

	if idx < 0 {
		if !options.SupportNegativeIndices {
			return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		if idx < -len(*d) {
			return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		idx += len(*d)
	}

	(*d)[idx] = val
	return nil
}

func (d *partialArray) add(key string, val *lazyNode, options *ApplyOptions) error {
	if key == "-" {
		*d = append(*d, val)
		return nil
	}

	idx, err := strconv.Atoi(key)
	if err != nil {
		return errors.Wrapf(err, "value was not a proper array index: '%s'", key)
	}

	sz := len(*d) + 1

	ary := make([]*lazyNode, sz)

	cur := *d

	if idx >= len(ary) {
		return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
	}

	if idx < 0 {
		if !options.SupportNegativeIndices {
			return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		if idx < -len(ary) {
			return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		idx += len(ary)
	}

	copy(ary[0:idx], cur[0:idx])
	ary[idx] = val
	copy(ary[idx+1:], cur[idx:])

	*d = ary
	return nil
}

func (d *partialArray) get(key string, options *ApplyOptions) (*lazyNode, error) {
	idx, err := strconv.Atoi(key)

	if err != nil {
		return nil, err
	}

	if idx < 0 {
		if !options.SupportNegativeIndices {
			return nil, errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		if idx < -len(*d) {
			return nil, errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		idx += len(*d)
	}

	if idx >= len(*d) {
		return nil, errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
	}

	return (*d)[idx], nil
}

func (d *partialArray) remove(key string, options *ApplyOptions) error {
	idx, err := strconv.Atoi(key)
	if err != nil {
		return err
	}

	cur := *d

	if idx >= len(cur) {
		if options.AllowMissingPathOnRemove {
			return nil
		}
		return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
	}

	if idx < 0 {
		if !options.SupportNegativeIndices {
			return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		if idx < -len(cur) {
			if options.AllowMissingPathOnRemove {
				return nil
			}
			return errors.Wrapf(ErrInvalidIndex, "Unable to access invalid index: %d", idx)
		}
		idx += len(cur)
	}

	ary := make([]*lazyNode, len(cur)-1)

	copy(ary[0:idx], cur[0:idx])
	copy(ary[idx:], cur[idx+1:])

	*d = ary
	return nil
}

func (p Patch) add(doc *container, op Operation, options *ApplyOptions) error {
	path, err := op.Path()
	if err != nil {
		return errors.Wrapf(ErrMissing, "add operation failed to decode path")
	}

	if options.EnsurePathExistsOnAdd {
		err = ensurePathExists(doc, path, options)

		if err != nil {
			return err
		}
	}

	con, key := findObject(doc, path, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "add operation does not apply: doc is missing path: \"%s\"", path)
	}

	err = con.add(key, op.value(), options)
	if err != nil {
		return errors.Wrapf(err, "error in add for path: '%s'", path)
	}

	return nil
}

// Given a document and a path to a key, walk the path and create all missing elements
// creating objects and arrays as needed.
func ensurePathExists(pd *container, path string, options *ApplyOptions) error {
	doc := *pd

	var err error
	var arrIndex int

	split := strings.Split(path, "/")

	if len(split) < 2 {
		return nil
	}

	parts := split[1:]

	for pi, part := range parts {

		// Have we reached the key part of the path?
		// If yes, we're done.
		if pi == len(parts)-1 {
			return nil
		}

		target, ok := doc.get(decodePatchKey(part), options)

		if target == nil || ok != nil {

			// If the current container is an array which has fewer elements than our target index,
			// pad the current container with nulls.
			if arrIndex, err = strconv.Atoi(part); err == nil {
				pa, ok := doc.(*partialArray)

				if ok && arrIndex >= len(*pa)+1 {
					// Pad the array with null values up to the required index.
					for i := len(*pa); i <= arrIndex-1; i++ {
						doc.add(strconv.Itoa(i), newLazyNode(newRawMessage(rawJSONNull)), options)
					}
				}
			}

			// Check if the next part is a numeric index or "-".
			// If yes, then create an array, otherwise, create an object.
			if arrIndex, err = strconv.Atoi(parts[pi+1]); err == nil || parts[pi+1] == "-" {
				if arrIndex < 0 {

					if !options.SupportNegativeIndices {
						return errors.Wrapf(ErrInvalidIndex, "Unable to ensure path for invalid index: %d", arrIndex)
					}

					if arrIndex < -1 {
						return errors.Wrapf(ErrInvalidIndex, "Unable to ensure path for negative index other than -1: %d", arrIndex)
					}

					arrIndex = 0
				}

				newNode := newLazyNode(newRawMessage(rawJSONArray))
				doc.add(part, newNode, options)
				doc, _ = newNode.intoAry()

				// Pad the new array with null values up to the required index.
				for i := 0; i < arrIndex; i++ {
					doc.add(strconv.Itoa(i), newLazyNode(newRawMessage(rawJSONNull)), options)
				}
			} else {
				newNode := newLazyNode(newRawMessage(rawJSONObject))

				doc.add(part, newNode, options)
				doc, _ = newNode.intoDoc()
			}
		} else {
			if isArray(*target.raw) {
				doc, err = target.intoAry()

				if err != nil {
					return err
				}
			} else {
				doc, err = target.intoDoc()

				if err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (p Patch) remove(doc *container, op Operation, options *ApplyOptions) error {
	path, err := op.Path()
	if err != nil {
		return errors.Wrapf(ErrMissing, "remove operation failed to decode path")
	}

	con, key := findObject(doc, path, options)

	if con == nil {
		if options.AllowMissingPathOnRemove {
			return nil
		}
		return errors.Wrapf(ErrMissing, "remove operation does not apply: doc is missing path: \"%s\"", path)
	}

	err = con.remove(key, options)
	if err != nil {
		return errors.Wrapf(err, "error in remove for path: '%s'", path)
	}

	return nil
}

func (p Patch) replace(doc *container, op Operation, options *ApplyOptions) error {
	path, err := op.Path()
	if err != nil {
		return errors.Wrapf(err, "replace operation failed to decode path")
	}

	if path == "" {
		val := op.value()

		if val.which == eRaw {
			if !val.tryDoc() {
				if !val.tryAry() {
					return errors.Wrapf(err, "replace operation value must be object or array")
				}
			}
		}

		switch val.which {
		case eAry:
			*doc = &val.ary
		case eDoc:
			*doc = val.doc
		case eRaw:
			return errors.Wrapf(err, "replace operation hit impossible case")
		}

		return nil
	}

	con, key := findObject(doc, path, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "replace operation does not apply: doc is missing path: %s", path)
	}

	_, ok := con.get(key, options)
	if ok != nil {
		return errors.Wrapf(ErrMissing, "replace operation does not apply: doc is missing key: %s", path)
	}

	err = con.set(key, op.value(), options)
	if err != nil {
		return errors.Wrapf(err, "error in remove for path: '%s'", path)
	}

	return nil
}

func (p Patch) move(doc *container, op Operation, options *ApplyOptions) error {
	from, err := op.From()
	if err != nil {
		return errors.Wrapf(err, "move operation failed to decode from")
	}

	con, key := findObject(doc, from, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "move operation does not apply: doc is missing from path: %s", from)
	}

	val, err := con.get(key, options)
	if err != nil {
		return errors.Wrapf(err, "error in move for path: '%s'", key)
	}

	err = con.remove(key, options)
	if err != nil {
		return errors.Wrapf(err, "error in move for path: '%s'", key)
	}

	path, err := op.Path()
	if err != nil {
		return errors.Wrapf(err, "move operation failed to decode path")
	}

	con, key = findObject(doc, path, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "move operation does not apply: doc is missing destination path: %s", path)
	}

	err = con.add(key, val, options)
	if err != nil {
		return errors.Wrapf(err, "error in move for path: '%s'", path)
	}

	return nil
}

func (p Patch) test(doc *container, op Operation, options *ApplyOptions) error {
	path, err := op.Path()
	if err != nil {
		return errors.Wrapf(err, "test operation failed to decode path")
	}

	if path == "" {
		var self lazyNode

		switch sv := (*doc).(type) {
		case *partialDoc:
			self.doc = sv
			self.which = eDoc
		case *partialArray:
			self.ary = *sv
			self.which = eAry
		}

		if self.equal(op.value()) {
			return nil
		}

		return errors.Wrapf(ErrTestFailed, "testing value %s failed", path)
	}

	con, key := findObject(doc, path, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "test operation does not apply: is missing path: %s", path)
	}

	val, err := con.get(key, options)
	if err != nil && errors.Cause(err) != ErrMissing {
		return errors.Wrapf(err, "error in test for path: '%s'", path)
	}

	if val == nil {
		if op.value().raw == nil {
			return nil
		}
		return errors.Wrapf(ErrTestFailed, "testing value %s failed", path)
	} else if op.value() == nil {
		return errors.Wrapf(ErrTestFailed, "testing value %s failed", path)
	}

	if val.equal(op.value()) {
		return nil
	}

	return errors.Wrapf(ErrTestFailed, "testing value %s failed", path)
}

func (p Patch) copy(doc *container, op Operation, accumulatedCopySize *int64, options *ApplyOptions) error {
	from, err := op.From()
	if err != nil {
		return errors.Wrapf(err, "copy operation failed to decode from")
	}

	con, key := findObject(doc, from, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "copy operation does not apply: doc is missing from path: %s", from)
	}

	val, err := con.get(key, options)
	if err != nil {
		return errors.Wrapf(err, "error in copy for from: '%s'", from)
	}

	path, err := op.Path()
	if err != nil {
		return errors.Wrapf(ErrMissing, "copy operation failed to decode path")
	}

	con, key = findObject(doc, path, options)

	if con == nil {
		return errors.Wrapf(ErrMissing, "copy operation does not apply: doc is missing destination path: %s", path)
	}

	valCopy, sz, err := deepCopy(val)
	if err != nil {
		return errors.Wrapf(err, "error while performing deep copy")
	}

	(*accumulatedCopySize) += int64(sz)
	if options.AccumulatedCopySizeLimit > 0 && *accumulatedCopySize > options.AccumulatedCopySizeLimit {
		return NewAccumulatedCopySizeError(options.AccumulatedCopySizeLimit, *accumulatedCopySize)
	}

	err = con.add(key, valCopy, options)
	if err != nil {
		return errors.Wrapf(err, "error while adding value during copy")
	}

	return nil
}

// Equal indicates if 2 JSON documents have the same structural equality.
func Equal(a, b []byte) bool {
	la := newLazyNode(newRawMessage(a))
	lb := newLazyNode(newRawMessage(b))

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
	return p.ApplyWithOptions(doc, NewApplyOptions())
}

// ApplyWithOptions mutates a JSON document according to the patch and the passed in ApplyOptions.
// It returns the new document.
func (p Patch) ApplyWithOptions(doc []byte, options *ApplyOptions) ([]byte, error) {
	return p.ApplyIndentWithOptions(doc, "", options)
}

// ApplyIndent mutates a JSON document according to the patch, and returns the new
// document indented.
func (p Patch) ApplyIndent(doc []byte, indent string) ([]byte, error) {
	return p.ApplyIndentWithOptions(doc, indent, NewApplyOptions())
}

// ApplyIndentWithOptions mutates a JSON document according to the patch and the passed in ApplyOptions.
// It returns the new document indented.
func (p Patch) ApplyIndentWithOptions(doc []byte, indent string, options *ApplyOptions) ([]byte, error) {
	if len(doc) == 0 {
		return doc, nil
	}

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

	var accumulatedCopySize int64

	for _, op := range p {
		switch op.Kind() {
		case "add":
			err = p.add(&pd, op, options)
		case "remove":
			err = p.remove(&pd, op, options)
		case "replace":
			err = p.replace(&pd, op, options)
		case "move":
			err = p.move(&pd, op, options)
		case "test":
			err = p.test(&pd, op, options)
		case "copy":
			err = p.copy(&pd, op, &accumulatedCopySize, options)
		default:
			err = fmt.Errorf("Unexpected kind: %s", op.Kind())
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
