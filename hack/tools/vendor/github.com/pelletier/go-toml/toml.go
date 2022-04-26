package toml

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"runtime"
	"strings"
)

type tomlValue struct {
	value     interface{} // string, int64, uint64, float64, bool, time.Time, [] of any of this list
	comment   string
	commented bool
	multiline bool
	literal   bool
	position  Position
}

// Tree is the result of the parsing of a TOML file.
type Tree struct {
	values    map[string]interface{} // string -> *tomlValue, *Tree, []*Tree
	comment   string
	commented bool
	inline    bool
	position  Position
}

func newTree() *Tree {
	return newTreeWithPosition(Position{})
}

func newTreeWithPosition(pos Position) *Tree {
	return &Tree{
		values:   make(map[string]interface{}),
		position: pos,
	}
}

// TreeFromMap initializes a new Tree object using the given map.
func TreeFromMap(m map[string]interface{}) (*Tree, error) {
	result, err := toTree(m)
	if err != nil {
		return nil, err
	}
	return result.(*Tree), nil
}

// Position returns the position of the tree.
func (t *Tree) Position() Position {
	return t.position
}

// Has returns a boolean indicating if the given key exists.
func (t *Tree) Has(key string) bool {
	if key == "" {
		return false
	}
	return t.HasPath(strings.Split(key, "."))
}

// HasPath returns true if the given path of keys exists, false otherwise.
func (t *Tree) HasPath(keys []string) bool {
	return t.GetPath(keys) != nil
}

// Keys returns the keys of the toplevel tree (does not recurse).
func (t *Tree) Keys() []string {
	keys := make([]string, len(t.values))
	i := 0
	for k := range t.values {
		keys[i] = k
		i++
	}
	return keys
}

// Get the value at key in the Tree.
// Key is a dot-separated path (e.g. a.b.c) without single/double quoted strings.
// If you need to retrieve non-bare keys, use GetPath.
// Returns nil if the path does not exist in the tree.
// If keys is of length zero, the current tree is returned.
func (t *Tree) Get(key string) interface{} {
	if key == "" {
		return t
	}
	return t.GetPath(strings.Split(key, "."))
}

// GetPath returns the element in the tree indicated by 'keys'.
// If keys is of length zero, the current tree is returned.
func (t *Tree) GetPath(keys []string) interface{} {
	if len(keys) == 0 {
		return t
	}
	subtree := t
	for _, intermediateKey := range keys[:len(keys)-1] {
		value, exists := subtree.values[intermediateKey]
		if !exists {
			return nil
		}
		switch node := value.(type) {
		case *Tree:
			subtree = node
		case []*Tree:
			// go to most recent element
			if len(node) == 0 {
				return nil
			}
			subtree = node[len(node)-1]
		default:
			return nil // cannot navigate through other node types
		}
	}
	// branch based on final node type
	switch node := subtree.values[keys[len(keys)-1]].(type) {
	case *tomlValue:
		return node.value
	default:
		return node
	}
}

// GetArray returns the value at key in the Tree.
// It returns []string, []int64, etc type if key has homogeneous lists
// Key is a dot-separated path (e.g. a.b.c) without single/double quoted strings.
// Returns nil if the path does not exist in the tree.
// If keys is of length zero, the current tree is returned.
func (t *Tree) GetArray(key string) interface{} {
	if key == "" {
		return t
	}
	return t.GetArrayPath(strings.Split(key, "."))
}

// GetArrayPath returns the element in the tree indicated by 'keys'.
// If keys is of length zero, the current tree is returned.
func (t *Tree) GetArrayPath(keys []string) interface{} {
	if len(keys) == 0 {
		return t
	}
	subtree := t
	for _, intermediateKey := range keys[:len(keys)-1] {
		value, exists := subtree.values[intermediateKey]
		if !exists {
			return nil
		}
		switch node := value.(type) {
		case *Tree:
			subtree = node
		case []*Tree:
			// go to most recent element
			if len(node) == 0 {
				return nil
			}
			subtree = node[len(node)-1]
		default:
			return nil // cannot navigate through other node types
		}
	}
	// branch based on final node type
	switch node := subtree.values[keys[len(keys)-1]].(type) {
	case *tomlValue:
		switch n := node.value.(type) {
		case []interface{}:
			return getArray(n)
		default:
			return node.value
		}
	default:
		return node
	}
}

// if homogeneous array, then return slice type object over []interface{}
func getArray(n []interface{}) interface{} {
	var s []string
	var i64 []int64
	var f64 []float64
	var bl []bool
	for _, value := range n {
		switch v := value.(type) {
		case string:
			s = append(s, v)
		case int64:
			i64 = append(i64, v)
		case float64:
			f64 = append(f64, v)
		case bool:
			bl = append(bl, v)
		default:
			return n
		}
	}
	if len(s) == len(n) {
		return s
	} else if len(i64) == len(n) {
		return i64
	} else if len(f64) == len(n) {
		return f64
	} else if len(bl) == len(n) {
		return bl
	}
	return n
}

// GetPosition returns the position of the given key.
func (t *Tree) GetPosition(key string) Position {
	if key == "" {
		return t.position
	}
	return t.GetPositionPath(strings.Split(key, "."))
}

// SetPositionPath sets the position of element in the tree indicated by 'keys'.
// If keys is of length zero, the current tree position is set.
func (t *Tree) SetPositionPath(keys []string, pos Position) {
	if len(keys) == 0 {
		t.position = pos
		return
	}
	subtree := t
	for _, intermediateKey := range keys[:len(keys)-1] {
		value, exists := subtree.values[intermediateKey]
		if !exists {
			return
		}
		switch node := value.(type) {
		case *Tree:
			subtree = node
		case []*Tree:
			// go to most recent element
			if len(node) == 0 {
				return
			}
			subtree = node[len(node)-1]
		default:
			return
		}
	}
	// branch based on final node type
	switch node := subtree.values[keys[len(keys)-1]].(type) {
	case *tomlValue:
		node.position = pos
		return
	case *Tree:
		node.position = pos
		return
	case []*Tree:
		// go to most recent element
		if len(node) == 0 {
			return
		}
		node[len(node)-1].position = pos
		return
	}
}

// GetPositionPath returns the element in the tree indicated by 'keys'.
// If keys is of length zero, the current tree is returned.
func (t *Tree) GetPositionPath(keys []string) Position {
	if len(keys) == 0 {
		return t.position
	}
	subtree := t
	for _, intermediateKey := range keys[:len(keys)-1] {
		value, exists := subtree.values[intermediateKey]
		if !exists {
			return Position{0, 0}
		}
		switch node := value.(type) {
		case *Tree:
			subtree = node
		case []*Tree:
			// go to most recent element
			if len(node) == 0 {
				return Position{0, 0}
			}
			subtree = node[len(node)-1]
		default:
			return Position{0, 0}
		}
	}
	// branch based on final node type
	switch node := subtree.values[keys[len(keys)-1]].(type) {
	case *tomlValue:
		return node.position
	case *Tree:
		return node.position
	case []*Tree:
		// go to most recent element
		if len(node) == 0 {
			return Position{0, 0}
		}
		return node[len(node)-1].position
	default:
		return Position{0, 0}
	}
}

// GetDefault works like Get but with a default value
func (t *Tree) GetDefault(key string, def interface{}) interface{} {
	val := t.Get(key)
	if val == nil {
		return def
	}
	return val
}

// SetOptions arguments are supplied to the SetWithOptions and SetPathWithOptions functions to modify marshalling behaviour.
// The default values within the struct are valid default options.
type SetOptions struct {
	Comment   string
	Commented bool
	Multiline bool
	Literal   bool
}

// SetWithOptions is the same as Set, but allows you to provide formatting
// instructions to the key, that will be used by Marshal().
func (t *Tree) SetWithOptions(key string, opts SetOptions, value interface{}) {
	t.SetPathWithOptions(strings.Split(key, "."), opts, value)
}

// SetPathWithOptions is the same as SetPath, but allows you to provide
// formatting instructions to the key, that will be reused by Marshal().
func (t *Tree) SetPathWithOptions(keys []string, opts SetOptions, value interface{}) {
	subtree := t
	for i, intermediateKey := range keys[:len(keys)-1] {
		nextTree, exists := subtree.values[intermediateKey]
		if !exists {
			nextTree = newTreeWithPosition(Position{Line: t.position.Line + i, Col: t.position.Col})
			subtree.values[intermediateKey] = nextTree // add new element here
		}
		switch node := nextTree.(type) {
		case *Tree:
			subtree = node
		case []*Tree:
			// go to most recent element
			if len(node) == 0 {
				// create element if it does not exist
				node = append(node, newTreeWithPosition(Position{Line: t.position.Line + i, Col: t.position.Col}))
				subtree.values[intermediateKey] = node
			}
			subtree = node[len(node)-1]
		}
	}

	var toInsert interface{}

	switch v := value.(type) {
	case *Tree:
		v.comment = opts.Comment
		v.commented = opts.Commented
		toInsert = value
	case []*Tree:
		for i := range v {
			v[i].commented = opts.Commented
		}
		toInsert = value
	case *tomlValue:
		v.comment = opts.Comment
		v.commented = opts.Commented
		v.multiline = opts.Multiline
		v.literal = opts.Literal
		toInsert = v
	default:
		toInsert = &tomlValue{value: value,
			comment:   opts.Comment,
			commented: opts.Commented,
			multiline: opts.Multiline,
			literal:   opts.Literal,
			position:  Position{Line: subtree.position.Line + len(subtree.values) + 1, Col: subtree.position.Col}}
	}

	subtree.values[keys[len(keys)-1]] = toInsert
}

// Set an element in the tree.
// Key is a dot-separated path (e.g. a.b.c).
// Creates all necessary intermediate trees, if needed.
func (t *Tree) Set(key string, value interface{}) {
	t.SetWithComment(key, "", false, value)
}

// SetWithComment is the same as Set, but allows you to provide comment
// information to the key, that will be reused by Marshal().
func (t *Tree) SetWithComment(key string, comment string, commented bool, value interface{}) {
	t.SetPathWithComment(strings.Split(key, "."), comment, commented, value)
}

// SetPath sets an element in the tree.
// Keys is an array of path elements (e.g. {"a","b","c"}).
// Creates all necessary intermediate trees, if needed.
func (t *Tree) SetPath(keys []string, value interface{}) {
	t.SetPathWithComment(keys, "", false, value)
}

// SetPathWithComment is the same as SetPath, but allows you to provide comment
// information to the key, that will be reused by Marshal().
func (t *Tree) SetPathWithComment(keys []string, comment string, commented bool, value interface{}) {
	t.SetPathWithOptions(keys, SetOptions{Comment: comment, Commented: commented}, value)
}

// Delete removes a key from the tree.
// Key is a dot-separated path (e.g. a.b.c).
func (t *Tree) Delete(key string) error {
	keys, err := parseKey(key)
	if err != nil {
		return err
	}
	return t.DeletePath(keys)
}

// DeletePath removes a key from the tree.
// Keys is an array of path elements (e.g. {"a","b","c"}).
func (t *Tree) DeletePath(keys []string) error {
	keyLen := len(keys)
	if keyLen == 1 {
		delete(t.values, keys[0])
		return nil
	}
	tree := t.GetPath(keys[:keyLen-1])
	item := keys[keyLen-1]
	switch node := tree.(type) {
	case *Tree:
		delete(node.values, item)
		return nil
	}
	return errors.New("no such key to delete")
}

// createSubTree takes a tree and a key and create the necessary intermediate
// subtrees to create a subtree at that point. In-place.
//
// e.g. passing a.b.c will create (assuming tree is empty) tree[a], tree[a][b]
// and tree[a][b][c]
//
// Returns nil on success, error object on failure
func (t *Tree) createSubTree(keys []string, pos Position) error {
	subtree := t
	for i, intermediateKey := range keys {
		nextTree, exists := subtree.values[intermediateKey]
		if !exists {
			tree := newTreeWithPosition(Position{Line: t.position.Line + i, Col: t.position.Col})
			tree.position = pos
			tree.inline = subtree.inline
			subtree.values[intermediateKey] = tree
			nextTree = tree
		}

		switch node := nextTree.(type) {
		case []*Tree:
			subtree = node[len(node)-1]
		case *Tree:
			subtree = node
		default:
			return fmt.Errorf("unknown type for path %s (%s): %T (%#v)",
				strings.Join(keys, "."), intermediateKey, nextTree, nextTree)
		}
	}
	return nil
}

// LoadBytes creates a Tree from a []byte.
func LoadBytes(b []byte) (tree *Tree, err error) {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}
			err = errors.New(r.(string))
		}
	}()

	if len(b) >= 4 && (hasUTF32BigEndianBOM4(b) || hasUTF32LittleEndianBOM4(b)) {
		b = b[4:]
	} else if len(b) >= 3 && hasUTF8BOM3(b) {
		b = b[3:]
	} else if len(b) >= 2 && (hasUTF16BigEndianBOM2(b) || hasUTF16LittleEndianBOM2(b)) {
		b = b[2:]
	}

	tree = parseToml(lexToml(b))
	return
}

func hasUTF16BigEndianBOM2(b []byte) bool {
	return b[0] == 0xFE && b[1] == 0xFF
}

func hasUTF16LittleEndianBOM2(b []byte) bool {
	return b[0] == 0xFF && b[1] == 0xFE
}

func hasUTF8BOM3(b []byte) bool {
	return b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF
}

func hasUTF32BigEndianBOM4(b []byte) bool {
	return b[0] == 0x00 && b[1] == 0x00 && b[2] == 0xFE && b[3] == 0xFF
}

func hasUTF32LittleEndianBOM4(b []byte) bool {
	return b[0] == 0xFF && b[1] == 0xFE && b[2] == 0x00 && b[3] == 0x00
}

// LoadReader creates a Tree from any io.Reader.
func LoadReader(reader io.Reader) (tree *Tree, err error) {
	inputBytes, err := ioutil.ReadAll(reader)
	if err != nil {
		return
	}
	tree, err = LoadBytes(inputBytes)
	return
}

// Load creates a Tree from a string.
func Load(content string) (tree *Tree, err error) {
	return LoadBytes([]byte(content))
}

// LoadFile creates a Tree from a file.
func LoadFile(path string) (tree *Tree, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return LoadReader(file)
}
