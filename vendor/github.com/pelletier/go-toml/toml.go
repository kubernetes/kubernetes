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
	value    interface{} // string, int64, uint64, float64, bool, time.Time, [] of any of this list
	position Position
}

// Tree is the result of the parsing of a TOML file.
type Tree struct {
	values   map[string]interface{} // string -> *tomlValue, *Tree, []*Tree
	position Position
}

func newTree() *Tree {
	return &Tree{
		values:   make(map[string]interface{}),
		position: Position{},
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
// Key is a dot-separated path (e.g. a.b.c).
// Returns nil if the path does not exist in the tree.
// If keys is of length zero, the current tree is returned.
func (t *Tree) Get(key string) interface{} {
	if key == "" {
		return t
	}
	comps, err := parseKey(key)
	if err != nil {
		return nil
	}
	return t.GetPath(comps)
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

// GetPosition returns the position of the given key.
func (t *Tree) GetPosition(key string) Position {
	if key == "" {
		return t.position
	}
	return t.GetPositionPath(strings.Split(key, "."))
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

// Set an element in the tree.
// Key is a dot-separated path (e.g. a.b.c).
// Creates all necessary intermediate trees, if needed.
func (t *Tree) Set(key string, value interface{}) {
	t.SetPath(strings.Split(key, "."), value)
}

// SetPath sets an element in the tree.
// Keys is an array of path elements (e.g. {"a","b","c"}).
// Creates all necessary intermediate trees, if needed.
func (t *Tree) SetPath(keys []string, value interface{}) {
	subtree := t
	for _, intermediateKey := range keys[:len(keys)-1] {
		nextTree, exists := subtree.values[intermediateKey]
		if !exists {
			nextTree = newTree()
			subtree.values[intermediateKey] = nextTree // add new element here
		}
		switch node := nextTree.(type) {
		case *Tree:
			subtree = node
		case []*Tree:
			// go to most recent element
			if len(node) == 0 {
				// create element if it does not exist
				subtree.values[intermediateKey] = append(node, newTree())
			}
			subtree = node[len(node)-1]
		}
	}

	var toInsert interface{}

	switch value.(type) {
	case *Tree:
		toInsert = value
	case []*Tree:
		toInsert = value
	case *tomlValue:
		toInsert = value
	default:
		toInsert = &tomlValue{value: value}
	}

	subtree.values[keys[len(keys)-1]] = toInsert
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
	for _, intermediateKey := range keys {
		nextTree, exists := subtree.values[intermediateKey]
		if !exists {
			tree := newTree()
			tree.position = pos
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
	tree = parseToml(lexToml(b))
	return
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
