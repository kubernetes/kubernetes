package toml

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
)

type tomlValue struct {
	value    interface{}
	position Position
}

// TomlTree is the result of the parsing of a TOML file.
type TomlTree struct {
	values   map[string]interface{}
	position Position
}

func newTomlTree() *TomlTree {
	return &TomlTree{
		values:   make(map[string]interface{}),
		position: Position{},
	}
}

// TreeFromMap initializes a new TomlTree object using the given map.
func TreeFromMap(m map[string]interface{}) *TomlTree {
	return &TomlTree{
		values: m,
	}
}

// Has returns a boolean indicating if the given key exists.
func (t *TomlTree) Has(key string) bool {
	if key == "" {
		return false
	}
	return t.HasPath(strings.Split(key, "."))
}

// HasPath returns true if the given path of keys exists, false otherwise.
func (t *TomlTree) HasPath(keys []string) bool {
	return t.GetPath(keys) != nil
}

// Keys returns the keys of the toplevel tree.
// Warning: this is a costly operation.
func (t *TomlTree) Keys() []string {
	var keys []string
	for k := range t.values {
		keys = append(keys, k)
	}
	return keys
}

// Get the value at key in the TomlTree.
// Key is a dot-separated path (e.g. a.b.c).
// Returns nil if the path does not exist in the tree.
// If keys is of length zero, the current tree is returned.
func (t *TomlTree) Get(key string) interface{} {
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
func (t *TomlTree) GetPath(keys []string) interface{} {
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
		case *TomlTree:
			subtree = node
		case []*TomlTree:
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
func (t *TomlTree) GetPosition(key string) Position {
	if key == "" {
		return t.position
	}
	return t.GetPositionPath(strings.Split(key, "."))
}

// GetPositionPath returns the element in the tree indicated by 'keys'.
// If keys is of length zero, the current tree is returned.
func (t *TomlTree) GetPositionPath(keys []string) Position {
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
		case *TomlTree:
			subtree = node
		case []*TomlTree:
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
	case *TomlTree:
		return node.position
	case []*TomlTree:
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
func (t *TomlTree) GetDefault(key string, def interface{}) interface{} {
	val := t.Get(key)
	if val == nil {
		return def
	}
	return val
}

// Set an element in the tree.
// Key is a dot-separated path (e.g. a.b.c).
// Creates all necessary intermediates trees, if needed.
func (t *TomlTree) Set(key string, value interface{}) {
	t.SetPath(strings.Split(key, "."), value)
}

// SetPath sets an element in the tree.
// Keys is an array of path elements (e.g. {"a","b","c"}).
// Creates all necessary intermediates trees, if needed.
func (t *TomlTree) SetPath(keys []string, value interface{}) {
	subtree := t
	for _, intermediateKey := range keys[:len(keys)-1] {
		nextTree, exists := subtree.values[intermediateKey]
		if !exists {
			nextTree = newTomlTree()
			subtree.values[intermediateKey] = nextTree // add new element here
		}
		switch node := nextTree.(type) {
		case *TomlTree:
			subtree = node
		case []*TomlTree:
			// go to most recent element
			if len(node) == 0 {
				// create element if it does not exist
				subtree.values[intermediateKey] = append(node, newTomlTree())
			}
			subtree = node[len(node)-1]
		}
	}

	var toInsert interface{}

	switch value.(type) {
	case *TomlTree:
		toInsert = value
	case []*TomlTree:
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
func (t *TomlTree) createSubTree(keys []string, pos Position) error {
	subtree := t
	for _, intermediateKey := range keys {
		if intermediateKey == "" {
			return fmt.Errorf("empty intermediate table")
		}
		nextTree, exists := subtree.values[intermediateKey]
		if !exists {
			tree := newTomlTree()
			tree.position = pos
			subtree.values[intermediateKey] = tree
			nextTree = tree
		}

		switch node := nextTree.(type) {
		case []*TomlTree:
			subtree = node[len(node)-1]
		case *TomlTree:
			subtree = node
		default:
			return fmt.Errorf("unknown type for path %s (%s): %T (%#v)",
				strings.Join(keys, "."), intermediateKey, nextTree, nextTree)
		}
	}
	return nil
}

// Query compiles and executes a query on a tree and returns the query result.
func (t *TomlTree) Query(query string) (*QueryResult, error) {
	q, err := CompileQuery(query)
	if err != nil {
		return nil, err
	}
	return q.Execute(t), nil
}

// LoadReader creates a TomlTree from any io.Reader.
func LoadReader(reader io.Reader) (tree *TomlTree, err error) {
	defer func() {
		if r := recover(); r != nil {
			if _, ok := r.(runtime.Error); ok {
				panic(r)
			}
			err = errors.New(r.(string))
		}
	}()
	tree = parseToml(lexToml(reader))
	return
}

// Load creates a TomlTree from a string.
func Load(content string) (tree *TomlTree, err error) {
	return LoadReader(strings.NewReader(content))
}

// LoadFile creates a TomlTree from a file.
func LoadFile(path string) (tree *TomlTree, err error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	return LoadReader(file)
}
