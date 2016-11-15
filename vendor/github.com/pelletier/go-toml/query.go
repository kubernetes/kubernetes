package toml

import (
	"time"
)

// NodeFilterFn represents a user-defined filter function, for use with
// Query.SetFilter().
//
// The return value of the function must indicate if 'node' is to be included
// at this stage of the TOML path.  Returning true will include the node, and
// returning false will exclude it.
//
// NOTE: Care should be taken to write script callbacks such that they are safe
// to use from multiple goroutines.
type NodeFilterFn func(node interface{}) bool

// QueryResult is the result of Executing a Query.
type QueryResult struct {
	items     []interface{}
	positions []Position
}

// appends a value/position pair to the result set.
func (r *QueryResult) appendResult(node interface{}, pos Position) {
	r.items = append(r.items, node)
	r.positions = append(r.positions, pos)
}

// Values is a set of values within a QueryResult.  The order of values is not
// guaranteed to be in document order, and may be different each time a query is
// executed.
func (r QueryResult) Values() []interface{} {
	values := make([]interface{}, len(r.items))
	for i, v := range r.items {
		o, ok := v.(*tomlValue)
		if ok {
			values[i] = o.value
		} else {
			values[i] = v
		}
	}
	return values
}

// Positions is a set of positions for values within a QueryResult.  Each index
// in Positions() corresponds to the entry in Value() of the same index.
func (r QueryResult) Positions() []Position {
	return r.positions
}

// runtime context for executing query paths
type queryContext struct {
	result       *QueryResult
	filters      *map[string]NodeFilterFn
	lastPosition Position
}

// generic path functor interface
type pathFn interface {
	setNext(next pathFn)
	call(node interface{}, ctx *queryContext)
}

// A Query is the representation of a compiled TOML path.  A Query is safe
// for concurrent use by multiple goroutines.
type Query struct {
	root    pathFn
	tail    pathFn
	filters *map[string]NodeFilterFn
}

func newQuery() *Query {
	return &Query{
		root:    nil,
		tail:    nil,
		filters: &defaultFilterFunctions,
	}
}

func (q *Query) appendPath(next pathFn) {
	if q.root == nil {
		q.root = next
	} else {
		q.tail.setNext(next)
	}
	q.tail = next
	next.setNext(newTerminatingFn()) // init the next functor
}

// CompileQuery compiles a TOML path expression.  The returned Query can be used
// to match elements within a TomlTree and its descendants.
func CompileQuery(path string) (*Query, error) {
	return parseQuery(lexQuery(path))
}

// Execute executes a query against a TomlTree, and returns the result of the query.
func (q *Query) Execute(tree *TomlTree) *QueryResult {
	result := &QueryResult{
		items:     []interface{}{},
		positions: []Position{},
	}
	if q.root == nil {
		result.appendResult(tree, tree.GetPosition(""))
	} else {
		ctx := &queryContext{
			result:  result,
			filters: q.filters,
		}
		q.root.call(tree, ctx)
	}
	return result
}

// SetFilter sets a user-defined filter function.  These may be used inside
// "?(..)" query expressions to filter TOML document elements within a query.
func (q *Query) SetFilter(name string, fn NodeFilterFn) {
	if q.filters == &defaultFilterFunctions {
		// clone the static table
		q.filters = &map[string]NodeFilterFn{}
		for k, v := range defaultFilterFunctions {
			(*q.filters)[k] = v
		}
	}
	(*q.filters)[name] = fn
}

var defaultFilterFunctions = map[string]NodeFilterFn{
	"tree": func(node interface{}) bool {
		_, ok := node.(*TomlTree)
		return ok
	},
	"int": func(node interface{}) bool {
		_, ok := node.(int64)
		return ok
	},
	"float": func(node interface{}) bool {
		_, ok := node.(float64)
		return ok
	},
	"string": func(node interface{}) bool {
		_, ok := node.(string)
		return ok
	},
	"time": func(node interface{}) bool {
		_, ok := node.(time.Time)
		return ok
	},
	"bool": func(node interface{}) bool {
		_, ok := node.(bool)
		return ok
	},
}
