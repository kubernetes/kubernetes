package state

import (
	"fmt"
	"strings"

	"github.com/hashicorp/consul/consul/prepared_query"
)

// PreparedQueryIndex is a custom memdb indexer used to manage index prepared
// query templates. None of the built-in indexers do what we need, and our
// use case is pretty specific so it's better to put the logic here.
type PreparedQueryIndex struct {
}

// FromObject is used to compute the index key when inserting or updating an
// object.
func (*PreparedQueryIndex) FromObject(obj interface{}) (bool, []byte, error) {
	wrapped, ok := obj.(*queryWrapper)
	if !ok {
		return false, nil, fmt.Errorf("invalid object given to index as prepared query")
	}

	query := toPreparedQuery(wrapped)
	if !prepared_query.IsTemplate(query) {
		return false, nil, nil
	}

	// Always prepend a null so that we can represent even an empty name.
	out := "\x00" + strings.ToLower(query.Name)
	return true, []byte(out), nil
}

// FromArgs is used when querying for an exact match. Since we don't add any
// suffix we can just call the prefix version.
func (p *PreparedQueryIndex) FromArgs(args ...interface{}) ([]byte, error) {
	return p.PrefixFromArgs(args...)
}

// PrefixFromArgs is used when doing a prefix scan for an object.
func (*PreparedQueryIndex) PrefixFromArgs(args ...interface{}) ([]byte, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("must provide only a single argument")
	}
	arg, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("argument must be a string: %#v", args[0])
	}
	arg = "\x00" + strings.ToLower(arg)
	return []byte(arg), nil
}
