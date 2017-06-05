// Copyright 2016 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import "golang.org/x/net/context"

// Datastore kinds for the metadata entities.
const (
	namespaceKind   = "__namespace__"
	kindKind        = "__kind__"
	propertyKind    = "__property__"
	entityGroupKind = "__entitygroup__"
)

// Namespaces returns all the datastore namespaces.
func Namespaces(ctx context.Context) ([]string, error) {
	// TODO(djd): Support range queries.
	q := NewQuery(namespaceKind).KeysOnly()
	keys, err := q.GetAll(ctx, nil)
	if err != nil {
		return nil, err
	}
	// The empty namespace key uses a numeric ID (==1), but luckily
	// the string ID defaults to "" for numeric IDs anyway.
	return keyNames(keys), nil
}

// Kinds returns the names of all the kinds in the current namespace.
func Kinds(ctx context.Context) ([]string, error) {
	// TODO(djd): Support range queries.
	q := NewQuery(kindKind).KeysOnly()
	keys, err := q.GetAll(ctx, nil)
	if err != nil {
		return nil, err
	}
	return keyNames(keys), nil
}

// keyNames returns a slice of the provided keys' names (string IDs).
func keyNames(keys []*Key) []string {
	n := make([]string, 0, len(keys))
	for _, k := range keys {
		n = append(n, k.StringID())
	}
	return n
}

// KindProperties returns all the indexed properties for the given kind.
// The properties are returned as a map of property names to a slice of the
// representation types. The representation types for the supported Go property
// types are:
//   "INT64":     signed integers and time.Time
//   "DOUBLE":    float32 and float64
//   "BOOLEAN":   bool
//   "STRING":    string, []byte and ByteString
//   "POINT":     appengine.GeoPoint
//   "REFERENCE": *Key
//   "USER":      (not used in the Go runtime)
func KindProperties(ctx context.Context, kind string) (map[string][]string, error) {
	// TODO(djd): Support range queries.
	kindKey := NewKey(ctx, kindKind, kind, 0, nil)
	q := NewQuery(propertyKind).Ancestor(kindKey)

	propMap := map[string][]string{}
	props := []struct {
		Repr []string `datastore:property_representation`
	}{}

	keys, err := q.GetAll(ctx, &props)
	if err != nil {
		return nil, err
	}
	for i, p := range props {
		propMap[keys[i].StringID()] = p.Repr
	}
	return propMap, nil
}
