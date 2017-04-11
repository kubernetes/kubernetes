// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package search

// Field is a name/value pair. A search index's document can be loaded and
// saved as a sequence of Fields.
type Field struct {
	// Name is the field name. A valid field name matches /[A-Za-z][A-Za-z0-9_]*/.
	Name string
	// Value is the field value. The valid types are:
	//  - string,
	//  - search.Atom,
	//  - search.HTML,
	//  - time.Time (stored with millisecond precision),
	//  - float64,
	//  - GeoPoint.
	Value interface{}
	// Language is a two-letter ISO 639-1 code for the field's language,
	// defaulting to "en" if nothing is specified. It may only be specified for
	// fields of type string and search.HTML.
	Language string
	// Derived marks fields that were calculated as a result of a
	// FieldExpression provided to Search. This field is ignored when saving a
	// document.
	Derived bool
}

// Facet is a name/value pair which is used to add categorical information to a
// document.
type Facet struct {
	// Name is the facet name. A valid facet name matches /[A-Za-z][A-Za-z0-9_]*/.
	// A facet name cannot be longer than 500 characters.
	Name string
	// Value is the facet value.
	//
	// When being used in documents (for example, in
	// DocumentMetadata.Facets), the valid types are:
	//  - search.Atom,
	//  - float64.
	//
	// When being used in SearchOptions.Refinements or being returned
	// in FacetResult, the valid types are:
	//  - search.Atom,
	//  - search.Range.
	Value interface{}
}

// DocumentMetadata is a struct containing information describing a given document.
type DocumentMetadata struct {
	// Rank is an integer specifying the order the document will be returned in
	// search results. If zero, the rank will be set to the number of seconds since
	// 2011-01-01 00:00:00 UTC when being Put into an index.
	Rank int
	// Facets is the set of facets for this document.
	Facets []Facet
}

// FieldLoadSaver can be converted from and to a slice of Fields
// with additional document metadata.
type FieldLoadSaver interface {
	Load([]Field, *DocumentMetadata) error
	Save() ([]Field, *DocumentMetadata, error)
}

// FieldList converts a []Field to implement FieldLoadSaver.
type FieldList []Field

// Load loads all of the provided fields into l.
// It does not first reset *l to an empty slice.
func (l *FieldList) Load(f []Field, _ *DocumentMetadata) error {
	*l = append(*l, f...)
	return nil
}

// Save returns all of l's fields as a slice of Fields.
func (l *FieldList) Save() ([]Field, *DocumentMetadata, error) {
	return *l, nil, nil
}

var _ FieldLoadSaver = (*FieldList)(nil)
