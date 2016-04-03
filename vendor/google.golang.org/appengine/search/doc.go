// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package search provides a client for App Engine's search service.


Basic Operations

Indexes contain documents. Each index is identified by its name: a
human-readable ASCII string.

Within an index, documents are associated with an ID, which is also
a human-readable ASCII string. A document's contents are a mapping from
case-sensitive field names to values. Valid types for field values are:
  - string,
  - search.Atom,
  - search.HTML,
  - time.Time (stored with millisecond precision),
  - float64 (value between -2,147,483,647 and 2,147,483,647 inclusive),
  - appengine.GeoPoint.

The Get and Put methods on an Index load and save a document.
A document's contents are typically represented by a struct pointer.

Example code:

	type Doc struct {
		Author   string
		Comment  string
		Creation time.Time
	}

	index, err := search.Open("comments")
	if err != nil {
		return err
	}
	newID, err := index.Put(ctx, "", &Doc{
		Author:   "gopher",
		Comment:  "the truth of the matter",
		Creation: time.Now(),
	})
	if err != nil {
		return err
	}

A single document can be retrieved by its ID. Pass a destination struct
to Get to hold the resulting document.

	var doc Doc
	err := index.Get(ctx, id, &doc)
	if err != nil {
		return err
	}


Search and Listing Documents

Indexes have two methods for retrieving multiple documents at once: Search and
List.

Searching an index for a query will result in an iterator. As with an iterator
from package datastore, pass a destination struct to Next to decode the next
result. Next will return Done when the iterator is exhausted.

	for t := index.Search(ctx, "Comment:truth", nil); ; {
		var doc Doc
		id, err := t.Next(&doc)
		if err == search.Done {
			break
		}
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s -> %#v\n", id, doc)
	}

Search takes a string query to determine which documents to return. The query
can be simple, such as a single word to match, or complex. The query
language is described at
https://cloud.google.com/appengine/docs/go/search/query_strings

Search also takes an optional SearchOptions struct which gives much more
control over how results are calculated and returned.

Call List to iterate over all documents in an index.

	for t := index.List(ctx, nil); ; {
		var doc Doc
		id, err := t.Next(&doc)
		if err == search.Done {
			break
		}
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s -> %#v\n", id, doc)
	}


Fields and Facets

A document's contents can be represented by a variety of types. These are
typically struct pointers, but they can also be represented by any type
implementing the FieldLoadSaver interface. The FieldLoadSaver allows metadata
to be set for the document with the DocumentMetadata type. Struct pointers are
more strongly typed and are easier to use; FieldLoadSavers are more flexible.

A document's contents can be expressed in two ways: fields and facets.

Fields are the most common way of providing content for documents. Fields can
store data in multiple types and can be matched in searches using query
strings.

Facets provide a way to attach categorical information to a document. The only
valid types for facets are search.Atom and float64. Facets allow search
results to contain summaries of the categories matched in a search, and to
restrict searches to only match against specific categories.

By default, for struct pointers, all of the struct fields are used as document
fields, and the field name used is the same as on the struct (and hence must
start with an upper case letter). Struct fields may have a
`search:"name,options"` tag. The name must start with a letter and be
composed only of word characters. A "-" tag name means that the field will be
ignored.  If options is "facet" then the struct field will be used as a
document facet. If options is "" then the comma may be omitted. There are no
other recognized options.

Example code:

	// A and B are renamed to a and b.
	// A, C and I are facets.
	// D's tag is equivalent to having no tag at all (E).
	// F and G are ignored entirely by the search package.
	// I has tag information for both the search and json packages.
	type TaggedStruct struct {
		A float64 `search:"a,facet"`
		B float64 `search:"b"`
		C float64 `search:",facet"`
		D float64 `search:""`
		E float64
		F float64 `search:"-"`
		G float64 `search:"-,facet"`
		I float64 `search:",facet" json:"i"`
	}


The FieldLoadSaver Interface

A document's contents can also be represented by any type that implements the
FieldLoadSaver interface. This type may be a struct pointer, but it
does not have to be. The search package will call Load when loading the
document's contents, and Save when saving them. In addition to a slice of
Fields, the Load and Save methods also use the DocumentMetadata type to
provide additional information about a document (such as its Rank, or set of
Facets). Possible uses for this interface include deriving non-stored fields,
verifying fields or setting specific languages for string and HTML fields.

Example code:

	type CustomFieldsExample struct {
		// Item's title and which language it is in.
		Title string
		Lang  string
		// Mass, in grams.
		Mass int
	}

	func (x *CustomFieldsExample) Load(fields []search.Field, meta *search.DocumentMetadata) error {
		// Load the title field, failing if any other field is found.
		for _, f := range fields {
			if f.Name != "title" {
				return fmt.Errorf("unknown field %q", f.Name)
			}
			s, ok := f.Value.(string)
			if !ok {
				return fmt.Errorf("unsupported type %T for field %q", f.Value, f.Name)
			}
			x.Title = s
			x.Lang = f.Language
		}
		// Load the mass facet, failing if any other facet is found.
		for _, f := range meta.Facets {
			if f.Name != "mass" {
				return fmt.Errorf("unknown facet %q", f.Name)
			}
			m, ok := f.Value.(float64)
			if !ok {
				return fmt.Errorf("unsupported type %T for facet %q", f.Value, f.Name)
			}
			x.Mass = int(m)
		}
		return nil
	}

	func (x *CustomFieldsExample) Save() ([]search.Field, *search.DocumentMetadata, error) {
		fields := []search.Field{
			{Name: "title", Value: x.Title, Language: x.Lang},
		}
		meta := &search.DocumentMetadata{
			Facets: {
				{Name: "mass", Value: float64(x.Mass)},
			},
		}
		return fields, meta, nil
	}
*/
package search
