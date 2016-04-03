// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package datastore provides a client for App Engine's datastore service.


Basic Operations

Entities are the unit of storage and are associated with a key. A key
consists of an optional parent key, a string application ID, a string kind
(also known as an entity type), and either a StringID or an IntID. A
StringID is also known as an entity name or key name.

It is valid to create a key with a zero StringID and a zero IntID; this is
called an incomplete key, and does not refer to any saved entity. Putting an
entity into the datastore under an incomplete key will cause a unique key
to be generated for that entity, with a non-zero IntID.

An entity's contents are a mapping from case-sensitive field names to values.
Valid value types are:
  - signed integers (int, int8, int16, int32 and int64),
  - bool,
  - string,
  - float32 and float64,
  - []byte (up to 1 megabyte in length),
  - any type whose underlying type is one of the above predeclared types,
  - ByteString,
  - *Key,
  - time.Time (stored with microsecond precision),
  - appengine.BlobKey,
  - appengine.GeoPoint,
  - structs whose fields are all valid value types,
  - slices of any of the above.

Slices of structs are valid, as are structs that contain slices. However, if
one struct contains another, then at most one of those can be repeated. This
disqualifies recursively defined struct types: any struct T that (directly or
indirectly) contains a []T.

The Get and Put functions load and save an entity's contents. An entity's
contents are typically represented by a struct pointer.

Example code:

	type Entity struct {
		Value string
	}

	func handle(w http.ResponseWriter, r *http.Request) {
		ctx := appengine.NewContext(r)

		k := datastore.NewKey(ctx, "Entity", "stringID", 0, nil)
		e := new(Entity)
		if err := datastore.Get(ctx, k, e); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		old := e.Value
		e.Value = r.URL.Path

		if _, err := datastore.Put(ctx, k, e); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintf(w, "old=%q\nnew=%q\n", old, e.Value)
	}

GetMulti, PutMulti and DeleteMulti are batch versions of the Get, Put and
Delete functions. They take a []*Key instead of a *Key, and may return an
appengine.MultiError when encountering partial failure.


Properties

An entity's contents can be represented by a variety of types. These are
typically struct pointers, but can also be any type that implements the
PropertyLoadSaver interface. If using a struct pointer, you do not have to
explicitly implement the PropertyLoadSaver interface; the datastore will
automatically convert via reflection. If a struct pointer does implement that
interface then those methods will be used in preference to the default
behavior for struct pointers. Struct pointers are more strongly typed and are
easier to use; PropertyLoadSavers are more flexible.

The actual types passed do not have to match between Get and Put calls or even
across different App Engine requests. It is valid to put a *PropertyList and
get that same entity as a *myStruct, or put a *myStruct0 and get a *myStruct1.
Conceptually, any entity is saved as a sequence of properties, and is loaded
into the destination value on a property-by-property basis. When loading into
a struct pointer, an entity that cannot be completely represented (such as a
missing field) will result in an ErrFieldMismatch error but it is up to the
caller whether this error is fatal, recoverable or ignorable.

By default, for struct pointers, all properties are potentially indexed, and
the property name is the same as the field name (and hence must start with an
upper case letter). Fields may have a `datastore:"name,options"` tag. The tag
name is the property name, which must be one or more valid Go identifiers
joined by ".", but may start with a lower case letter. An empty tag name means
to just use the field name. A "-" tag name means that the datastore will
ignore that field. If options is "noindex" then the field will not be indexed.
If the options is "" then the comma may be omitted. There are no other
recognized options.

Fields (except for []byte) are indexed by default. Strings longer than 1500
bytes cannot be indexed; fields used to store long strings should be
tagged with "noindex". Similarly, ByteStrings longer than 1500 bytes cannot be
indexed.

Example code:

	// A and B are renamed to a and b.
	// A, C and J are not indexed.
	// D's tag is equivalent to having no tag at all (E).
	// I is ignored entirely by the datastore.
	// J has tag information for both the datastore and json packages.
	type TaggedStruct struct {
		A int `datastore:"a,noindex"`
		B int `datastore:"b"`
		C int `datastore:",noindex"`
		D int `datastore:""`
		E int
		I int `datastore:"-"`
		J int `datastore:",noindex" json:"j"`
	}


Structured Properties

If the struct pointed to contains other structs, then the nested or embedded
structs are flattened. For example, given these definitions:

	type Inner1 struct {
		W int32
		X string
	}

	type Inner2 struct {
		Y float64
	}

	type Inner3 struct {
		Z bool
	}

	type Outer struct {
		A int16
		I []Inner1
		J Inner2
		Inner3
	}

then an Outer's properties would be equivalent to those of:

	type OuterEquivalent struct {
		A     int16
		IDotW []int32  `datastore:"I.W"`
		IDotX []string `datastore:"I.X"`
		JDotY float64  `datastore:"J.Y"`
		Z     bool
	}

If Outer's embedded Inner3 field was tagged as `datastore:"Foo"` then the
equivalent field would instead be: FooDotZ bool `datastore:"Foo.Z"`.

If an outer struct is tagged "noindex" then all of its implicit flattened
fields are effectively "noindex".


The PropertyLoadSaver Interface

An entity's contents can also be represented by any type that implements the
PropertyLoadSaver interface. This type may be a struct pointer, but it does
not have to be. The datastore package will call Load when getting the entity's
contents, and Save when putting the entity's contents.
Possible uses include deriving non-stored fields, verifying fields, or indexing
a field only if its value is positive.

Example code:

	type CustomPropsExample struct {
		I, J int
		// Sum is not stored, but should always be equal to I + J.
		Sum int `datastore:"-"`
	}

	func (x *CustomPropsExample) Load(ps []datastore.Property) error {
		// Load I and J as usual.
		if err := datastore.LoadStruct(x, ps); err != nil {
			return err
		}
		// Derive the Sum field.
		x.Sum = x.I + x.J
		return nil
	}

	func (x *CustomPropsExample) Save() ([]datastore.Property, error) {
		// Validate the Sum field.
		if x.Sum != x.I + x.J {
			return errors.New("CustomPropsExample has inconsistent sum")
		}
		// Save I and J as usual. The code below is equivalent to calling
		// "return datastore.SaveStruct(x)", but is done manually for
		// demonstration purposes.
		return []datastore.Property{
			{
				Name:  "I",
				Value: int64(x.I),
			},
			{
				Name:  "J",
				Value: int64(x.J),
			},
		}
	}

The *PropertyList type implements PropertyLoadSaver, and can therefore hold an
arbitrary entity's contents.


Queries

Queries retrieve entities based on their properties or key's ancestry. Running
a query yields an iterator of results: either keys or (key, entity) pairs.
Queries are re-usable and it is safe to call Query.Run from concurrent
goroutines. Iterators are not safe for concurrent use.

Queries are immutable, and are either created by calling NewQuery, or derived
from an existing query by calling a method like Filter or Order that returns a
new query value. A query is typically constructed by calling NewQuery followed
by a chain of zero or more such methods. These methods are:
  - Ancestor and Filter constrain the entities returned by running a query.
  - Order affects the order in which they are returned.
  - Project constrains the fields returned.
  - Distinct de-duplicates projected entities.
  - KeysOnly makes the iterator return only keys, not (key, entity) pairs.
  - Start, End, Offset and Limit define which sub-sequence of matching entities
    to return. Start and End take cursors, Offset and Limit take integers. Start
    and Offset affect the first result, End and Limit affect the last result.
    If both Start and Offset are set, then the offset is relative to Start.
    If both End and Limit are set, then the earliest constraint wins. Limit is
    relative to Start+Offset, not relative to End. As a special case, a
    negative limit means unlimited.

Example code:

	type Widget struct {
		Description string
		Price       int
	}

	func handle(w http.ResponseWriter, r *http.Request) {
		ctx := appengine.NewContext(r)
		q := datastore.NewQuery("Widget").
			Filter("Price <", 1000).
			Order("-Price")
		b := new(bytes.Buffer)
		for t := q.Run(ctx); ; {
			var x Widget
			key, err := t.Next(&x)
			if err == datastore.Done {
				break
			}
			if err != nil {
				serveError(ctx, w, err)
				return
			}
			fmt.Fprintf(b, "Key=%v\nWidget=%#v\n\n", key, x)
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		io.Copy(w, b)
	}


Transactions

RunInTransaction runs a function in a transaction.

Example code:

	type Counter struct {
		Count int
	}

	func inc(ctx context.Context, key *datastore.Key) (int, error) {
		var x Counter
		if err := datastore.Get(ctx, key, &x); err != nil && err != datastore.ErrNoSuchEntity {
			return 0, err
		}
		x.Count++
		if _, err := datastore.Put(ctx, key, &x); err != nil {
			return 0, err
		}
		return x.Count, nil
	}

	func handle(w http.ResponseWriter, r *http.Request) {
		ctx := appengine.NewContext(r)
		var count int
		err := datastore.RunInTransaction(ctx, func(ctx context.Context) error {
			var err1 error
			count, err1 = inc(ctx, datastore.NewKey(ctx, "Counter", "singleton", 0, nil))
			return err1
		}, nil)
		if err != nil {
			serveError(ctx, w, err)
			return
		}
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintf(w, "Count=%d", count)
	}


Metadata

The datastore package provides access to some of App Engine's datastore
metadata. This metadata includes information about the entity groups,
namespaces, entity kinds, and properties in the datastore, as well as the
property representations for each property.

Example code:

	func handle(w http.ResponseWriter, r *http.Request) {
		// Print all the kinds in the datastore, with all the indexed
		// properties (and their representations) for each.
		ctx := appengine.NewContext(r)

		kinds, err := datastore.Kinds(ctx)
		if err != nil {
			serveError(ctx, w, err)
			return
		}

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		for _, kind := range kinds {
			fmt.Fprintf(w, "%s:\n", kind)
			props, err := datastore.KindProperties(ctx, kind)
			if err != nil {
				fmt.Fprintln(w, "\t(unable to retrieve properties)")
				continue
			}
			for p, rep := range props {
				fmt.Fprintf(w, "\t-%s (%s)\n", p, strings.Join(", ", rep))
			}
		}
	}
*/
package datastore // import "google.golang.org/appengine/datastore"
