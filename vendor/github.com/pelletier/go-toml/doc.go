// Package toml is a TOML markup language parser.
//
// This version supports the specification as described in
// https://github.com/toml-lang/toml/blob/master/versions/en/toml-v0.4.0.md
//
// TOML Parsing
//
// TOML data may be parsed in two ways: by file, or by string.
//
//   // load TOML data by filename
//   tree, err := toml.LoadFile("filename.toml")
//
//   // load TOML data stored in a string
//   tree, err := toml.Load(stringContainingTomlData)
//
// Either way, the result is a TomlTree object that can be used to navigate the
// structure and data within the original document.
//
//
// Getting data from the TomlTree
//
// After parsing TOML data with Load() or LoadFile(), use the Has() and Get()
// methods on the returned TomlTree, to find your way through the document data.
//
//   if tree.Has('foo') {
//     fmt.Prinln("foo is: %v", tree.Get('foo'))
//   }
//
// Working with Paths
//
// Go-toml has support for basic dot-separated key paths on the Has(), Get(), Set()
// and GetDefault() methods.  These are the same kind of key paths used within the
// TOML specification for struct tames.
//
//   // looks for a key named 'baz', within struct 'bar', within struct 'foo'
//   tree.Has("foo.bar.baz")
//
//   // returns the key at this path, if it is there
//   tree.Get("foo.bar.baz")
//
// TOML allows keys to contain '.', which can cause this syntax to be problematic
// for some documents.  In such cases, use the GetPath(), HasPath(), and SetPath(),
// methods to explicitly define the path.  This form is also faster, since
// it avoids having to parse the passed key for '.' delimiters.
//
//   // looks for a key named 'baz', within struct 'bar', within struct 'foo'
//   tree.HasPath(string{}{"foo","bar","baz"})
//
//   // returns the key at this path, if it is there
//   tree.GetPath(string{}{"foo","bar","baz"})
//
// Note that this is distinct from the heavyweight query syntax supported by
// TomlTree.Query() and the Query() struct (see below).
//
// Position Support
//
// Each element within the TomlTree is stored with position metadata, which is
// invaluable for providing semantic feedback to a user.  This helps in
// situations where the TOML file parses correctly, but contains data that is
// not correct for the application.  In such cases, an error message can be
// generated that indicates the problem line and column number in the source
// TOML document.
//
//   // load TOML data
//   tree, _ := toml.Load("filename.toml")
//
//   // get an entry and report an error if it's the wrong type
//   element := tree.Get("foo")
//   if value, ok := element.(int64); !ok {
//       return fmt.Errorf("%v: Element 'foo' must be an integer", tree.GetPosition("foo"))
//   }
//
//   // report an error if an expected element is missing
//   if !tree.Has("bar") {
//      return fmt.Errorf("%v: Expected 'bar' element", tree.GetPosition(""))
//   }
//
// Query Support
//
// The TOML query path implementation is based loosely on the JSONPath specification:
// http://goessner.net/articles/JsonPath/
//
// The idea behind a query path is to allow quick access to any element, or set
// of elements within TOML document, with a single expression.
//
//   result, err := tree.Query("$.foo.bar.baz")
//
// This is roughly equivalent to:
//
//   next := tree.Get("foo")
//   if next != nil {
//     next = next.Get("bar")
//     if next != nil {
//       next = next.Get("baz")
//     }
//   }
//   result := next
//
// err is nil if any parsing exception occurs.
//
// If no node in the tree matches the query, result will simply contain an empty list of
// items.
//
// As illustrated above, the query path is much more efficient, especially since
// the structure of the TOML file can vary.  Rather than making assumptions about
// a document's structure, a query allows the programmer to make structured
// requests into the document, and get zero or more values as a result.
//
// The syntax of a query begins with a root token, followed by any number
// sub-expressions:
//
//   $
//                    Root of the TOML tree.  This must always come first.
//   .name
//                    Selects child of this node, where 'name' is a TOML key
//                    name.
//   ['name']
//                    Selects child of this node, where 'name' is a string
//                    containing a TOML key name.
//   [index]
//                    Selcts child array element at 'index'.
//   ..expr
//                    Recursively selects all children, filtered by an a union,
//                    index, or slice expression.
//   ..*
//                    Recursive selection of all nodes at this point in the
//                    tree.
//   .*
//                    Selects all children of the current node.
//   [expr,expr]
//                    Union operator - a logical 'or' grouping of two or more
//                    sub-expressions: index, key name, or filter.
//   [start:end:step]
//                    Slice operator - selects array elements from start to
//                    end-1, at the given step.  All three arguments are
//                    optional.
//   [?(filter)]
//                    Named filter expression - the function 'filter' is
//                    used to filter children at this node.
//
// Query Indexes And Slices
//
// Index expressions perform no bounds checking, and will contribute no
// values to the result set if the provided index or index range is invalid.
// Negative indexes represent values from the end of the array, counting backwards.
//
//   // select the last index of the array named 'foo'
//   tree.Query("$.foo[-1]")
//
// Slice expressions are supported, by using ':' to separate a start/end index pair.
//
//   // select up to the first five elements in the array
//   tree.Query("$.foo[0:5]")
//
// Slice expressions also allow negative indexes for the start and stop
// arguments.
//
//   // select all array elements.
//   tree.Query("$.foo[0:-1]")
//
// Slice expressions may have an optional stride/step parameter:
//
//   // select every other element
//   tree.Query("$.foo[0:-1:2]")
//
// Slice start and end parameters are also optional:
//
//   // these are all equivalent and select all the values in the array
//   tree.Query("$.foo[:]")
//   tree.Query("$.foo[0:]")
//   tree.Query("$.foo[:-1]")
//   tree.Query("$.foo[0:-1:]")
//   tree.Query("$.foo[::1]")
//   tree.Query("$.foo[0::1]")
//   tree.Query("$.foo[:-1:1]")
//   tree.Query("$.foo[0:-1:1]")
//
// Query Filters
//
// Query filters are used within a Union [,] or single Filter [] expression.
// A filter only allows nodes that qualify through to the next expression,
// and/or into the result set.
//
//   // returns children of foo that are permitted by the 'bar' filter.
//   tree.Query("$.foo[?(bar)]")
//
// There are several filters provided with the library:
//
//   tree
//          Allows nodes of type TomlTree.
//   int
//          Allows nodes of type int64.
//   float
//          Allows nodes of type float64.
//   string
//          Allows nodes of type string.
//   time
//          Allows nodes of type time.Time.
//   bool
//          Allows nodes of type bool.
//
// Query Results
//
// An executed query returns a QueryResult object.  This contains the nodes
// in the TOML tree that qualify the query expression.  Position information
// is also available for each value in the set.
//
//   // display the results of a query
//   results := tree.Query("$.foo.bar.baz")
//   for idx, value := results.Values() {
//       fmt.Println("%v: %v", results.Positions()[idx], value)
//   }
//
// Compiled Queries
//
// Queries may be executed directly on a TomlTree object, or compiled ahead
// of time and executed discretely.  The former is more convienent, but has the
// penalty of having to recompile the query expression each time.
//
//   // basic query
//   results := tree.Query("$.foo.bar.baz")
//
//   // compiled query
//   query := toml.CompileQuery("$.foo.bar.baz")
//   results := query.Execute(tree)
//
//   // run the compiled query again on a different tree
//   moreResults := query.Execute(anotherTree)
//
// User Defined Query Filters
//
// Filter expressions may also be user defined by using the SetFilter()
// function on the Query object.  The function must return true/false, which
// signifies if the passed node is kept or discarded, respectively.
//
//   // create a query that references a user-defined filter
//   query, _ := CompileQuery("$[?(bazOnly)]")
//
//   // define the filter, and assign it to the query
//   query.SetFilter("bazOnly", func(node interface{}) bool{
//       if tree, ok := node.(*TomlTree); ok {
//           return tree.Has("baz")
//       }
//       return false  // reject all other node types
//   })
//
//   // run the query
//   query.Execute(tree)
//
package toml
