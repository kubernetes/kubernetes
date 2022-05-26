/*
Package strcase is a package for converting strings into various word cases
(e.g. snake_case, camelCase)

 go get -u github.com/ettle/strcase

Example usage

 strcase.ToSnake("Hello World")     // hello_world
 strcase.ToSNAKE("Hello World")     // HELLO_WORLD

 strcase.ToKebab("helloWorld")      // hello-world
 strcase.ToKEBAB("helloWorld")      // HELLO-WORLD

 strcase.ToPascal("hello-world")    // HelloWorld
 strcase.ToCamel("hello-world")     // helloWorld

 // Handle odd cases
 strcase.ToSnake("FOOBar")          // foo_bar

 // Support Go initialisms
 strcase.ToGoPascal("http_response") // HTTPResponse

 // Specify case and delimiter
 strcase.ToCase("HelloWorld", strcase.UpperCase, '.') // HELLO.WORLD

Why this package

String strcase is pretty straight forward and there are a number of methods to
do it. This package is fully featured, more customizable, better tested, and
faster* than other packages and what you would probably whip up yourself.

Unicode support

We work for with unicode strings and pay very little performance penalty for it
as we optimized for the common use case of ASCII only strings.

Customization

You can create a custom caser that changes the behavior to what you want. This
customization also reduces the pressure for us to change the default behavior
which means that things are more stable for everyone involved.  The goal is to
make the common path easy and fast, while making the uncommon path possible.

 c := NewCaser(
	// Use Go's default initialisms e.g. ID, HTML
 	true,
	// Override initialisms (e.g. don't initialize HTML but initialize SSL
 	map[string]bool{"SSL": true, "HTML": false},
	// Write your own custom SplitFn
	//
 	NewSplitFn(
 		[]rune{'*', '.', ','},
 		SplitCase,
 		SplitAcronym,
 		PreserveNumberFormatting,
 		SplitBeforeNumber,
 		SplitAfterNumber,
 	))
 assert.Equal(t, "http_200", c.ToSnake("http200"))

Initialism support

By default, we use the golint intialisms list. You can customize and override
the initialisms if you wish to add additional ones, such as "SSL" or "CMS" or
domain specific ones to your industry.

  ToGoPascal("http_response") // HTTPResponse
  ToGoSnake("http_response") // HTTP_response

Test coverage

We have a wide ranging test suite to make sure that we understand our behavior.
Test coverage isn't everything, but we aim for 100% coverage.

Fast

Optimized to reduce memory allocations with Builder. Benchmarked and optimized
around common cases.

We're on par with the fastest packages (that have less features) and much
faster than others. We also benchmarked against code snippets. Using string
builders to reduce memory allocation and reordering boolean checks for the
common cases have a large performance impact.

Hopefully I was fair to each library and happy to rerun benchmarks differently
or reword my commentary based on suggestions or updates.

  // This package - faster then almost all libraries
  // Initialisms are more complicated and slightly slower, but still faster then other libraries that do less
  BenchmarkToTitle-4                       7821166               221 ns/op              32 B/op          1 allocs/op
  BenchmarkToSnake-4                       9378589               202 ns/op              32 B/op          1 allocs/op
  BenchmarkToSNAKE-4                       6174453               223 ns/op              32 B/op          1 allocs/op
  BenchmarkToGoSnake-4                     3114266               434 ns/op              44 B/op          4 allocs/op
  BenchmarkToCustomCaser-4                 2973855               448 ns/op              56 B/op          4 allocs/op

  // Segment has very fast snake case and camel case libraries
  // No features or customization, but very very fast
  BenchmarkSegment-4                      24003495                64.9 ns/op            16 B/op          1 allocs/op

  // Stdlib strings.Title for comparison, even though it only splits on spaces
  BenchmarkToTitleStrings-4               11259376               161 ns/op              16 B/op          1 allocs/op

  // Other libraries or code snippets
  // - Most are slower, by up to an order of magnitude
  // - None support initialisms or customization
  // - Some generate only camelCase or snake_case
  // - Many lack unicode support
  BenchmarkToSnakeStoewer-4                7103268               297 ns/op              64 B/op          2 allocs/op
  // Copying small rune arrays is slow
  BenchmarkToSnakeSiongui-4                3710768               413 ns/op              48 B/op         10 allocs/op
  BenchmarkGoValidator-4                   2416479              1049 ns/op             184 B/op          9 allocs/op
  // String alloction is slow
  BenchmarkToSnakeFatih-4                  1000000              2407 ns/op             624 B/op         26 allocs/op
  BenchmarkToSnakeIanColeman-4             1005766              1426 ns/op             160 B/op         13 allocs/op
  // Regexp is slow
  BenchmarkToSnakeGolangPrograms-4          614689              2237 ns/op             225 B/op         11 allocs/op



  // These results aren't a surprise - my initial version of this library was
  // painfully slow. I think most of us, without spending some time with
  // profilers and benchmarks, would write also something on the slower side.


Why not this package

If every nanosecond matters and this is used in a tight loop, use segment.io's
libraries (https://github.com/segmentio/go-snakecase and
https://github.com/segmentio/go-camelcase). They lack features, but make up for
it by being blazing fast. Alternatively, if you need your code to work slightly
differently, fork them and tailor it for your use case.

If you don't like having external imports, I get it. This package only imports
packages for testing, otherwise it only uses the standard library. If that's
not enough, you can use this repo as the foundation for your own. MIT Licensed.

This package is still relatively new and while I've used it for a while
personally, it doesn't have the miles that other packages do. I've tested this
code agains't their test cases to make sure that there aren't any surprises.

Migrating from other packages

If you are migrating from from another package, you may find slight differences
in output. To reduce the delta, you may find it helpful to use the following
custom casers to mimic the behavior of the other package.

  // From https://github.com/iancoleman/strcase
  var c = NewCaser(false, nil, NewSplitFn([]rune{'_', '-', '.'}, SplitCase, SplitAcronym, SplitBeforeNumber))

  // From https://github.com/stoewer/go-strcase
  var c = NewCaser(false, nil, NewSplitFn([]rune{'_', '-'}, SplitCase), SplitAcronym)

*/
package strcase
