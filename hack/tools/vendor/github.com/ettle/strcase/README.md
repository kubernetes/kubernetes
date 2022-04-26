
# Go Strcase

[![Go Report Card](https://goreportcard.com/badge/github.com/ettle/strcase)](https://goreportcard.com/report/github.com/ettle/strcase)
[![Coverage](http://gocover.io/_badge/github.com/ettle/strcase?0)](http://gocover.io/github.com/ettle/strcase)
[![GoDoc](https://godoc.org/github.com/ettle/strcase?status.svg)](https://pkg.go.dev/github.com/ettle/strcase)

Convert strings to `snake_case`, `camelCase`, `PascalCase`, `kebab-case` and more! Supports Go initialisms, customization, and Unicode.

`import "github.com/ettle/strcase"`

## <a name="pkg-overview">Overview</a>
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
	strcase.ToGoCamel("http_response") // HTTPResponse
	
	// Specify case and delimiter
	strcase.ToCase("HelloWorld", strcase.UpperCase, '.') // HELLO.WORLD

### Why this package
String strcase is pretty straight forward and there are a number of methods to
do it. This package is fully featured, more customizable, better tested, and
faster* than other packages and what you would probably whip up yourself.

### Unicode support
We work for with unicode strings and pay very little performance penalty for it
as we optimized for the common use case of ASCII only strings.

### Customization
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

### Initialism support
By default, we use the golint intialisms list. You can customize and override
the initialisms if you wish to add additional ones, such as "SSL" or "CMS" or
domain specific ones to your industry.


	ToGoCamel("http_response") // HTTPResponse
	ToGoSnake("http_response") // HTTP_response

### Test coverage
We have a wide ranging test suite to make sure that we understand our behavior.
Test coverage isn't everything, but we aim for 100% coverage.

### Fast
Optimized to reduce memory allocations with Builder. Benchmarked and optimized
around common cases.

We're on par with the fastest packages (that have less features) and much
faster than others. We also benchmarked against code snippets. Using string
builders to reduce memory allocation and reordering boolean checks for the
common cases have a large performance impact.

Hopefully I was fair to each library and happy to rerun benchmarks differently
or reword my commentary based on suggestions or updates.


	// This package
	// Go intialisms and custom casers are slower
	BenchmarkToTitle-4                992491              1559 ns/op              32 B/op          1 allocs/op
	BenchmarkToSnake-4               1000000              1475 ns/op              32 B/op          1 allocs/op
	BenchmarkToSNAKE-4               1000000              1609 ns/op              32 B/op          1 allocs/op
	BenchmarkToGoSnake-4              275010              3697 ns/op              44 B/op          4 allocs/op
	BenchmarkToCustomCaser-4          342704              4191 ns/op              56 B/op          4 allocs/op
	
	// Segment has very fast snake case and camel case libraries
	// No features or customization, but very very fast
	BenchmarkSegment-4               1303809               938 ns/op              16 B/op          1 allocs/op
	
	// Stdlib strings.Title for comparison, even though it only splits on spaces
	BenchmarkToTitleStrings-4        1213467              1164 ns/op              16 B/op          1 allocs/op
	
	// Other libraries or code snippets
	// - Most are slower, by up to an order of magnitude
	// - None support initialisms or customization
	// - Some generate only camelCase or snake_case
	// - Many lack unicode support
	BenchmarkToSnakeStoewer-4         973200              2075 ns/op              64 B/op          2 allocs/op
	// Copying small rune arrays is slow
	BenchmarkToSnakeSiongui-4         264315              4229 ns/op              48 B/op         10 allocs/op
	BenchmarkGoValidator-4            206811              5152 ns/op             184 B/op          9 allocs/op
	// String alloction is slow
	BenchmarkToSnakeFatih-4            82675             12280 ns/op             392 B/op         26 allocs/op
	BenchmarkToSnakeIanColeman-4       83276             13903 ns/op             145 B/op         13 allocs/op
	// Regexp is slow
	BenchmarkToSnakeGolangPrograms-4   74448             18586 ns/op             176 B/op         11 allocs/op
	
	// These results aren't a surprise - my initial version of this library was
	// painfully slow. I think most of us, without spending some time with
	// profilers and benchmarks, would write also something on the slower side.

### Why not this package
If every nanosecond matters and this is used in a tight loop, use segment.io's
libraries (<a href="https://github.com/segmentio/go-snakecase">https://github.com/segmentio/go-snakecase</a> and
<a href="https://github.com/segmentio/go-camelcase">https://github.com/segmentio/go-camelcase</a>). They lack features, but make up for
it by being blazing fast. Alternatively, if you need your code to work slightly
differently, fork them and tailor it for your use case.

If you don't like having external imports, I get it. This package only imports
packages for testing, otherwise it only uses the standard library. If that's
not enough, you can use this repo as the foundation for your own. MIT Licensed.

This package is still relatively new and while I've used it for a while
personally, it doesn't have the miles that other packages do. I've tested this
code agains't their test cases to make sure that there aren't any surprises.

### Migrating from other packages
If you are migrating from from another package, you may find slight differences
in output. To reduce the delta, you may find it helpful to use the following
custom casers to mimic the behavior of the other package.


	// From <a href="https://github.com/iancoleman/strcase">https://github.com/iancoleman/strcase</a>
	var c = NewCaser(false, nil, NewSplitFn([]rune{'_', '-', '.'}, SplitCase, SplitAcronym, SplitBeforeNumber))
	
	// From <a href="https://github.com/stoewer/go-strcase">https://github.com/stoewer/go-strcase</a>
	var c = NewCaser(false, nil, NewSplitFn([]rune{'_', '-'}, SplitCase), SplitAcronym)




## <a name="pkg-index">Index</a>
* [func ToCamel(s string) string](#ToCamel)
* [func ToCase(s string, wordCase WordCase, delimiter rune) string](#ToCase)
* [func ToGoCamel(s string) string](#ToGoCamel)
* [func ToGoCase(s string, wordCase WordCase, delimiter rune) string](#ToGoCase)
* [func ToGoKebab(s string) string](#ToGoKebab)
* [func ToGoPascal(s string) string](#ToGoPascal)
* [func ToGoSnake(s string) string](#ToGoSnake)
* [func ToKEBAB(s string) string](#ToKEBAB)
* [func ToKebab(s string) string](#ToKebab)
* [func ToPascal(s string) string](#ToPascal)
* [func ToSNAKE(s string) string](#ToSNAKE)
* [func ToSnake(s string) string](#ToSnake)
* [type Caser](#Caser)
  * [func NewCaser(goInitialisms bool, initialismOverrides map[string]bool, splitFn SplitFn) *Caser](#NewCaser)
  * [func (c *Caser) ToCamel(s string) string](#Caser.ToCamel)
  * [func (c *Caser) ToCase(s string, wordCase WordCase, delimiter rune) string](#Caser.ToCase)
  * [func (c *Caser) ToKEBAB(s string) string](#Caser.ToKEBAB)
  * [func (c *Caser) ToKebab(s string) string](#Caser.ToKebab)
  * [func (c *Caser) ToPascal(s string) string](#Caser.ToPascal)
  * [func (c *Caser) ToSNAKE(s string) string](#Caser.ToSNAKE)
  * [func (c *Caser) ToSnake(s string) string](#Caser.ToSnake)
* [type SplitAction](#SplitAction)
* [type SplitFn](#SplitFn)
  * [func NewSplitFn(delimiters []rune, splitOptions ...SplitOption) SplitFn](#NewSplitFn)
* [type SplitOption](#SplitOption)
* [type WordCase](#WordCase)





## <a name="ToCamel">func</a> [ToCamel](./strcase.go#L57)
``` go
func ToCamel(s string) string
```
ToCamel returns words in camelCase (capitalized words concatenated together, with first word lower case).
Also known as lowerCamelCase or mixedCase.



## <a name="ToCase">func</a> [ToCase](./strcase.go#L70)
``` go
func ToCase(s string, wordCase WordCase, delimiter rune) string
```
ToCase returns words in given case and delimiter.



## <a name="ToGoCamel">func</a> [ToGoCamel](./strcase.go#L65)
``` go
func ToGoCamel(s string) string
```
ToGoCamel returns words in camelCase (capitalized words concatenated together, with first word lower case).
Also known as lowerCamelCase or mixedCase.

Respects Go's common initialisms (e.g. httpResponse -> HTTPResponse).



## <a name="ToGoCase">func</a> [ToGoCase](./strcase.go#L77)
``` go
func ToGoCase(s string, wordCase WordCase, delimiter rune) string
```
ToGoCase returns words in given case and delimiter.

Respects Go's common initialisms (e.g. httpResponse -> HTTPResponse).



## <a name="ToGoKebab">func</a> [ToGoKebab](./strcase.go#L31)
``` go
func ToGoKebab(s string) string
```
ToGoKebab returns words in kebab-case (lower case words with dashes).
Also known as dash-case.

Respects Go's common initialisms (e.g. http-response -> HTTP-response).



## <a name="ToGoPascal">func</a> [ToGoPascal](./strcase.go#L51)
``` go
func ToGoPascal(s string) string
```
ToGoPascal returns words in PascalCase (capitalized words concatenated together).
Also known as UpperPascalCase.

Respects Go's common initialisms (e.g. HttpResponse -> HTTPResponse).



## <a name="ToGoSnake">func</a> [ToGoSnake](./strcase.go#L11)
``` go
func ToGoSnake(s string) string
```
ToGoSnake returns words in snake_case (lower case words with underscores).

Respects Go's common initialisms (e.g. http_response -> HTTP_response).



## <a name="ToKEBAB">func</a> [ToKEBAB](./strcase.go#L37)
``` go
func ToKEBAB(s string) string
```
ToKEBAB returns words in KEBAB-CASE (upper case words with dashes).
Also known as SCREAMING-KEBAB-CASE or SCREAMING-DASH-CASE.



## <a name="ToKebab">func</a> [ToKebab](./strcase.go#L23)
``` go
func ToKebab(s string) string
```
ToKebab returns words in kebab-case (lower case words with dashes).
Also known as dash-case.



## <a name="ToPascal">func</a> [ToPascal](./strcase.go#L43)
``` go
func ToPascal(s string) string
```
ToPascal returns words in PascalCase (capitalized words concatenated together).
Also known as UpperPascalCase.



## <a name="ToSNAKE">func</a> [ToSNAKE](./strcase.go#L17)
``` go
func ToSNAKE(s string) string
```
ToSNAKE returns words in SNAKE_CASE (upper case words with underscores).
Also known as SCREAMING_SNAKE_CASE or UPPER_CASE.



## <a name="ToSnake">func</a> [ToSnake](./strcase.go#L4)
``` go
func ToSnake(s string) string
```
ToSnake returns words in snake_case (lower case words with underscores).




## <a name="Caser">type</a> [Caser](./caser.go#L4-L7)
``` go
type Caser struct {
    // contains filtered or unexported fields
}

```
Caser allows for customization of parsing and intialisms







### <a name="NewCaser">func</a> [NewCaser](./caser.go#L24)
``` go
func NewCaser(goInitialisms bool, initialismOverrides map[string]bool, splitFn SplitFn) *Caser
```
NewCaser returns a configured Caser.

A Caser should be created when you want fine grained control over how the words are split.


	Notes on function arguments
	
	goInitialisms: Whether to use Golint's intialisms
	
	initialismOverrides: A mapping of extra initialisms
	Keys must be in ALL CAPS. Merged with Golint's if goInitialisms is set.
	Setting a key to false will override Golint's.
	
	splitFn: How to separate words
	Override the default split function. Consider using NewSplitFn to
	configure one instead of writing your own.





### <a name="Caser.ToCamel">func</a> (\*Caser) [ToCamel](./caser.go#L80)
``` go
func (c *Caser) ToCamel(s string) string
```
ToCamel returns words in camelCase (capitalized words concatenated together, with first word lower case).
Also known as lowerCamelCase or mixedCase.




### <a name="Caser.ToCase">func</a> (\*Caser) [ToCase](./caser.go#L85)
``` go
func (c *Caser) ToCase(s string, wordCase WordCase, delimiter rune) string
```
ToCase returns words with a given case and delimiter.




### <a name="Caser.ToKEBAB">func</a> (\*Caser) [ToKEBAB](./caser.go#L68)
``` go
func (c *Caser) ToKEBAB(s string) string
```
ToKEBAB returns words in KEBAB-CASE (upper case words with dashes).
Also known as SCREAMING-KEBAB-CASE or SCREAMING-DASH-CASE.




### <a name="Caser.ToKebab">func</a> (\*Caser) [ToKebab](./caser.go#L62)
``` go
func (c *Caser) ToKebab(s string) string
```
ToKebab returns words in kebab-case (lower case words with dashes).
Also known as dash-case.




### <a name="Caser.ToPascal">func</a> (\*Caser) [ToPascal](./caser.go#L74)
``` go
func (c *Caser) ToPascal(s string) string
```
ToPascal returns words in PascalCase (capitalized words concatenated together).
Also known as UpperPascalCase.




### <a name="Caser.ToSNAKE">func</a> (\*Caser) [ToSNAKE](./caser.go#L56)
``` go
func (c *Caser) ToSNAKE(s string) string
```
ToSNAKE returns words in SNAKE_CASE (upper case words with underscores).
Also known as SCREAMING_SNAKE_CASE or UPPER_CASE.




### <a name="Caser.ToSnake">func</a> (\*Caser) [ToSnake](./caser.go#L50)
``` go
func (c *Caser) ToSnake(s string) string
```
ToSnake returns words in snake_case (lower case words with underscores).




## <a name="SplitAction">type</a> [SplitAction](./split.go#L110)
``` go
type SplitAction int
```
SplitAction defines if and how to split a string


``` go
const (
    // Noop - Continue to next character
    Noop SplitAction = iota
    // Split - Split between words
    // e.g. to split between wordsWithoutDelimiters
    Split
    // SkipSplit - Split the word and drop the character
    // e.g. to split words with delimiters
    SkipSplit
    // Skip - Remove the character completely
    Skip
)
```









## <a name="SplitFn">type</a> [SplitFn](./split.go#L6)
``` go
type SplitFn func(prev, curr, next rune) SplitAction
```
SplitFn defines how to split a string into words







### <a name="NewSplitFn">func</a> [NewSplitFn](./split.go#L14-L17)
``` go
func NewSplitFn(
    delimiters []rune,
    splitOptions ...SplitOption,
) SplitFn
```
NewSplitFn returns a SplitFn based on the options provided.

NewSplitFn covers the majority of common options that other strcase
libraries provide and should allow you to simply create a custom caser.
For more complicated use cases, feel free to write your own SplitFn
nolint:gocyclo





## <a name="SplitOption">type</a> [SplitOption](./split.go#L93)
``` go
type SplitOption int
```
SplitOption are options that allow for configuring NewSplitFn


``` go
const (
    // SplitCase - FooBar -> Foo_Bar
    SplitCase SplitOption = iota
    // SplitAcronym - FOOBar -> Foo_Bar
    // It won't preserve FOO's case. If you want, you can set the Caser's initialisms so FOO will be in all caps
    SplitAcronym
    // SplitBeforeNumber - port80 -> port_80
    SplitBeforeNumber
    // SplitAfterNumber - 200status -> 200_status
    SplitAfterNumber
    // PreserveNumberFormatting - a.b.2,000.3.c -> a_b_2,000.3_c
    PreserveNumberFormatting
)
```









## <a name="WordCase">type</a> [WordCase](./convert.go#L6)
``` go
type WordCase int
```
WordCase is an enumeration of the ways to format a word.


``` go
const (
    // Original - Preserve the original input strcase
    Original WordCase = iota
    // LowerCase - All letters lower cased (example)
    LowerCase
    // UpperCase - All letters upper cased (EXAMPLE)
    UpperCase
    // TitleCase - Only first letter upper cased (Example)
    TitleCase
    // CamelCase - TitleCase except lower case first word (exampleText)
    CamelCase
)
```













