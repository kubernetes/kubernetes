package simple

import "honnef.co/go/tools/analysis/lint"

var Docs = lint.Markdownify(map[string]*lint.Documentation{
	"S1000": {
		Title: `Use plain channel send or receive instead of single-case select`,
		Text: `Select statements with a single case can be replaced with a simple
send or receive.

Before:

    select {
    case x := <-ch:
        fmt.Println(x)
    }

After:

    x := <-ch
    fmt.Println(x)`,
		Since: "2017.1",
	},

	"S1001": {
		Title: `Replace for loop with call to copy`,
		Text: `Use \'copy()\' for copying elements from one slice to another.

Before:

    for i, x := range src {
        dst[i] = x
    }

After:

    copy(dst, src)`,
		Since: "2017.1",
	},

	"S1002": {
		Title: `Omit comparison with boolean constant`,
		Text: `Before:

    if x == true {}

After:

    if x {}`,
		Since: "2017.1",
	},

	"S1003": {
		Title: "Replace call to `strings.Index` with `strings.Contains`",
		Text: `Before:

    if strings.Index(x, y) != -1 {}

After:

    if strings.Contains(x, y) {}`,
		Since: "2017.1",
	},

	"S1004": {
		Title: "Replace call to `bytes.Compare` with `bytes.Equal`",
		Text: `Before:

    if bytes.Compare(x, y) == 0 {}

After:

    if bytes.Equal(x, y) {}`,
		Since: "2017.1",
	},

	"S1005": {
		Title: `Drop unnecessary use of the blank identifier`,
		Text: `In many cases, assigning to the blank identifier is unnecessary.

Before:

    for _ = range s {}
    x, _ = someMap[key]
    _ = <-ch

After:

    for range s{}
    x = someMap[key]
    <-ch`,
		Since: "2017.1",
	},

	"S1006": {
		Title: "Use `for { ... }` for infinite loops",
		Text:  `For infinite loops, using \'for { ... }\' is the most idiomatic choice.`,
		Since: "2017.1",
	},

	"S1007": {
		Title: `Simplify regular expression by using raw string literal`,
		Text: `Raw string literals use \' instead of " and do not support
any escape sequences. This means that the backslash (\'\\') can be used
freely, without the need of escaping.

Since regular expressions have their own escape sequences, raw strings
can improve their readability.

Before:

    regexp.Compile("\\A(\\w+) profile: total \\d+\\n\\z")

After:

    regexp.Compile(` + "`" + `\A(\w+) profile: total \d+\n\z` + "`" + `)`,
		Since: "2017.1",
	},

	"S1008": {
		Title: `Simplify returning boolean expression`,
		Text: `Before:

    if <expr> {
        return true
    }
    return false

After:

    return <expr>`,
		Since: "2017.1",
	},

	"S1009": {
		Title: `Omit redundant nil check on slices`,
		Text: `The \'len\' function is defined for all slices, even nil ones, which have
a length of zero. It is not necessary to check if a slice is not nil
before checking that its length is not zero.

Before:

    if x != nil && len(x) != 0 {}

After:

    if len(x) != 0 {}`,
		Since: "2017.1",
	},

	"S1010": {
		Title: `Omit default slice index`,
		Text: `When slicing, the second index defaults to the length of the value,
making \'s[n:len(s)]\' and \'s[n:]\' equivalent.`,
		Since: "2017.1",
	},

	"S1011": {
		Title: "Use a single `append` to concatenate two slices",
		Text: `Before:

    for _, e := range y {
        x = append(x, e)
    }

After:

    x = append(x, y...)`,
		Since: "2017.1",
	},

	"S1012": {
		Title: "Replace `time.Now().Sub(x)` with `time.Since(x)`",
		Text: `The \'time.Since\' helper has the same effect as using \'time.Now().Sub(x)\'
but is easier to read.

Before:

    time.Now().Sub(x)

After:

    time.Since(x)`,
		Since: "2017.1",
	},

	"S1016": {
		Title: `Use a type conversion instead of manually copying struct fields`,
		Text: `Two struct types with identical fields can be converted between each
other. In older versions of Go, the fields had to have identical
struct tags. Since Go 1.8, however, struct tags are ignored during
conversions. It is thus not necessary to manually copy every field
individually.

Before:

    var x T1
    y := T2{
        Field1: x.Field1,
        Field2: x.Field2,
    }

After:

    var x T1
    y := T2(x)`,
		Since: "2017.1",
	},

	"S1017": {
		Title: "Replace manual trimming with `strings.TrimPrefix`",
		Text: `Instead of using \'strings.HasPrefix\' and manual slicing, use the
\'strings.TrimPrefix\' function. If the string doesn't start with the
prefix, the original string will be returned. Using \'strings.TrimPrefix\'
reduces complexity, and avoids common bugs, such as off-by-one
mistakes.

Before:

    if strings.HasPrefix(str, prefix) {
        str = str[len(prefix):]
    }

After:

    str = strings.TrimPrefix(str, prefix)`,
		Since: "2017.1",
	},

	"S1018": {
		Title: "Use `copy` for sliding elements",
		Text: `\'copy()\' permits using the same source and destination slice, even with
overlapping ranges. This makes it ideal for sliding elements in a
slice.

Before:

    for i := 0; i < n; i++ {
        bs[i] = bs[offset+i]
    }

After:

    copy(bs[:n], bs[offset:])`,
		Since: "2017.1",
	},

	"S1019": {
		Title: "Simplify `make` call by omitting redundant arguments",
		Text: `The \'make\' function has default values for the length and capacity
arguments. For channels, the length defaults to zero, and for slices, the capacity defaults to the length.`,
		Since: "2017.1",
	},

	"S1020": {
		Title: `Omit redundant nil check in type assertion`,
		Text: `Before:

    if _, ok := i.(T); ok && i != nil {}

After:

    if _, ok := i.(T); ok {}`,
		Since: "2017.1",
	},

	"S1021": {
		Title: `Merge variable declaration and assignment`,
		Text: `Before:

    var x uint
    x = 1

After:

    var x uint = 1`,
		Since: "2017.1",
	},

	"S1023": {
		Title: `Omit redundant control flow`,
		Text: `Functions that have no return value do not need a return statement as
the final statement of the function.

Switches in Go do not have automatic fallthrough, unlike languages
like C. It is not necessary to have a break statement as the final
statement in a case block.`,
		Since: "2017.1",
	},

	"S1024": {
		Title: "Replace `x.Sub(time.Now())` with `time.Until(x)`",
		Text: `The \'time.Until\' helper has the same effect as using \'x.Sub(time.Now())\'
but is easier to read.

Before:

    x.Sub(time.Now())

After:

    time.Until(x)`,
		Since: "2017.1",
	},

	"S1025": {
		Title: "Don't use `fmt.Sprintf(\"%s\", x)` unnecessarily",
		Text: `In many instances, there are easier and more efficient ways of getting
a value's string representation. Whenever a value's underlying type is
a string already, or the type has a String method, they should be used
directly.

Given the following shared definitions

    type T1 string
    type T2 int

    func (T2) String() string { return "Hello, world" }

    var x string
    var y T1
    var z T2

we can simplify the following

    fmt.Sprintf("%s", x)
    fmt.Sprintf("%s", y)
    fmt.Sprintf("%s", z)

to

    x
    string(y)
    z.String()`,
		Since: "2017.1",
	},

	"S1028": {
		Title: "Simplify error construction with `fmt.Errorf`",
		Text: `Before:

    errors.New(fmt.Sprintf(...))

After:

    fmt.Errorf(...)`,
		Since: "2017.1",
	},

	"S1029": {
		Title: `Range over the string directly`,
		Text: `Ranging over a string will yield byte offsets and runes. If the offset
isn't used, this is functionally equivalent to converting the string
to a slice of runes and ranging over that. Ranging directly over the
string will be more performant, however, as it avoids allocating a new
slice, the size of which depends on the length of the string.

Before:

    for _, r := range []rune(s) {}

After:

    for _, r := range s {}`,
		Since: "2017.1",
	},

	"S1030": {
		Title: "Use `bytes.Buffer.String` or `bytes.Buffer.Bytes`",
		Text: `\'bytes.Buffer\' has both a \'String\' and a \'Bytes\' method. It is almost never
necessary to use \'string(buf.Bytes())\' or \'[]byte(buf.String())\' – simply
use the other method.

The only exception to this are map lookups. Due to a compiler optimization,
\'m[string(buf.Bytes())]\' is more efficient than \'m[buf.String()]\'.
`,
		Since: "2017.1",
	},

	"S1031": {
		Title: `Omit redundant nil check around loop`,
		Text: `You can use range on nil slices and maps, the loop will simply never
execute. This makes an additional nil check around the loop
unnecessary.

Before:

    if s != nil {
        for _, x := range s {
            ...
        }
    }

After:

    for _, x := range s {
        ...
    }`,
		Since: "2017.1",
	},

	"S1032": {
		Title: "Use `sort.Ints(x)`, `sort.Float64s(x)`, and `sort.Strings(x)`",
		Text: `The \'sort.Ints\', \'sort.Float64s\' and \'sort.Strings\' functions are easier to
read than \'sort.Sort(sort.IntSlice(x))\', \'sort.Sort(sort.Float64Slice(x))\'
and \'sort.Sort(sort.StringSlice(x))\'.

Before:

    sort.Sort(sort.StringSlice(x))

After:

    sort.Strings(x)`,
		Since: "2019.1",
	},

	"S1033": {
		Title: "Unnecessary guard around call to `delete`",
		Text:  `Calling \'delete\' on a nil map is a no-op.`,
		Since: "2019.2",
	},

	"S1034": {
		Title: `Use result of type assertion to simplify cases`,
		Since: "2019.2",
	},

	"S1035": {
		Title: "Redundant call to `net/http.CanonicalHeaderKey` in method call on `net/http.Header`",
		Text: `The methods on \'net/http.Header\', namely \'Add\', \'Del\', \'Get\' and \'Set\', already
canonicalize the given header name.`,
		Since: "2020.1",
	},

	"S1036": {
		Title: `Unnecessary guard around map access`,

		Text: `When accessing a map key that doesn't exist yet, one
receives a zero value. Often, the zero value is a suitable value, for example when using append or doing integer math.

The following

    if _, ok := m["foo"]; ok {
        m["foo"] = append(m["foo"], "bar")
    } else {
        m["foo"] = []string{"bar"}
    }

can be simplified to

    m["foo"] = append(m["foo"], "bar")

and

    if _, ok := m2["k"]; ok {
        m2["k"] += 4
    } else {
        m2["k"] = 4
    }

can be simplified to

    m["k"] += 4
`,
		Since: "2020.1",
	},

	"S1037": {
		Title: `Elaborate way of sleeping`,
		Text: `Using a select statement with a single case receiving
from the result of \'time.After\' is a very elaborate way of sleeping that
can much simpler be expressed with a simple call to time.Sleep.`,
		Since: "2020.1",
	},

	"S1038": {
		Title: "Unnecessarily complex way of printing formatted string",
		Text:  `Instead of using \'fmt.Print(fmt.Sprintf(...))\', one can use \'fmt.Printf(...)\'.`,
		Since: "2020.1",
	},

	"S1039": {
		Title: "Unnecessary use of `fmt.Sprint`",
		Text:  `Calling \'fmt.Sprint\' with a single string argument is unnecessary and identical to using the string directly.`,
		Since: "2020.1",
	},
	"S1040": {
		Title: "Type assertion to current type",
		Text: `The type assertion \'x.(SomeInterface)\', when \'x\' already has type
\'SomeInterface\', can only fail if \'x\' is nil. Usually, this is
left-over code from when \'x\' had a different type and you can safely
delete the type assertion. If you want to check that \'x\' is not nil,
consider being explicit and using an actual \'if x == nil\' comparison
instead of relying on the type assertion panicking.`,
		Since: "2021.1",
	},
})
