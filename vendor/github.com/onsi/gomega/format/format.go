/*
Gomega's format package pretty-prints objects.  It explores input objects recursively and generates formatted, indented output with type information.
*/

// untested sections: 4

package format

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// Use MaxDepth to set the maximum recursion depth when printing deeply nested objects
var MaxDepth = uint(10)

/*
By default, all objects (even those that implement fmt.Stringer and fmt.GoStringer) are recursively inspected to generate output.

Set UseStringerRepresentation = true to use GoString (for fmt.GoStringers) or String (for fmt.Stringer) instead.

Note that GoString and String don't always have all the information you need to understand why a test failed!
*/
var UseStringerRepresentation = false

/*
Print the content of context objects. By default it will be suppressed.

Set PrintContextObjects = true to enable printing of the context internals.
*/
var PrintContextObjects = false

// TruncatedDiff choose if we should display a truncated pretty diff or not
var TruncatedDiff = true

// TruncateThreshold (default 50) specifies the maximum length string to print in string comparison assertion error
// messages.
var TruncateThreshold uint = 50

// CharactersAroundMismatchToInclude (default 5) specifies how many contextual characters should be printed before and
// after the first diff location in a truncated string assertion error message.
var CharactersAroundMismatchToInclude uint = 5

// Ctx interface defined here to keep backwards compatibility with go < 1.7
// It matches the context.Context interface
type Ctx interface {
	Deadline() (deadline time.Time, ok bool)
	Done() <-chan struct{}
	Err() error
	Value(key interface{}) interface{}
}

var contextType = reflect.TypeOf((*Ctx)(nil)).Elem()
var timeType = reflect.TypeOf(time.Time{})

//The default indentation string emitted by the format package
var Indent = "    "

var longFormThreshold = 20

/*
Generates a formatted matcher success/failure message of the form:

	Expected
		<pretty printed actual>
	<message>
		<pretty printed expected>

If expected is omitted, then the message looks like:

	Expected
		<pretty printed actual>
	<message>
*/
func Message(actual interface{}, message string, expected ...interface{}) string {
	if len(expected) == 0 {
		return fmt.Sprintf("Expected\n%s\n%s", Object(actual, 1), message)
	}
	return fmt.Sprintf("Expected\n%s\n%s\n%s", Object(actual, 1), message, Object(expected[0], 1))
}

/*

Generates a nicely formatted matcher success / failure message

Much like Message(...), but it attempts to pretty print diffs in strings

Expected
    <string>: "...aaaaabaaaaa..."
to equal               |
    <string>: "...aaaaazaaaaa..."

*/

func MessageWithDiff(actual, message, expected string) string {
	if TruncatedDiff && len(actual) >= int(TruncateThreshold) && len(expected) >= int(TruncateThreshold) {
		diffPoint := findFirstMismatch(actual, expected)
		formattedActual := truncateAndFormat(actual, diffPoint)
		formattedExpected := truncateAndFormat(expected, diffPoint)

		spacesBeforeFormattedMismatch := findFirstMismatch(formattedActual, formattedExpected)

		tabLength := 4
		spaceFromMessageToActual := tabLength + len("<string>: ") - len(message)
		padding := strings.Repeat(" ", spaceFromMessageToActual+spacesBeforeFormattedMismatch) + "|"
		return Message(formattedActual, message+padding, formattedExpected)
	}

	actual = escapedWithGoSyntax(actual)
	expected = escapedWithGoSyntax(expected)

	return Message(actual, message, expected)
}

func escapedWithGoSyntax(str string) string {
	withQuotes := fmt.Sprintf("%q", str)
	return withQuotes[1 : len(withQuotes)-1]
}

func truncateAndFormat(str string, index int) string {
	leftPadding := `...`
	rightPadding := `...`

	start := index - int(CharactersAroundMismatchToInclude)
	if start < 0 {
		start = 0
		leftPadding = ""
	}

	// slice index must include the mis-matched character
	lengthOfMismatchedCharacter := 1
	end := index + int(CharactersAroundMismatchToInclude) + lengthOfMismatchedCharacter
	if end > len(str) {
		end = len(str)
		rightPadding = ""

	}
	return fmt.Sprintf("\"%s\"", leftPadding+str[start:end]+rightPadding)
}

func findFirstMismatch(a, b string) int {
	aSlice := strings.Split(a, "")
	bSlice := strings.Split(b, "")

	for index, str := range aSlice {
		if index > len(bSlice)-1 {
			return index
		}
		if str != bSlice[index] {
			return index
		}
	}

	if len(b) > len(a) {
		return len(a) + 1
	}

	return 0
}

/*
Pretty prints the passed in object at the passed in indentation level.

Object recurses into deeply nested objects emitting pretty-printed representations of their components.

Modify format.MaxDepth to control how deep the recursion is allowed to go
Set format.UseStringerRepresentation to true to return object.GoString() or object.String() when available instead of
recursing into the object.

Set PrintContextObjects to true to print the content of objects implementing context.Context
*/
func Object(object interface{}, indentation uint) string {
	indent := strings.Repeat(Indent, int(indentation))
	value := reflect.ValueOf(object)
	return fmt.Sprintf("%s<%s>: %s", indent, formatType(object), formatValue(value, indentation))
}

/*
IndentString takes a string and indents each line by the specified amount.
*/
func IndentString(s string, indentation uint) string {
	components := strings.Split(s, "\n")
	result := ""
	indent := strings.Repeat(Indent, int(indentation))
	for i, component := range components {
		result += indent + component
		if i < len(components)-1 {
			result += "\n"
		}
	}

	return result
}

func formatType(object interface{}) string {
	t := reflect.TypeOf(object)
	if t == nil {
		return "nil"
	}
	switch t.Kind() {
	case reflect.Chan:
		v := reflect.ValueOf(object)
		return fmt.Sprintf("%T | len:%d, cap:%d", object, v.Len(), v.Cap())
	case reflect.Ptr:
		return fmt.Sprintf("%T | %p", object, object)
	case reflect.Slice:
		v := reflect.ValueOf(object)
		return fmt.Sprintf("%T | len:%d, cap:%d", object, v.Len(), v.Cap())
	case reflect.Map:
		v := reflect.ValueOf(object)
		return fmt.Sprintf("%T | len:%d", object, v.Len())
	default:
		return fmt.Sprintf("%T", object)
	}
}

func formatValue(value reflect.Value, indentation uint) string {
	if indentation > MaxDepth {
		return "..."
	}

	if isNilValue(value) {
		return "nil"
	}

	if UseStringerRepresentation {
		if value.CanInterface() {
			obj := value.Interface()
			switch x := obj.(type) {
			case fmt.GoStringer:
				return x.GoString()
			case fmt.Stringer:
				return x.String()
			}
		}
	}

	if !PrintContextObjects {
		if value.Type().Implements(contextType) && indentation > 1 {
			return "<suppressed context>"
		}
	}

	switch value.Kind() {
	case reflect.Bool:
		return fmt.Sprintf("%v", value.Bool())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return fmt.Sprintf("%v", value.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return fmt.Sprintf("%v", value.Uint())
	case reflect.Uintptr:
		return fmt.Sprintf("0x%x", value.Uint())
	case reflect.Float32, reflect.Float64:
		return fmt.Sprintf("%v", value.Float())
	case reflect.Complex64, reflect.Complex128:
		return fmt.Sprintf("%v", value.Complex())
	case reflect.Chan:
		return fmt.Sprintf("0x%x", value.Pointer())
	case reflect.Func:
		return fmt.Sprintf("0x%x", value.Pointer())
	case reflect.Ptr:
		return formatValue(value.Elem(), indentation)
	case reflect.Slice:
		return formatSlice(value, indentation)
	case reflect.String:
		return formatString(value.String(), indentation)
	case reflect.Array:
		return formatSlice(value, indentation)
	case reflect.Map:
		return formatMap(value, indentation)
	case reflect.Struct:
		if value.Type() == timeType && value.CanInterface() {
			t, _ := value.Interface().(time.Time)
			return t.Format(time.RFC3339Nano)
		}
		return formatStruct(value, indentation)
	case reflect.Interface:
		return formatValue(value.Elem(), indentation)
	default:
		if value.CanInterface() {
			return fmt.Sprintf("%#v", value.Interface())
		}
		return fmt.Sprintf("%#v", value)
	}
}

func formatString(object interface{}, indentation uint) string {
	if indentation == 1 {
		s := fmt.Sprintf("%s", object)
		components := strings.Split(s, "\n")
		result := ""
		for i, component := range components {
			if i == 0 {
				result += component
			} else {
				result += Indent + component
			}
			if i < len(components)-1 {
				result += "\n"
			}
		}

		return result
	} else {
		return fmt.Sprintf("%q", object)
	}
}

func formatSlice(v reflect.Value, indentation uint) string {
	if v.Kind() == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8 && isPrintableString(string(v.Bytes())) {
		return formatString(v.Bytes(), indentation)
	}

	l := v.Len()
	result := make([]string, l)
	longest := 0
	for i := 0; i < l; i++ {
		result[i] = formatValue(v.Index(i), indentation+1)
		if len(result[i]) > longest {
			longest = len(result[i])
		}
	}

	if longest > longFormThreshold {
		indenter := strings.Repeat(Indent, int(indentation))
		return fmt.Sprintf("[\n%s%s,\n%s]", indenter+Indent, strings.Join(result, ",\n"+indenter+Indent), indenter)
	}
	return fmt.Sprintf("[%s]", strings.Join(result, ", "))
}

func formatMap(v reflect.Value, indentation uint) string {
	l := v.Len()
	result := make([]string, l)

	longest := 0
	for i, key := range v.MapKeys() {
		value := v.MapIndex(key)
		result[i] = fmt.Sprintf("%s: %s", formatValue(key, indentation+1), formatValue(value, indentation+1))
		if len(result[i]) > longest {
			longest = len(result[i])
		}
	}

	if longest > longFormThreshold {
		indenter := strings.Repeat(Indent, int(indentation))
		return fmt.Sprintf("{\n%s%s,\n%s}", indenter+Indent, strings.Join(result, ",\n"+indenter+Indent), indenter)
	}
	return fmt.Sprintf("{%s}", strings.Join(result, ", "))
}

func formatStruct(v reflect.Value, indentation uint) string {
	t := v.Type()

	l := v.NumField()
	result := []string{}
	longest := 0
	for i := 0; i < l; i++ {
		structField := t.Field(i)
		fieldEntry := v.Field(i)
		representation := fmt.Sprintf("%s: %s", structField.Name, formatValue(fieldEntry, indentation+1))
		result = append(result, representation)
		if len(representation) > longest {
			longest = len(representation)
		}
	}
	if longest > longFormThreshold {
		indenter := strings.Repeat(Indent, int(indentation))
		return fmt.Sprintf("{\n%s%s,\n%s}", indenter+Indent, strings.Join(result, ",\n"+indenter+Indent), indenter)
	}
	return fmt.Sprintf("{%s}", strings.Join(result, ", "))
}

func isNilValue(a reflect.Value) bool {
	switch a.Kind() {
	case reflect.Invalid:
		return true
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return a.IsNil()
	}

	return false
}

/*
Returns true when the string is entirely made of printable runes, false otherwise.
*/
func isPrintableString(str string) bool {
	for _, runeValue := range str {
		if !strconv.IsPrint(runeValue) {
			return false
		}
	}
	return true
}
