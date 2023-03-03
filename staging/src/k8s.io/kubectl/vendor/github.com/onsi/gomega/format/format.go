/*
Gomega's format package pretty-prints objects.  It explores input objects recursively and generates formatted, indented output with type information.
*/

// untested sections: 4

package format

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// Use MaxDepth to set the maximum recursion depth when printing deeply nested objects
var MaxDepth = uint(10)

// MaxLength of the string representation of an object.
// If MaxLength is set to 0, the Object will not be truncated.
var MaxLength = 4000

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

var contextType = reflect.TypeOf((*context.Context)(nil)).Elem()
var timeType = reflect.TypeOf(time.Time{})

//The default indentation string emitted by the format package
var Indent = "    "

var longFormThreshold = 20

// GomegaStringer allows for custom formating of objects for gomega.
type GomegaStringer interface {
	// GomegaString will be used to custom format an object.
	// It does not follow UseStringerRepresentation value and will always be called regardless.
	// It also ignores the MaxLength value.
	GomegaString() string
}

/*
CustomFormatters can be registered with Gomega via RegisterCustomFormatter()
Any value to be rendered by Gomega is passed to each registered CustomFormatters.
The CustomFormatter signals that it will handle formatting the value by returning (formatted-string, true)
If the CustomFormatter does not want to handle the object it should return ("", false)

Strings returned by CustomFormatters are not truncated
*/
type CustomFormatter func(value interface{}) (string, bool)
type CustomFormatterKey uint

var customFormatterKey CustomFormatterKey = 1

type customFormatterKeyPair struct {
	CustomFormatter
	CustomFormatterKey
}

/*
RegisterCustomFormatter registers a CustomFormatter and returns a CustomFormatterKey

You can call UnregisterCustomFormatter with the returned key to unregister the associated CustomFormatter
*/
func RegisterCustomFormatter(customFormatter CustomFormatter) CustomFormatterKey {
	key := customFormatterKey
	customFormatterKey += 1
	customFormatters = append(customFormatters, customFormatterKeyPair{customFormatter, key})
	return key
}

/*
UnregisterCustomFormatter unregisters a previously registered CustomFormatter.  You should pass in the key returned by RegisterCustomFormatter
*/
func UnregisterCustomFormatter(key CustomFormatterKey) {
	formatters := []customFormatterKeyPair{}
	for _, f := range customFormatters {
		if f.CustomFormatterKey == key {
			continue
		}
		formatters = append(formatters, f)
	}
	customFormatters = formatters
}

var customFormatters = []customFormatterKeyPair{}

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

		paddingCount := spaceFromMessageToActual + spacesBeforeFormattedMismatch
		if paddingCount < 0 {
			return Message(formattedActual, message, formattedExpected)
		}

		padding := strings.Repeat(" ", paddingCount) + "|"
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

const truncateHelpText = `
Gomega truncated this representation as it exceeds 'format.MaxLength'.
Consider having the object provide a custom 'GomegaStringer' representation
or adjust the parameters in Gomega's 'format' package.

Learn more here: https://onsi.github.io/gomega/#adjusting-output
`

func truncateLongStrings(s string) string {
	if MaxLength > 0 && len(s) > MaxLength {
		var sb strings.Builder
		for i, r := range s {
			if i < MaxLength {
				sb.WriteRune(r)
				continue
			}
			break
		}

		sb.WriteString("...\n")
		sb.WriteString(truncateHelpText)

		return sb.String()
	}
	return s
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
	return fmt.Sprintf("%s<%s>: %s", indent, formatType(value), formatValue(value, indentation))
}

/*
IndentString takes a string and indents each line by the specified amount.
*/
func IndentString(s string, indentation uint) string {
	return indentString(s, indentation, true)
}

func indentString(s string, indentation uint, indentFirstLine bool) string {
	result := &strings.Builder{}
	components := strings.Split(s, "\n")
	indent := strings.Repeat(Indent, int(indentation))
	for i, component := range components {
		if i > 0 || indentFirstLine {
			result.WriteString(indent)
		}
		result.WriteString(component)
		if i < len(components)-1 {
			result.WriteString("\n")
		}
	}

	return result.String()
}

func formatType(v reflect.Value) string {
	switch v.Kind() {
	case reflect.Invalid:
		return "nil"
	case reflect.Chan:
		return fmt.Sprintf("%s | len:%d, cap:%d", v.Type(), v.Len(), v.Cap())
	case reflect.Ptr:
		return fmt.Sprintf("%s | 0x%x", v.Type(), v.Pointer())
	case reflect.Slice:
		return fmt.Sprintf("%s | len:%d, cap:%d", v.Type(), v.Len(), v.Cap())
	case reflect.Map:
		return fmt.Sprintf("%s | len:%d", v.Type(), v.Len())
	default:
		return fmt.Sprintf("%s", v.Type())
	}
}

func formatValue(value reflect.Value, indentation uint) string {
	if indentation > MaxDepth {
		return "..."
	}

	if isNilValue(value) {
		return "nil"
	}

	if value.CanInterface() {
		obj := value.Interface()

		// if a CustomFormatter handles this values, we'll go with that
		for _, customFormatter := range customFormatters {
			formatted, handled := customFormatter.CustomFormatter(obj)
			// do not truncate a user-provided CustomFormatter()
			if handled {
				return indentString(formatted, indentation+1, false)
			}
		}

		// GomegaStringer will take precedence to other representations and disregards UseStringerRepresentation
		if x, ok := obj.(GomegaStringer); ok {
			// do not truncate a user-defined GomegaString() value
			return indentString(x.GomegaString(), indentation+1, false)
		}

		if UseStringerRepresentation {
			switch x := obj.(type) {
			case fmt.GoStringer:
				return indentString(truncateLongStrings(x.GoString()), indentation+1, false)
			case fmt.Stringer:
				return indentString(truncateLongStrings(x.String()), indentation+1, false)
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
		return truncateLongStrings(formatSlice(value, indentation))
	case reflect.String:
		return truncateLongStrings(formatString(value.String(), indentation))
	case reflect.Array:
		return truncateLongStrings(formatSlice(value, indentation))
	case reflect.Map:
		return truncateLongStrings(formatMap(value, indentation))
	case reflect.Struct:
		if value.Type() == timeType && value.CanInterface() {
			t, _ := value.Interface().(time.Time)
			return t.Format(time.RFC3339Nano)
		}
		return truncateLongStrings(formatStruct(value, indentation))
	case reflect.Interface:
		return formatInterface(value, indentation)
	default:
		if value.CanInterface() {
			return truncateLongStrings(fmt.Sprintf("%#v", value.Interface()))
		}
		return truncateLongStrings(fmt.Sprintf("%#v", value))
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

func formatInterface(v reflect.Value, indentation uint) string {
	return fmt.Sprintf("<%s>%s", formatType(v.Elem()), formatValue(v.Elem(), indentation))
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
