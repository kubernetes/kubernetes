package ginkgo

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/onsi/ginkgo/v2/internal"
	"github.com/onsi/ginkgo/v2/types"
)

/*
The EntryDescription decorator allows you to pass a format string to DescribeTable() and Entry().  This format string is used to generate entry names via:

    fmt.Sprintf(formatString, parameters...)

where parameters are the parameters passed into the entry.

When passed into an Entry the EntryDescription is used to generate the name or that entry.  When passed to DescribeTable, the EntryDescription is used to generate teh names for any entries that have `nil` descriptions.

You can learn more about generating EntryDescriptions here: https://onsi.github.io/ginkgo/#generating-entry-descriptions
*/
type EntryDescription string

func (ed EntryDescription) render(args ...interface{}) string {
	return fmt.Sprintf(string(ed), args...)
}

/*
DescribeTable describes a table-driven spec.

For example:

    DescribeTable("a simple table",
        func(x int, y int, expected bool) {
            Ω(x > y).Should(Equal(expected))
        },
        Entry("x > y", 1, 0, true),
        Entry("x == y", 0, 0, false),
        Entry("x < y", 0, 1, false),
    )

You can learn more about DescribeTable here: https://onsi.github.io/ginkgo/#table-specs
And can explore some Table patterns here: https://onsi.github.io/ginkgo/#table-specs-patterns
*/
func DescribeTable(description string, args ...interface{}) bool {
	generateTable(description, args...)
	return true
}

/*
You can focus a table with `FDescribeTable`.  This is equivalent to `FDescribe`.
*/
func FDescribeTable(description string, args ...interface{}) bool {
	args = append(args, internal.Focus)
	generateTable(description, args...)
	return true
}

/*
You can mark a table as pending with `PDescribeTable`.  This is equivalent to `PDescribe`.
*/
func PDescribeTable(description string, args ...interface{}) bool {
	args = append(args, internal.Pending)
	generateTable(description, args...)
	return true
}

/*
You can mark a table as pending with `XDescribeTable`.  This is equivalent to `XDescribe`.
*/
var XDescribeTable = PDescribeTable

/*
TableEntry represents an entry in a table test.  You generally use the `Entry` constructor.
*/
type TableEntry struct {
	description  interface{}
	decorations  []interface{}
	parameters   []interface{}
	codeLocation types.CodeLocation
}

/*
Entry constructs a TableEntry.

The first argument is a description.  This can be a string, a function that accepts the parameters passed to the TableEntry and returns a string, an EntryDescription format string, or nil.  If nil is provided then the name of the Entry is derived using the table-level entry description.
Subsequent arguments accept any Ginkgo decorators.  These are filtered out and the remaining arguments are passed into the Spec function associated with the table.

Each Entry ends up generating an individual Ginkgo It.  The body of the it is the Table Body function with the Entry parameters passed in.

You can learn more about Entry here: https://onsi.github.io/ginkgo/#table-specs
*/
func Entry(description interface{}, args ...interface{}) TableEntry {
	decorations, parameters := internal.PartitionDecorations(args...)
	return TableEntry{description: description, decorations: decorations, parameters: parameters, codeLocation: types.NewCodeLocation(1)}
}

/*
You can focus a particular entry with FEntry.  This is equivalent to FIt.
*/
func FEntry(description interface{}, args ...interface{}) TableEntry {
	decorations, parameters := internal.PartitionDecorations(args...)
	decorations = append(decorations, internal.Focus)
	return TableEntry{description: description, decorations: decorations, parameters: parameters, codeLocation: types.NewCodeLocation(1)}
}

/*
You can mark a particular entry as pending with PEntry.  This is equivalent to PIt.
*/
func PEntry(description interface{}, args ...interface{}) TableEntry {
	decorations, parameters := internal.PartitionDecorations(args...)
	decorations = append(decorations, internal.Pending)
	return TableEntry{description: description, decorations: decorations, parameters: parameters, codeLocation: types.NewCodeLocation(1)}
}

/*
You can mark a particular entry as pending with XEntry.  This is equivalent to XIt.
*/
var XEntry = PEntry

func generateTable(description string, args ...interface{}) {
	cl := types.NewCodeLocation(2)
	containerNodeArgs := []interface{}{cl}

	entries := []TableEntry{}
	var itBody interface{}

	var tableLevelEntryDescription interface{}
	tableLevelEntryDescription = func(args ...interface{}) string {
		out := []string{}
		for _, arg := range args {
			out = append(out, fmt.Sprint(arg))
		}
		return "Entry: " + strings.Join(out, ", ")
	}

	for _, arg := range args {
		switch t := reflect.TypeOf(arg); {
		case t == reflect.TypeOf(TableEntry{}):
			entries = append(entries, arg.(TableEntry))
		case t == reflect.TypeOf([]TableEntry{}):
			entries = append(entries, arg.([]TableEntry)...)
		case t == reflect.TypeOf(EntryDescription("")):
			tableLevelEntryDescription = arg.(EntryDescription).render
		case t.Kind() == reflect.Func && t.NumOut() == 1 && t.Out(0) == reflect.TypeOf(""):
			tableLevelEntryDescription = arg
		case t.Kind() == reflect.Func:
			if itBody != nil {
				exitIfErr(types.GinkgoErrors.MultipleEntryBodyFunctionsForTable(cl))
			}
			itBody = arg
		default:
			containerNodeArgs = append(containerNodeArgs, arg)
		}
	}

	containerNodeArgs = append(containerNodeArgs, func() {
		for _, entry := range entries {
			var err error
			entry := entry
			var description string
			switch t := reflect.TypeOf(entry.description); {
			case t == nil:
				err = validateParameters(tableLevelEntryDescription, entry.parameters, "Entry Description function", entry.codeLocation)
				if err == nil {
					description = invokeFunction(tableLevelEntryDescription, entry.parameters)[0].String()
				}
			case t == reflect.TypeOf(EntryDescription("")):
				description = entry.description.(EntryDescription).render(entry.parameters...)
			case t == reflect.TypeOf(""):
				description = entry.description.(string)
			case t.Kind() == reflect.Func && t.NumOut() == 1 && t.Out(0) == reflect.TypeOf(""):
				err = validateParameters(entry.description, entry.parameters, "Entry Description function", entry.codeLocation)
				if err == nil {
					description = invokeFunction(entry.description, entry.parameters)[0].String()
				}
			default:
				err = types.GinkgoErrors.InvalidEntryDescription(entry.codeLocation)
			}

			if err == nil {
				err = validateParameters(itBody, entry.parameters, "Table Body function", entry.codeLocation)
			}
			itNodeArgs := []interface{}{entry.codeLocation}
			itNodeArgs = append(itNodeArgs, entry.decorations...)
			itNodeArgs = append(itNodeArgs, func() {
				if err != nil {
					panic(err)
				}
				invokeFunction(itBody, entry.parameters)
			})

			pushNode(internal.NewNode(deprecationTracker, types.NodeTypeIt, description, itNodeArgs...))
		}
	})

	pushNode(internal.NewNode(deprecationTracker, types.NodeTypeContainer, description, containerNodeArgs...))
}

func invokeFunction(function interface{}, parameters []interface{}) []reflect.Value {
	inValues := make([]reflect.Value, len(parameters))

	funcType := reflect.TypeOf(function)
	limit := funcType.NumIn()
	if funcType.IsVariadic() {
		limit = limit - 1
	}

	for i := 0; i < limit && i < len(parameters); i++ {
		inValues[i] = computeValue(parameters[i], funcType.In(i))
	}

	if funcType.IsVariadic() {
		variadicType := funcType.In(limit).Elem()
		for i := limit; i < len(parameters); i++ {
			inValues[i] = computeValue(parameters[i], variadicType)
		}
	}

	return reflect.ValueOf(function).Call(inValues)
}

func validateParameters(function interface{}, parameters []interface{}, kind string, cl types.CodeLocation) error {
	funcType := reflect.TypeOf(function)
	limit := funcType.NumIn()
	if funcType.IsVariadic() {
		limit = limit - 1
	}
	if len(parameters) < limit {
		return types.GinkgoErrors.TooFewParametersToTableFunction(limit, len(parameters), kind, cl)
	}
	if len(parameters) > limit && !funcType.IsVariadic() {
		return types.GinkgoErrors.TooManyParametersToTableFunction(limit, len(parameters), kind, cl)
	}
	var i = 0
	for ; i < limit; i++ {
		actual := reflect.TypeOf(parameters[i])
		expected := funcType.In(i)
		if !(actual == nil) && !actual.AssignableTo(expected) {
			return types.GinkgoErrors.IncorrectParameterTypeToTableFunction(i+1, expected, actual, kind, cl)
		}
	}
	if funcType.IsVariadic() {
		expected := funcType.In(limit).Elem()
		for ; i < len(parameters); i++ {
			actual := reflect.TypeOf(parameters[i])
			if !(actual == nil) && !actual.AssignableTo(expected) {
				return types.GinkgoErrors.IncorrectVariadicParameterTypeToTableFunction(expected, actual, kind, cl)
			}
		}
	}

	return nil
}

func computeValue(parameter interface{}, t reflect.Type) reflect.Value {
	if parameter == nil {
		return reflect.Zero(t)
	} else {
		return reflect.ValueOf(parameter)
	}
}
