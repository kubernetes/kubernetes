package ginkgo

import (
	"context"
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

When passed into an Entry the EntryDescription is used to generate the name or that entry.  When passed to DescribeTable, the EntryDescription is used to generate the names for any entries that have `nil` descriptions.

You can learn more about generating EntryDescriptions here: https://onsi.github.io/ginkgo/#generating-entry-descriptions
*/
type EntryDescription string

func (ed EntryDescription) render(args ...any) string {
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
func DescribeTable(description string, args ...any) bool {
	GinkgoHelper()
	generateTable(description, false, args...)
	return true
}

/*
You can focus a table with `FDescribeTable`.  This is equivalent to `FDescribe`.
*/
func FDescribeTable(description string, args ...any) bool {
	GinkgoHelper()
	args = append(args, internal.Focus)
	generateTable(description, false, args...)
	return true
}

/*
You can mark a table as pending with `PDescribeTable`.  This is equivalent to `PDescribe`.
*/
func PDescribeTable(description string, args ...any) bool {
	GinkgoHelper()
	args = append(args, internal.Pending)
	generateTable(description, false, args...)
	return true
}

/*
You can mark a table as pending with `XDescribeTable`.  This is equivalent to `XDescribe`.
*/
var XDescribeTable = PDescribeTable

/*
DescribeTableSubtree describes a table-driven spec that generates a set of tests for each entry.

For example:

	DescribeTableSubtree("a subtree table",
	    func(url string, code int, message string) {
			var resp *http.Response
			BeforeEach(func() {
				var err error
				resp, err = http.Get(url)
				Expect(err).NotTo(HaveOccurred())
				DeferCleanup(resp.Body.Close)
			})

			It("should return the expected status code", func() {
				Expect(resp.StatusCode).To(Equal(code))
			})

			It("should return the expected message", func() {
				body, err := io.ReadAll(resp.Body)
				Expect(err).NotTo(HaveOccurred())
				Expect(string(body)).To(Equal(message))
			})
	    },
	    Entry("default response", "example.com/response", http.StatusOK, "hello world"),
	    Entry("missing response", "example.com/missing", http.StatusNotFound, "wat?"),
	)

Note that you **must** place define an It inside the body function.

You can learn more about DescribeTableSubtree here: https://onsi.github.io/ginkgo/#table-specs
And can explore some Table patterns here: https://onsi.github.io/ginkgo/#table-specs-patterns
*/
func DescribeTableSubtree(description string, args ...any) bool {
	GinkgoHelper()
	generateTable(description, true, args...)
	return true
}

/*
You can focus a table with `FDescribeTableSubtree`.  This is equivalent to `FDescribe`.
*/
func FDescribeTableSubtree(description string, args ...any) bool {
	GinkgoHelper()
	args = append(args, internal.Focus)
	generateTable(description, true, args...)
	return true
}

/*
You can mark a table as pending with `PDescribeTableSubtree`.  This is equivalent to `PDescribe`.
*/
func PDescribeTableSubtree(description string, args ...any) bool {
	GinkgoHelper()
	args = append(args, internal.Pending)
	generateTable(description, true, args...)
	return true
}

/*
You can mark a table as pending with `XDescribeTableSubtree`.  This is equivalent to `XDescribe`.
*/
var XDescribeTableSubtree = PDescribeTableSubtree

/*
TableEntry represents an entry in a table test.  You generally use the `Entry` constructor.
*/
type TableEntry struct {
	description  any
	decorations  []any
	parameters   []any
	codeLocation types.CodeLocation
}

/*
Entry constructs a TableEntry.

The first argument is a description.  This can be a string, a function that accepts the parameters passed to the TableEntry and returns a string, an EntryDescription format string, or nil.  If nil is provided then the name of the Entry is derived using the table-level entry description.
Subsequent arguments accept any Ginkgo decorators.  These are filtered out and the remaining arguments are passed into the Spec function associated with the table.

Each Entry ends up generating an individual Ginkgo It.  The body of the it is the Table Body function with the Entry parameters passed in.

If you want to generate interruptible specs simply write a Table function that accepts a SpecContext as its first argument.  You can then decorate individual Entrys with the NodeTimeout and SpecTimeout decorators.

You can learn more about Entry here: https://onsi.github.io/ginkgo/#table-specs
*/
func Entry(description any, args ...any) TableEntry {
	GinkgoHelper()
	decorations, parameters := internal.PartitionDecorations(args...)
	return TableEntry{description: description, decorations: decorations, parameters: parameters, codeLocation: types.NewCodeLocation(0)}
}

/*
You can focus a particular entry with FEntry.  This is equivalent to FIt.
*/
func FEntry(description any, args ...any) TableEntry {
	GinkgoHelper()
	decorations, parameters := internal.PartitionDecorations(args...)
	decorations = append(decorations, internal.Focus)
	return TableEntry{description: description, decorations: decorations, parameters: parameters, codeLocation: types.NewCodeLocation(0)}
}

/*
You can mark a particular entry as pending with PEntry.  This is equivalent to PIt.
*/
func PEntry(description any, args ...any) TableEntry {
	GinkgoHelper()
	decorations, parameters := internal.PartitionDecorations(args...)
	decorations = append(decorations, internal.Pending)
	return TableEntry{description: description, decorations: decorations, parameters: parameters, codeLocation: types.NewCodeLocation(0)}
}

/*
You can mark a particular entry as pending with XEntry.  This is equivalent to XIt.
*/
var XEntry = PEntry

var contextType = reflect.TypeOf(new(context.Context)).Elem()
var specContextType = reflect.TypeOf(new(SpecContext)).Elem()

func generateTable(description string, isSubtree bool, args ...any) {
	GinkgoHelper()
	cl := types.NewCodeLocation(0)
	containerNodeArgs := []any{cl}

	entries := []TableEntry{}
	var internalBody any
	var internalBodyType reflect.Type

	var tableLevelEntryDescription any
	tableLevelEntryDescription = func(args ...any) string {
		out := []string{}
		for _, arg := range args {
			out = append(out, fmt.Sprint(arg))
		}
		return "Entry: " + strings.Join(out, ", ")
	}

	if len(args) == 1 {
		exitIfErr(types.GinkgoErrors.MissingParametersForTableFunction(cl))
	}

	for i, arg := range args {
		switch t := reflect.TypeOf(arg); {
		case t == nil:
			exitIfErr(types.GinkgoErrors.IncorrectParameterTypeForTable(i, "nil", cl))
		case t == reflect.TypeOf(TableEntry{}):
			entries = append(entries, arg.(TableEntry))
		case t == reflect.TypeOf([]TableEntry{}):
			entries = append(entries, arg.([]TableEntry)...)
		case t == reflect.TypeOf(EntryDescription("")):
			tableLevelEntryDescription = arg.(EntryDescription).render
		case t.Kind() == reflect.Func && t.NumOut() == 1 && t.Out(0) == reflect.TypeOf(""):
			tableLevelEntryDescription = arg
		case t.Kind() == reflect.Func:
			if internalBody != nil {
				exitIfErr(types.GinkgoErrors.MultipleEntryBodyFunctionsForTable(cl))
			}
			internalBody = arg
			internalBodyType = reflect.TypeOf(internalBody)
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
				err = validateParameters(tableLevelEntryDescription, entry.parameters, "Entry Description function", entry.codeLocation, false)
				if err == nil {
					description = invokeFunction(tableLevelEntryDescription, entry.parameters)[0].String()
				}
			case t == reflect.TypeOf(EntryDescription("")):
				description = entry.description.(EntryDescription).render(entry.parameters...)
			case t == reflect.TypeOf(""):
				description = entry.description.(string)
			case t.Kind() == reflect.Func && t.NumOut() == 1 && t.Out(0) == reflect.TypeOf(""):
				err = validateParameters(entry.description, entry.parameters, "Entry Description function", entry.codeLocation, false)
				if err == nil {
					description = invokeFunction(entry.description, entry.parameters)[0].String()
				}
			default:
				err = types.GinkgoErrors.InvalidEntryDescription(entry.codeLocation)
			}

			internalNodeArgs := []any{entry.codeLocation}
			internalNodeArgs = append(internalNodeArgs, entry.decorations...)

			hasContext := false
			if internalBodyType.NumIn() > 0 {
				if internalBodyType.In(0).Implements(specContextType) {
					hasContext = true
				} else if internalBodyType.In(0).Implements(contextType) {
					hasContext = true
					if len(entry.parameters) > 0 && reflect.TypeOf(entry.parameters[0]) != nil && reflect.TypeOf(entry.parameters[0]).Implements(contextType) {
						// we allow you to pass in a non-nil context
						hasContext = false
					}
				}
			}

			if err == nil {
				err = validateParameters(internalBody, entry.parameters, "Table Body function", entry.codeLocation, hasContext)
			}

			if hasContext {
				internalNodeArgs = append(internalNodeArgs, func(c SpecContext) {
					if err != nil {
						panic(err)
					}
					invokeFunction(internalBody, append([]any{c}, entry.parameters...))
				})
				if isSubtree {
					exitIfErr(types.GinkgoErrors.ContextsCannotBeUsedInSubtreeTables(cl))
				}
			} else {
				internalNodeArgs = append(internalNodeArgs, func() {
					if err != nil {
						panic(err)
					}
					invokeFunction(internalBody, entry.parameters)
				})
			}

			internalNodeType := types.NodeTypeIt
			if isSubtree {
				internalNodeType = types.NodeTypeContainer
			}

			pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, internalNodeType, description, internalNodeArgs...)))
		}
	})

	pushNode(internal.NewNode(internal.TransformNewNodeArgs(exitIfErrors, deprecationTracker, types.NodeTypeContainer, description, containerNodeArgs...)))
}

func invokeFunction(function any, parameters []any) []reflect.Value {
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

func validateParameters(function any, parameters []any, kind string, cl types.CodeLocation, hasContext bool) error {
	funcType := reflect.TypeOf(function)
	limit := funcType.NumIn()
	offset := 0
	if hasContext {
		limit = limit - 1
		offset = 1
	}
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
		expected := funcType.In(i + offset)
		if !(actual == nil) && !actual.AssignableTo(expected) {
			return types.GinkgoErrors.IncorrectParameterTypeToTableFunction(i+1, expected, actual, kind, cl)
		}
	}
	if funcType.IsVariadic() {
		expected := funcType.In(limit + offset).Elem()
		for ; i < len(parameters); i++ {
			actual := reflect.TypeOf(parameters[i])
			if !(actual == nil) && !actual.AssignableTo(expected) {
				return types.GinkgoErrors.IncorrectVariadicParameterTypeToTableFunction(expected, actual, kind, cl)
			}
		}
	}

	return nil
}

func computeValue(parameter any, t reflect.Type) reflect.Value {
	if parameter == nil {
		return reflect.Zero(t)
	} else {
		return reflect.ValueOf(parameter)
	}
}
