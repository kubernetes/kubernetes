/*

Table provides a simple DSL for Ginkgo-native Table-Driven Tests

The godoc documentation describes Table's API.  More comprehensive documentation (with examples!) is available at http://onsi.github.io/ginkgo#table-driven-tests

*/

package table

import (
	"fmt"
	"reflect"

	"github.com/onsi/ginkgo"
)

/*
DescribeTable describes a table-driven test.

For example:

    DescribeTable("a simple table",
        func(x int, y int, expected bool) {
            Ω(x > y).Should(Equal(expected))
        },
        Entry("x > y", 1, 0, true),
        Entry("x == y", 0, 0, false),
        Entry("x < y", 0, 1, false),
    )

The first argument to `DescribeTable` is a string description.
The second argument is a function that will be run for each table entry.  Your assertions go here - the function is equivalent to a Ginkgo It.
The subsequent arguments must be of type `TableEntry`.  We recommend using the `Entry` convenience constructors.

The `Entry` constructor takes a string description followed by an arbitrary set of parameters.  These parameters are passed into your function.

Under the hood, `DescribeTable` simply generates a new Ginkgo `Describe`.  Each `Entry` is turned into an `It` within the `Describe`.

It's important to understand that the `Describe`s and `It`s are generated at evaluation time (i.e. when Ginkgo constructs the tree of tests and before the tests run).

Individual Entries can be focused (with FEntry) or marked pending (with PEntry or XEntry).  In addition, the entire table can be focused or marked pending with FDescribeTable and PDescribeTable/XDescribeTable.
*/
func DescribeTable(description string, itBody interface{}, entries ...TableEntry) bool {
	describeTable(description, itBody, entries, false, false)
	return true
}

/*
You can focus a table with `FDescribeTable`.  This is equivalent to `FDescribe`.
*/
func FDescribeTable(description string, itBody interface{}, entries ...TableEntry) bool {
	describeTable(description, itBody, entries, false, true)
	return true
}

/*
You can mark a table as pending with `PDescribeTable`.  This is equivalent to `PDescribe`.
*/
func PDescribeTable(description string, itBody interface{}, entries ...TableEntry) bool {
	describeTable(description, itBody, entries, true, false)
	return true
}

/*
You can mark a table as pending with `XDescribeTable`.  This is equivalent to `XDescribe`.
*/
func XDescribeTable(description string, itBody interface{}, entries ...TableEntry) bool {
	describeTable(description, itBody, entries, true, false)
	return true
}

func describeTable(description string, itBody interface{}, entries []TableEntry, pending bool, focused bool) {
	itBodyValue := reflect.ValueOf(itBody)
	if itBodyValue.Kind() != reflect.Func {
		panic(fmt.Sprintf("DescribeTable expects a function, got %#v", itBody))
	}

	if pending {
		ginkgo.PDescribe(description, func() {
			for _, entry := range entries {
				entry.generateIt(itBodyValue)
			}
		})
	} else if focused {
		ginkgo.FDescribe(description, func() {
			for _, entry := range entries {
				entry.generateIt(itBodyValue)
			}
		})
	} else {
		ginkgo.Describe(description, func() {
			for _, entry := range entries {
				entry.generateIt(itBodyValue)
			}
		})
	}
}
