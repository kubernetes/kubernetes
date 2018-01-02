package cldr_test

import (
	"fmt"

	"golang.org/x/text/unicode/cldr"
)

func ExampleSlice() {
	var dr *cldr.CLDR // assume this is initialized

	x, _ := dr.LDML("en")
	cs := x.Collations.Collation
	// remove all but the default
	cldr.MakeSlice(&cs).Filter(func(e cldr.Elem) bool {
		return e.GetCommon().Type != x.Collations.Default()
	})
	for i, c := range cs {
		fmt.Println(i, c.Type)
	}
}
