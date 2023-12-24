package root1

import (
	"k8s.io/gengo/v2/parser/testdata/root1/lib1"
	"k8s.io/gengo/v2/parser/testdata/rootpeer"
)

// This comment must be on lines 8 and 9
// or else unit tests will fail.
var X = lib1.X + rootpeer.X
