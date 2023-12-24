package root2

import (
	"k8s.io/gengo/v2/parser/testdata/root2/lib2"
	"k8s.io/gengo/v2/parser/testdata/rootpeer"
)

// SecondClosest X comment

// X comment
var X = lib2.X + rootpeer.X

// SecondClosest Y comment

// Y comment
const Y = "Y"

// SecondClosest Int comment

// Int comment
type Int int

// SecondClosest String comment

// String comment
type String string

// SecondClosest EmptyStruct comment

// EmptyStruct comment
type EmptyStruct struct{}

// SecondClosest Struct comment

// Struct comment
type Struct struct {
	// SecondClosest I comment

	// I comment
	I int
	// PI comment
	PI *int
	// S comment
	S String
	// PS comment
	PS *string
	// II comment
	II Int
	// PII comment
	PII *Int
	// SS comment
	SS String
	// PSS comment
	PSS *String
	// ES comment
	ES EmptyStruct
	// PES comment
	PES *EmptyStruct
}

// SecondClosest privateMethod comment

// privateMethod comment
func (Struct) privateMethod() {}

// SecondClosest PublicMethod comment

// PublicMethod comment
func (Struct) PublicMethod() {}

// SecondClosest M comment

// M comment
type M map[string]*Struct

// SecondClosest privateFunc comment

// privateFunc comment
func privateFunc() {}

// SecondClosest PublicFunc comment

// PublicFunc comment
func PublicFunc() {}
