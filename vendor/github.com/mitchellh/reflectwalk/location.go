package reflectwalk

//go:generate stringer -type=Location location.go

type Location uint

const (
	None Location = iota
	Map
	MapKey
	MapValue
	Slice
	SliceElem
	Struct
	StructField
	WalkLoc
)
