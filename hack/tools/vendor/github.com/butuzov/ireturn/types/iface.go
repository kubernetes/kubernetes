package types

type IFace struct {
	Name string // Preserved for named interfaces
	Pos  int    // Position in return tuple
	Type IType  // Type of the interface
}
