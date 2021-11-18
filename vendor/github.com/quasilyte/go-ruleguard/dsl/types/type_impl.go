package types

// Method stubs to make various types implement Type interface.
//
// Nothing interesting here, hence it's moved to a separate file.

func (*Array) String() string     { return "" }
func (*Slice) String() string     { return "" }
func (*Pointer) String() string   { return "" }
func (*Interface) String() string { return "" }

func (*Array) Underlying() Type     { return nil }
func (*Slice) Underlying() Type     { return nil }
func (*Pointer) Underlying() Type   { return nil }
func (*Interface) Underlying() Type { return nil }
