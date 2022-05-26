package types

type IType uint8

const (
	EmptyInterface    IType = 1 << iota // ref as empty
	AnonInterface                       // ref as anon
	ErrorInterface                      // ref as error
	NamedInterface                      // ref as named
	NamedStdInterface                   // ref as named stdlib
)
