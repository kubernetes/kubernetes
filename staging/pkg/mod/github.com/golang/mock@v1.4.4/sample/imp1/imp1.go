package imp1

import "bufio"

type Imp1 struct{}

type ImpT int

type ForeignEmbedded interface {
	// The return value here also makes sure that
	// the generated mock picks up the "bufio" import.
	ForeignEmbeddedMethod() *bufio.Reader

	// This method uses a type in this package,
	// which should be qualified when this interface is embedded.
	ImplicitPackage(s string, t ImpT, st []ImpT, pt *ImpT, ct chan ImpT)
}
