package proxyprotocol

import "net"

// HeaderParserBuilderFunc wrap builder func into HeaderParserBuilder
type HeaderParserBuilderFunc func(logger Logger) HeaderParser

// Build implement HeaderParserBuilder for build func
func (funcBuilder HeaderParserBuilderFunc) Build(logger Logger) HeaderParser {
	return funcBuilder(logger)
}

// TextHeaderParserBuilder build TextHeaderParser
var TextHeaderParserBuilder = HeaderParserBuilderFunc(func(logger Logger) HeaderParser {
	return NewTextHeaderParser(logger)
})

// BinaryHeaderParserBuilder build BinaryHeaderParser
var BinaryHeaderParserBuilder = HeaderParserBuilderFunc(func(logger Logger) HeaderParser {
	return NewBinaryHeaderParser(logger)
})

// StubHeaderParserBuilder build StubHeaderParser
var StubHeaderParserBuilder = HeaderParserBuilderFunc(func(logger Logger) HeaderParser {
	return NewStubHeaderParser()
})

// DefaultFallbackHeaderParserBuilder build FallbackHeaderParserBuilder with
// default HeaderParserList (TextHeaderParser, BinaryHeaderParser, StubHeaderParser)
var DefaultFallbackHeaderParserBuilder = NewFallbackHeaderParserBuilder(
	TextHeaderParserBuilder,
	BinaryHeaderParserBuilder,
	StubHeaderParserBuilder,
)

// NewDefaultListener construct proxyprotocol.Listener from other net.Listener
// with DefaultFallbackHeaderParserBuilder.
func NewDefaultListener(listener net.Listener) Listener {
	return NewListener(
		listener,
		DefaultFallbackHeaderParserBuilder,
	)
}
