package proxyprotocol

import (
	"bufio"
	"errors"
)

// StubHeaderParser always return nil Header
type StubHeaderParser struct{}

// NewStubHeaderParser construct StubHeaderParser
func NewStubHeaderParser() StubHeaderParser {
	return StubHeaderParser{}
}

// Parse always return nil, nil
func (parser StubHeaderParser) Parse(*bufio.Reader) (*Header, error) {
	return nil, nil
}

// FallbackHeaderParserBuilder build FallbackHeaderParser
type FallbackHeaderParserBuilder []HeaderParserBuilder

// NewFallbackHeaderParserBuilder construct FallbackHeaderParserBuilder
func NewFallbackHeaderParserBuilder(
	headerParserBuilders ...HeaderParserBuilder,
) FallbackHeaderParserBuilder {
	return FallbackHeaderParserBuilder(headerParserBuilders)
}

// Build FallbackHeaderParser from headerParserBuilders
func (headerParserBuilders FallbackHeaderParserBuilder) Build(logger Logger) HeaderParser {
	headerParsers := make([]HeaderParser, 0, len(headerParserBuilders))
	for _, headerParserBuilder := range headerParserBuilders {
		headerParser := headerParserBuilder.Build(logger)
		headerParsers = append(headerParsers, headerParser)
	}
	return FallbackHeaderParser{
		Logger:        logger,
		HeaderParsers: headerParsers,
	}
}

// ErrInvalidHeader returned by FallbackHeaderParser when all headerParsers return
// ErrInvalidSignature
var ErrInvalidHeader = errors.New("invalid header")

// FallbackHeaderParser iterate over HeaderParser until parser not return nil error.
type FallbackHeaderParser struct {
	Logger        Logger
	HeaderParsers []HeaderParser
}

// NewFallbackHeaderParser create new instance of FallbackHeaderParser
func NewFallbackHeaderParser(logger Logger, headerParsers ...HeaderParser) FallbackHeaderParser {
	return FallbackHeaderParser{
		Logger:        logger,
		HeaderParsers: headerParsers,
	}
}

// Parse iterate over headerParsers call Parse().
//
// If any parser return not nil or not ErrInvalidSignature error, then return its error.
//
// If any parser return nil error, then return header.
//
// If all parsers return error ErrInvalidSignature, then return ErrInvalidHeader.
func (parser FallbackHeaderParser) Parse(buf *bufio.Reader) (*Header, error) {
	for _, headerParser := range parser.HeaderParsers {
		header, err := headerParser.Parse(buf)
		switch err {
		case nil:
			parser.Logger.Printf("Use header remote addr")
			return header, nil
		case ErrInvalidSignature:
			continue
		default:
			parser.Logger.Printf("Parse header error: %s", err)
			return nil, err
		}
	}
	return nil, ErrInvalidHeader
}
