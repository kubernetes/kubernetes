package proxyprotocol

import (
	"net"
)

const bufferSize = 1400

// SourceChecker check trusted address
type SourceChecker func(net.Addr) (bool, error)

// NewListener construct Listener
func NewListener(listener net.Listener, headerParserBuilder HeaderParserBuilder) Listener {
	return Listener{
		Listener:            listener,
		HeaderParserBuilder: headerParserBuilder,
	}
}

// Listener implement net.Listener
type Listener struct {
	net.Listener
	Logger
	HeaderParserBuilder
	SourceChecker
}

// WithLogger copy Listener and set Logger
func (listener Listener) WithLogger(logger Logger) Listener {
	newListener := listener
	newListener.Logger = logger
	return newListener
}

// WithHeaderParserBuilder copy Listener and set HeaderParserBuilder.
// Can be used to disable or reorder HeaderParser's.
func (listener Listener) WithHeaderParserBuilder(headerParserBuilder HeaderParserBuilder) Listener {
	newListener := listener
	newListener.HeaderParserBuilder = headerParserBuilder
	return newListener
}

// WithSourceChecker copy Listener and set SourceChecker
func (listener Listener) WithSourceChecker(sourceChecker SourceChecker) Listener {
	newListener := listener
	newListener.SourceChecker = sourceChecker
	return newListener
}

// Accept implement net.Listener.Accept().
//
// When listener have SourceChecker, then check source address.
// If source checker return error, then return error.
// If source checker return false, then return raw connection.
//
// Otherwise connection wrapped into Conn with header parser.
func (listener Listener) Accept() (net.Conn, error) {
	rawConn, err := listener.Listener.Accept()
	if err != nil {
		return nil, err
	}

	logger := FallbackLogger{Logger: listener.Logger}
	trusted := true
	if listener.SourceChecker != nil {
		trusted, err = listener.SourceChecker(rawConn.RemoteAddr())
		if err != nil {
			logger.Printf("Source check error: %s", err)
			return nil, err
		}
	}

	if trusted {
		logger.Printf("Trusted connection")
	} else {
		logger.Printf("Not trusted connection")
	}

	headerParser := listener.HeaderParserBuilder.Build(logger)

	return NewConn(rawConn, logger, headerParser, trusted), nil
}
