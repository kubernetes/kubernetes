package proxyprotocol

import (
	"bufio"
	"net"
	"sync"
)

// Conn is wrapper on net.Conn with RemoteAddr() override.
//
// On first call Read() or RemoteAddr() parse proxyprotocol header and store
// local and remote addresses.
type Conn struct {
	net.Conn
	logger       Logger
	readBuf      *bufio.Reader
	header       *Header
	headerErr    error
	headerParser HeaderParser
	trustedAddr  bool
	once         sync.Once
}

// NewConn create wrapper on net.Conn.
func NewConn(conn net.Conn, logger Logger, headerParser HeaderParser, trustedAddr bool) net.Conn {
	readBuf := bufio.NewReaderSize(conn, bufferSize)

	return &Conn{
		Conn:         conn,
		readBuf:      readBuf,
		logger:       logger,
		headerParser: headerParser,
		trustedAddr:  trustedAddr,
	}
}

func (conn *Conn) parseHeader() {
	conn.header, conn.headerErr = conn.headerParser.Parse(conn.readBuf)
	if conn.headerErr != nil {
		conn.logger.Printf("Header parse error: %s", conn.headerErr)
		return
	}
	conn.logger.Printf("Header parsed %v", conn.header)
}

// Read on first call parse proxyprotocol header.
//
// If header parser return error, then error stored and returned. Otherwise call
// Read on source connection.
//
// Following calls of Read function check parse header error.
// If error not nil, then error returned. Otherwise called source "conn.Read".
func (conn *Conn) Read(buf []byte) (int, error) {
	conn.once.Do(conn.parseHeader)

	if conn.headerErr != nil {
		return 0, conn.headerErr
	}

	return conn.readBuf.Read(buf)
}

// LocalAddr proxy to conn.LocalAddr
func (conn *Conn) LocalAddr() net.Addr {
	conn.once.Do(conn.parseHeader)

	if conn.trustedAddr && conn.header != nil {
		return conn.header.DstAddr
	}

	return conn.Conn.LocalAddr()
}

// RemoteAddr on first call parse proxyprotocol header.
//
// If header parser return header, then return source address from header.
// Otherwise return original source address.
func (conn *Conn) RemoteAddr() net.Addr {
	conn.once.Do(conn.parseHeader)

	if conn.trustedAddr && conn.header != nil {
		return conn.header.SrcAddr
	}

	return conn.Conn.RemoteAddr()
}
