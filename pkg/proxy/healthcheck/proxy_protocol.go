/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package healthcheck

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
)

// proxyV2Signature is the fixed 12-byte preamble of a PROXY protocol v2 header.
// See https://www.haproxy.org/download/3.4/doc/proxy-protocol.txt section 2.2.
var proxyV2Signature = []byte{0x0D, 0x0A, 0x0D, 0x0A, 0x00, 0x0D, 0x0A, 0x51, 0x55, 0x49, 0x54, 0x0A}

const (
	// signature (12) + version/command (1) + family/protocol (1) + length (2)
	proxyV2HeaderLen = 16
	// Upper bound for the variable part; the length field is uint16, but
	// anything near that is not a header any sane load balancer would send.
	proxyV2MaxAddrLen = 4096
)

// proxyProtocolListener wraps a net.Listener so that connections optionally
// starting with a PROXY protocol v2 header have that header discarded before
// the HTTP server sees the stream. Some cloud load balancers send the header
// on health check probes when PROXY protocol is enabled on the backend; if we
// don't remove them then the HTTP server will reply 400 and the load balancer
// will mark the node unhealthy.
//
// Detection is unambiguous: neither an HTTP request nor a TLS handshake can
// start with the v2 signature, so plain HTTP probes keep working unchanged.
type proxyProtocolListener struct {
	net.Listener
}

func (l proxyProtocolListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	// Header parsing is deferred to the first Read, which runs in the
	// per-connection goroutine, so a stalled client cannot block Accept.
	return &proxyProtocolConn{Conn: c, r: bufio.NewReader(c)}, nil
}

type proxyProtocolConn struct {
	net.Conn
	r       *bufio.Reader
	once    sync.Once
	initErr error
}

func (c *proxyProtocolConn) Read(b []byte) (int, error) {
	c.once.Do(func() { c.initErr = stripProxyV2Header(c.r) })
	if c.initErr != nil {
		return 0, c.initErr
	}
	return c.r.Read(b)
}

// stripProxyV2Header discards a PROXY v2 header from r if one is present.
// Waiting for a full signature never delays a legitimate probe: any HTTP
// request line or TLS ClientHello is longer than 12 bytes.
func stripProxyV2Header(r *bufio.Reader) error {
	p, err := r.Peek(len(proxyV2Signature))
	if err != nil {
		if err == io.EOF {
			// Stream ended before 12 bytes: let the HTTP server handle what arrived.
			return nil
		}
		// Timeout, reset, etc.: nothing left for the HTTP server to handle.
		return err
	}
	if !bytes.Equal(p, proxyV2Signature) {
		// Not a PROXY v2 header: pass the stream through untouched.
		return nil
	}

	hdr, err := r.Peek(proxyV2HeaderLen)
	if err != nil {
		return fmt.Errorf("reading PROXY protocol v2 header: %w", err)
	}
	if version := hdr[12] >> 4; version != 2 {
		return fmt.Errorf("unsupported PROXY protocol version %d, expected 2", version)
	}
	// The spec requires receivers to drop anything other than LOCAL (0x0) or PROXY (0x1).
	if cmd := hdr[12] & 0x0F; cmd > 0x1 {
		return fmt.Errorf("unsupported PROXY protocol v2 command 0x%x, expected 0x0 (LOCAL) or 0x1 (PROXY)", cmd)
	}
	addrLen := int(binary.BigEndian.Uint16(hdr[14:16]))
	if addrLen > proxyV2MaxAddrLen {
		return fmt.Errorf("PROXY protocol v2 address block too large: %d bytes, max %d", addrLen, proxyV2MaxAddrLen)
	}
	if _, err := r.Discard(proxyV2HeaderLen + addrLen); err != nil {
		return fmt.Errorf("discarding PROXY protocol v2 header: %w", err)
	}
	return nil
}
