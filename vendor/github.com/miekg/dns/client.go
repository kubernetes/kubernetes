package dns

// A client implementation.

import (
	"bytes"
	"crypto/tls"
	"encoding/binary"
	"io"
	"net"
	"time"
)

const dnsTimeout time.Duration = 2 * time.Second
const tcpIdleTimeout time.Duration = 8 * time.Second

// A Conn represents a connection to a DNS server.
type Conn struct {
	net.Conn                         // a net.Conn holding the connection
	UDPSize        uint16            // minimum receive buffer for UDP messages
	TsigSecret     map[string]string // secret(s) for Tsig map[<zonename>]<base64 secret>, zonename must be fully qualified
	rtt            time.Duration
	t              time.Time
	tsigRequestMAC string
}

// A Client defines parameters for a DNS client.
type Client struct {
	Net            string            // if "tcp" or "tcp-tls" (DNS over TLS) a TCP query will be initiated, otherwise an UDP one (default is "" for UDP)
	UDPSize        uint16            // minimum receive buffer for UDP messages
	TLSConfig      *tls.Config       // TLS connection configuration
	Timeout        time.Duration     // a cumulative timeout for dial, write and read, defaults to 0 (disabled) - overrides DialTimeout, ReadTimeout and WriteTimeout when non-zero
	DialTimeout    time.Duration     // net.DialTimeout, defaults to 2 seconds - overridden by Timeout when that value is non-zero
	ReadTimeout    time.Duration     // net.Conn.SetReadTimeout value for connections, defaults to 2 seconds - overridden by Timeout when that value is non-zero
	WriteTimeout   time.Duration     // net.Conn.SetWriteTimeout value for connections, defaults to 2 seconds - overridden by Timeout when that value is non-zero
	TsigSecret     map[string]string // secret(s) for Tsig map[<zonename>]<base64 secret>, zonename must be fully qualified
	SingleInflight bool              // if true suppress multiple outstanding queries for the same Qname, Qtype and Qclass
	group          singleflight
}

// Exchange performs a synchronous UDP query. It sends the message m to the address
// contained in a and waits for an reply. Exchange does not retry a failed query, nor
// will it fall back to TCP in case of truncation.
// See client.Exchange for more information on setting larger buffer sizes.
func Exchange(m *Msg, a string) (r *Msg, err error) {
	var co *Conn
	co, err = DialTimeout("udp", a, dnsTimeout)
	if err != nil {
		return nil, err
	}

	defer co.Close()

	opt := m.IsEdns0()
	// If EDNS0 is used use that for size.
	if opt != nil && opt.UDPSize() >= MinMsgSize {
		co.UDPSize = opt.UDPSize()
	}

	co.SetWriteDeadline(time.Now().Add(dnsTimeout))
	if err = co.WriteMsg(m); err != nil {
		return nil, err
	}

	co.SetReadDeadline(time.Now().Add(dnsTimeout))
	r, err = co.ReadMsg()
	if err == nil && r.Id != m.Id {
		err = ErrId
	}
	return r, err
}

// ExchangeConn performs a synchronous query. It sends the message m via the connection
// c and waits for a reply. The connection c is not closed by ExchangeConn.
// This function is going away, but can easily be mimicked:
//
//	co := &dns.Conn{Conn: c} // c is your net.Conn
//	co.WriteMsg(m)
//	in, _  := co.ReadMsg()
//	co.Close()
//
func ExchangeConn(c net.Conn, m *Msg) (r *Msg, err error) {
	println("dns: this function is deprecated")
	co := new(Conn)
	co.Conn = c
	if err = co.WriteMsg(m); err != nil {
		return nil, err
	}
	r, err = co.ReadMsg()
	if err == nil && r.Id != m.Id {
		err = ErrId
	}
	return r, err
}

// Exchange performs an synchronous query. It sends the message m to the address
// contained in a and waits for an reply. Basic use pattern with a *dns.Client:
//
//	c := new(dns.Client)
//	in, rtt, err := c.Exchange(message, "127.0.0.1:53")
//
// Exchange does not retry a failed query, nor will it fall back to TCP in
// case of truncation.
// It is up to the caller to create a message that allows for larger responses to be
// returned. Specifically this means adding an EDNS0 OPT RR that will advertise a larger
// buffer, see SetEdns0. Messsages without an OPT RR will fallback to the historic limit
// of 512 bytes.
func (c *Client) Exchange(m *Msg, a string) (r *Msg, rtt time.Duration, err error) {
	if !c.SingleInflight {
		return c.exchange(m, a)
	}
	// This adds a bunch of garbage, TODO(miek).
	t := "nop"
	if t1, ok := TypeToString[m.Question[0].Qtype]; ok {
		t = t1
	}
	cl := "nop"
	if cl1, ok := ClassToString[m.Question[0].Qclass]; ok {
		cl = cl1
	}
	r, rtt, err, shared := c.group.Do(m.Question[0].Name+t+cl, func() (*Msg, time.Duration, error) {
		return c.exchange(m, a)
	})
	if err != nil {
		return r, rtt, err
	}
	if shared {
		return r.Copy(), rtt, nil
	}
	return r, rtt, nil
}

func (c *Client) dialTimeout() time.Duration {
	if c.Timeout != 0 {
		return c.Timeout
	}
	if c.DialTimeout != 0 {
		return c.DialTimeout
	}
	return dnsTimeout
}

func (c *Client) readTimeout() time.Duration {
	if c.ReadTimeout != 0 {
		return c.ReadTimeout
	}
	return dnsTimeout
}

func (c *Client) writeTimeout() time.Duration {
	if c.WriteTimeout != 0 {
		return c.WriteTimeout
	}
	return dnsTimeout
}

func (c *Client) exchange(m *Msg, a string) (r *Msg, rtt time.Duration, err error) {
	var co *Conn
	network := "udp"
	tls := false

	switch c.Net {
	case "tcp-tls":
		network = "tcp"
		tls = true
	case "tcp4-tls":
		network = "tcp4"
		tls = true
	case "tcp6-tls":
		network = "tcp6"
		tls = true
	default:
		if c.Net != "" {
			network = c.Net
		}
	}

	var deadline time.Time
	if c.Timeout != 0 {
		deadline = time.Now().Add(c.Timeout)
	}

	if tls {
		co, err = DialTimeoutWithTLS(network, a, c.TLSConfig, c.dialTimeout())
	} else {
		co, err = DialTimeout(network, a, c.dialTimeout())
	}

	if err != nil {
		return nil, 0, err
	}
	defer co.Close()

	opt := m.IsEdns0()
	// If EDNS0 is used use that for size.
	if opt != nil && opt.UDPSize() >= MinMsgSize {
		co.UDPSize = opt.UDPSize()
	}
	// Otherwise use the client's configured UDP size.
	if opt == nil && c.UDPSize >= MinMsgSize {
		co.UDPSize = c.UDPSize
	}

	co.TsigSecret = c.TsigSecret
	co.SetWriteDeadline(deadlineOrTimeout(deadline, c.writeTimeout()))
	if err = co.WriteMsg(m); err != nil {
		return nil, 0, err
	}

	co.SetReadDeadline(deadlineOrTimeout(deadline, c.readTimeout()))
	r, err = co.ReadMsg()
	if err == nil && r.Id != m.Id {
		err = ErrId
	}
	return r, co.rtt, err
}

// ReadMsg reads a message from the connection co.
// If the received message contains a TSIG record the transaction
// signature is verified.
func (co *Conn) ReadMsg() (*Msg, error) {
	p, err := co.ReadMsgHeader(nil)
	if err != nil {
		return nil, err
	}

	m := new(Msg)
	if err := m.Unpack(p); err != nil {
		// If ErrTruncated was returned, we still want to allow the user to use
		// the message, but naively they can just check err if they don't want
		// to use a truncated message
		if err == ErrTruncated {
			return m, err
		}
		return nil, err
	}
	if t := m.IsTsig(); t != nil {
		if _, ok := co.TsigSecret[t.Hdr.Name]; !ok {
			return m, ErrSecret
		}
		// Need to work on the original message p, as that was used to calculate the tsig.
		err = TsigVerify(p, co.TsigSecret[t.Hdr.Name], co.tsigRequestMAC, false)
	}
	return m, err
}

// ReadMsgHeader reads a DNS message, parses and populates hdr (when hdr is not nil).
// Returns message as a byte slice to be parsed with Msg.Unpack later on.
// Note that error handling on the message body is not possible as only the header is parsed.
func (co *Conn) ReadMsgHeader(hdr *Header) ([]byte, error) {
	var (
		p   []byte
		n   int
		err error
	)

	switch t := co.Conn.(type) {
	case *net.TCPConn, *tls.Conn:
		r := t.(io.Reader)

		// First two bytes specify the length of the entire message.
		l, err := tcpMsgLen(r)
		if err != nil {
			return nil, err
		}
		p = make([]byte, l)
		n, err = tcpRead(r, p)
		co.rtt = time.Since(co.t)
	default:
		if co.UDPSize > MinMsgSize {
			p = make([]byte, co.UDPSize)
		} else {
			p = make([]byte, MinMsgSize)
		}
		n, err = co.Read(p)
		co.rtt = time.Since(co.t)
	}

	if err != nil {
		return nil, err
	} else if n < headerSize {
		return nil, ErrShortRead
	}

	p = p[:n]
	if hdr != nil {
		dh, _, err := unpackMsgHdr(p, 0)
		if err != nil {
			return nil, err
		}
		*hdr = dh
	}
	return p, err
}

// tcpMsgLen is a helper func to read first two bytes of stream as uint16 packet length.
func tcpMsgLen(t io.Reader) (int, error) {
	p := []byte{0, 0}
	n, err := t.Read(p)
	if err != nil {
		return 0, err
	}
	if n != 2 {
		return 0, ErrShortRead
	}
	l := binary.BigEndian.Uint16(p)
	if l == 0 {
		return 0, ErrShortRead
	}
	return int(l), nil
}

// tcpRead calls TCPConn.Read enough times to fill allocated buffer.
func tcpRead(t io.Reader, p []byte) (int, error) {
	n, err := t.Read(p)
	if err != nil {
		return n, err
	}
	for n < len(p) {
		j, err := t.Read(p[n:])
		if err != nil {
			return n, err
		}
		n += j
	}
	return n, err
}

// Read implements the net.Conn read method.
func (co *Conn) Read(p []byte) (n int, err error) {
	if co.Conn == nil {
		return 0, ErrConnEmpty
	}
	if len(p) < 2 {
		return 0, io.ErrShortBuffer
	}
	switch t := co.Conn.(type) {
	case *net.TCPConn, *tls.Conn:
		r := t.(io.Reader)

		l, err := tcpMsgLen(r)
		if err != nil {
			return 0, err
		}
		if l > len(p) {
			return int(l), io.ErrShortBuffer
		}
		return tcpRead(r, p[:l])
	}
	// UDP connection
	n, err = co.Conn.Read(p)
	if err != nil {
		return n, err
	}
	return n, err
}

// WriteMsg sends a message through the connection co.
// If the message m contains a TSIG record the transaction
// signature is calculated.
func (co *Conn) WriteMsg(m *Msg) (err error) {
	var out []byte
	if t := m.IsTsig(); t != nil {
		mac := ""
		if _, ok := co.TsigSecret[t.Hdr.Name]; !ok {
			return ErrSecret
		}
		out, mac, err = TsigGenerate(m, co.TsigSecret[t.Hdr.Name], co.tsigRequestMAC, false)
		// Set for the next read, although only used in zone transfers
		co.tsigRequestMAC = mac
	} else {
		out, err = m.Pack()
	}
	if err != nil {
		return err
	}
	co.t = time.Now()
	if _, err = co.Write(out); err != nil {
		return err
	}
	return nil
}

// Write implements the net.Conn Write method.
func (co *Conn) Write(p []byte) (n int, err error) {
	switch t := co.Conn.(type) {
	case *net.TCPConn, *tls.Conn:
		w := t.(io.Writer)

		lp := len(p)
		if lp < 2 {
			return 0, io.ErrShortBuffer
		}
		if lp > MaxMsgSize {
			return 0, &Error{err: "message too large"}
		}
		l := make([]byte, 2, lp+2)
		binary.BigEndian.PutUint16(l, uint16(lp))
		p = append(l, p...)
		n, err := io.Copy(w, bytes.NewReader(p))
		return int(n), err
	}
	n, err = co.Conn.(*net.UDPConn).Write(p)
	return n, err
}

// Dial connects to the address on the named network.
func Dial(network, address string) (conn *Conn, err error) {
	conn = new(Conn)
	conn.Conn, err = net.Dial(network, address)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

// DialTimeout acts like Dial but takes a timeout.
func DialTimeout(network, address string, timeout time.Duration) (conn *Conn, err error) {
	conn = new(Conn)
	conn.Conn, err = net.DialTimeout(network, address, timeout)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

// DialWithTLS connects to the address on the named network with TLS.
func DialWithTLS(network, address string, tlsConfig *tls.Config) (conn *Conn, err error) {
	conn = new(Conn)
	conn.Conn, err = tls.Dial(network, address, tlsConfig)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

// DialTimeoutWithTLS acts like DialWithTLS but takes a timeout.
func DialTimeoutWithTLS(network, address string, tlsConfig *tls.Config, timeout time.Duration) (conn *Conn, err error) {
	var dialer net.Dialer
	dialer.Timeout = timeout

	conn = new(Conn)
	conn.Conn, err = tls.DialWithDialer(&dialer, network, address, tlsConfig)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

func deadlineOrTimeout(deadline time.Time, timeout time.Duration) time.Time {
	if deadline.IsZero() {
		return time.Now().Add(timeout)
	}
	return deadline
}
