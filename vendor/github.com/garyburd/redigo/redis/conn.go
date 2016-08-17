// Copyright 2012 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redis

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"regexp"
	"strconv"
	"sync"
	"time"
)

// conn is the low-level implementation of Conn
type conn struct {

	// Shared
	mu      sync.Mutex
	pending int
	err     error
	conn    net.Conn

	// Read
	readTimeout time.Duration
	br          *bufio.Reader

	// Write
	writeTimeout time.Duration
	bw           *bufio.Writer

	// Scratch space for formatting argument length.
	// '*' or '$', length, "\r\n"
	lenScratch [32]byte

	// Scratch space for formatting integers and floats.
	numScratch [40]byte
}

// DialTimeout acts like Dial but takes timeouts for establishing the
// connection to the server, writing a command and reading a reply.
//
// Deprecated: Use Dial with options instead.
func DialTimeout(network, address string, connectTimeout, readTimeout, writeTimeout time.Duration) (Conn, error) {
	return Dial(network, address,
		DialConnectTimeout(connectTimeout),
		DialReadTimeout(readTimeout),
		DialWriteTimeout(writeTimeout))
}

// DialOption specifies an option for dialing a Redis server.
type DialOption struct {
	f func(*dialOptions)
}

type dialOptions struct {
	readTimeout  time.Duration
	writeTimeout time.Duration
	dial         func(network, addr string) (net.Conn, error)
	db           int
	password     string
}

// DialReadTimeout specifies the timeout for reading a single command reply.
func DialReadTimeout(d time.Duration) DialOption {
	return DialOption{func(do *dialOptions) {
		do.readTimeout = d
	}}
}

// DialWriteTimeout specifies the timeout for writing a single command.
func DialWriteTimeout(d time.Duration) DialOption {
	return DialOption{func(do *dialOptions) {
		do.writeTimeout = d
	}}
}

// DialConnectTimeout specifies the timeout for connecting to the Redis server.
func DialConnectTimeout(d time.Duration) DialOption {
	return DialOption{func(do *dialOptions) {
		dialer := net.Dialer{Timeout: d}
		do.dial = dialer.Dial
	}}
}

// DialNetDial specifies a custom dial function for creating TCP
// connections. If this option is left out, then net.Dial is
// used. DialNetDial overrides DialConnectTimeout.
func DialNetDial(dial func(network, addr string) (net.Conn, error)) DialOption {
	return DialOption{func(do *dialOptions) {
		do.dial = dial
	}}
}

// DialDatabase specifies the database to select when dialing a connection.
func DialDatabase(db int) DialOption {
	return DialOption{func(do *dialOptions) {
		do.db = db
	}}
}

// DialPassword specifies the password to use when connecting to
// the Redis server.
func DialPassword(password string) DialOption {
	return DialOption{func(do *dialOptions) {
		do.password = password
	}}
}

// Dial connects to the Redis server at the given network and
// address using the specified options.
func Dial(network, address string, options ...DialOption) (Conn, error) {
	do := dialOptions{
		dial: net.Dial,
	}
	for _, option := range options {
		option.f(&do)
	}

	netConn, err := do.dial(network, address)
	if err != nil {
		return nil, err
	}
	c := &conn{
		conn:         netConn,
		bw:           bufio.NewWriter(netConn),
		br:           bufio.NewReader(netConn),
		readTimeout:  do.readTimeout,
		writeTimeout: do.writeTimeout,
	}

	if do.password != "" {
		if _, err := c.Do("AUTH", do.password); err != nil {
			netConn.Close()
			return nil, err
		}
	}

	if do.db != 0 {
		if _, err := c.Do("SELECT", do.db); err != nil {
			netConn.Close()
			return nil, err
		}
	}

	return c, nil
}

var pathDBRegexp = regexp.MustCompile(`/(\d*)\z`)

// DialURL connects to a Redis server at the given URL using the Redis
// URI scheme. URLs should follow the draft IANA specification for the
// scheme (https://www.iana.org/assignments/uri-schemes/prov/redis).
func DialURL(rawurl string, options ...DialOption) (Conn, error) {
	u, err := url.Parse(rawurl)
	if err != nil {
		return nil, err
	}

	if u.Scheme != "redis" {
		return nil, fmt.Errorf("invalid redis URL scheme: %s", u.Scheme)
	}

	// As per the IANA draft spec, the host defaults to localhost and
	// the port defaults to 6379.
	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		// assume port is missing
		host = u.Host
		port = "6379"
	}
	if host == "" {
		host = "localhost"
	}
	address := net.JoinHostPort(host, port)

	if u.User != nil {
		password, isSet := u.User.Password()
		if isSet {
			options = append(options, DialPassword(password))
		}
	}

	match := pathDBRegexp.FindStringSubmatch(u.Path)
	if len(match) == 2 {
		db := 0
		if len(match[1]) > 0 {
			db, err = strconv.Atoi(match[1])
			if err != nil {
				return nil, fmt.Errorf("invalid database: %s", u.Path[1:])
			}
		}
		if db != 0 {
			options = append(options, DialDatabase(db))
		}
	} else if u.Path != "" {
		return nil, fmt.Errorf("invalid database: %s", u.Path[1:])
	}

	return Dial("tcp", address, options...)
}

// NewConn returns a new Redigo connection for the given net connection.
func NewConn(netConn net.Conn, readTimeout, writeTimeout time.Duration) Conn {
	return &conn{
		conn:         netConn,
		bw:           bufio.NewWriter(netConn),
		br:           bufio.NewReader(netConn),
		readTimeout:  readTimeout,
		writeTimeout: writeTimeout,
	}
}

func (c *conn) Close() error {
	c.mu.Lock()
	err := c.err
	if c.err == nil {
		c.err = errors.New("redigo: closed")
		err = c.conn.Close()
	}
	c.mu.Unlock()
	return err
}

func (c *conn) fatal(err error) error {
	c.mu.Lock()
	if c.err == nil {
		c.err = err
		// Close connection to force errors on subsequent calls and to unblock
		// other reader or writer.
		c.conn.Close()
	}
	c.mu.Unlock()
	return err
}

func (c *conn) Err() error {
	c.mu.Lock()
	err := c.err
	c.mu.Unlock()
	return err
}

func (c *conn) writeLen(prefix byte, n int) error {
	c.lenScratch[len(c.lenScratch)-1] = '\n'
	c.lenScratch[len(c.lenScratch)-2] = '\r'
	i := len(c.lenScratch) - 3
	for {
		c.lenScratch[i] = byte('0' + n%10)
		i -= 1
		n = n / 10
		if n == 0 {
			break
		}
	}
	c.lenScratch[i] = prefix
	_, err := c.bw.Write(c.lenScratch[i:])
	return err
}

func (c *conn) writeString(s string) error {
	c.writeLen('$', len(s))
	c.bw.WriteString(s)
	_, err := c.bw.WriteString("\r\n")
	return err
}

func (c *conn) writeBytes(p []byte) error {
	c.writeLen('$', len(p))
	c.bw.Write(p)
	_, err := c.bw.WriteString("\r\n")
	return err
}

func (c *conn) writeInt64(n int64) error {
	return c.writeBytes(strconv.AppendInt(c.numScratch[:0], n, 10))
}

func (c *conn) writeFloat64(n float64) error {
	return c.writeBytes(strconv.AppendFloat(c.numScratch[:0], n, 'g', -1, 64))
}

func (c *conn) writeCommand(cmd string, args []interface{}) (err error) {
	c.writeLen('*', 1+len(args))
	err = c.writeString(cmd)
	for _, arg := range args {
		if err != nil {
			break
		}
		switch arg := arg.(type) {
		case string:
			err = c.writeString(arg)
		case []byte:
			err = c.writeBytes(arg)
		case int:
			err = c.writeInt64(int64(arg))
		case int64:
			err = c.writeInt64(arg)
		case float64:
			err = c.writeFloat64(arg)
		case bool:
			if arg {
				err = c.writeString("1")
			} else {
				err = c.writeString("0")
			}
		case nil:
			err = c.writeString("")
		default:
			var buf bytes.Buffer
			fmt.Fprint(&buf, arg)
			err = c.writeBytes(buf.Bytes())
		}
	}
	return err
}

type protocolError string

func (pe protocolError) Error() string {
	return fmt.Sprintf("redigo: %s (possible server error or unsupported concurrent read by application)", string(pe))
}

func (c *conn) readLine() ([]byte, error) {
	p, err := c.br.ReadSlice('\n')
	if err == bufio.ErrBufferFull {
		return nil, protocolError("long response line")
	}
	if err != nil {
		return nil, err
	}
	i := len(p) - 2
	if i < 0 || p[i] != '\r' {
		return nil, protocolError("bad response line terminator")
	}
	return p[:i], nil
}

// parseLen parses bulk string and array lengths.
func parseLen(p []byte) (int, error) {
	if len(p) == 0 {
		return -1, protocolError("malformed length")
	}

	if p[0] == '-' && len(p) == 2 && p[1] == '1' {
		// handle $-1 and $-1 null replies.
		return -1, nil
	}

	var n int
	for _, b := range p {
		n *= 10
		if b < '0' || b > '9' {
			return -1, protocolError("illegal bytes in length")
		}
		n += int(b - '0')
	}

	return n, nil
}

// parseInt parses an integer reply.
func parseInt(p []byte) (interface{}, error) {
	if len(p) == 0 {
		return 0, protocolError("malformed integer")
	}

	var negate bool
	if p[0] == '-' {
		negate = true
		p = p[1:]
		if len(p) == 0 {
			return 0, protocolError("malformed integer")
		}
	}

	var n int64
	for _, b := range p {
		n *= 10
		if b < '0' || b > '9' {
			return 0, protocolError("illegal bytes in length")
		}
		n += int64(b - '0')
	}

	if negate {
		n = -n
	}
	return n, nil
}

var (
	okReply   interface{} = "OK"
	pongReply interface{} = "PONG"
)

func (c *conn) readReply() (interface{}, error) {
	line, err := c.readLine()
	if err != nil {
		return nil, err
	}
	if len(line) == 0 {
		return nil, protocolError("short response line")
	}
	switch line[0] {
	case '+':
		switch {
		case len(line) == 3 && line[1] == 'O' && line[2] == 'K':
			// Avoid allocation for frequent "+OK" response.
			return okReply, nil
		case len(line) == 5 && line[1] == 'P' && line[2] == 'O' && line[3] == 'N' && line[4] == 'G':
			// Avoid allocation in PING command benchmarks :)
			return pongReply, nil
		default:
			return string(line[1:]), nil
		}
	case '-':
		return Error(string(line[1:])), nil
	case ':':
		return parseInt(line[1:])
	case '$':
		n, err := parseLen(line[1:])
		if n < 0 || err != nil {
			return nil, err
		}
		p := make([]byte, n)
		_, err = io.ReadFull(c.br, p)
		if err != nil {
			return nil, err
		}
		if line, err := c.readLine(); err != nil {
			return nil, err
		} else if len(line) != 0 {
			return nil, protocolError("bad bulk string format")
		}
		return p, nil
	case '*':
		n, err := parseLen(line[1:])
		if n < 0 || err != nil {
			return nil, err
		}
		r := make([]interface{}, n)
		for i := range r {
			r[i], err = c.readReply()
			if err != nil {
				return nil, err
			}
		}
		return r, nil
	}
	return nil, protocolError("unexpected response line")
}

func (c *conn) Send(cmd string, args ...interface{}) error {
	c.mu.Lock()
	c.pending += 1
	c.mu.Unlock()
	if c.writeTimeout != 0 {
		c.conn.SetWriteDeadline(time.Now().Add(c.writeTimeout))
	}
	if err := c.writeCommand(cmd, args); err != nil {
		return c.fatal(err)
	}
	return nil
}

func (c *conn) Flush() error {
	if c.writeTimeout != 0 {
		c.conn.SetWriteDeadline(time.Now().Add(c.writeTimeout))
	}
	if err := c.bw.Flush(); err != nil {
		return c.fatal(err)
	}
	return nil
}

func (c *conn) Receive() (reply interface{}, err error) {
	if c.readTimeout != 0 {
		c.conn.SetReadDeadline(time.Now().Add(c.readTimeout))
	}
	if reply, err = c.readReply(); err != nil {
		return nil, c.fatal(err)
	}
	// When using pub/sub, the number of receives can be greater than the
	// number of sends. To enable normal use of the connection after
	// unsubscribing from all channels, we do not decrement pending to a
	// negative value.
	//
	// The pending field is decremented after the reply is read to handle the
	// case where Receive is called before Send.
	c.mu.Lock()
	if c.pending > 0 {
		c.pending -= 1
	}
	c.mu.Unlock()
	if err, ok := reply.(Error); ok {
		return nil, err
	}
	return
}

func (c *conn) Do(cmd string, args ...interface{}) (interface{}, error) {
	c.mu.Lock()
	pending := c.pending
	c.pending = 0
	c.mu.Unlock()

	if cmd == "" && pending == 0 {
		return nil, nil
	}

	if c.writeTimeout != 0 {
		c.conn.SetWriteDeadline(time.Now().Add(c.writeTimeout))
	}

	if cmd != "" {
		if err := c.writeCommand(cmd, args); err != nil {
			return nil, c.fatal(err)
		}
	}

	if err := c.bw.Flush(); err != nil {
		return nil, c.fatal(err)
	}

	if c.readTimeout != 0 {
		c.conn.SetReadDeadline(time.Now().Add(c.readTimeout))
	}

	if cmd == "" {
		reply := make([]interface{}, pending)
		for i := range reply {
			r, e := c.readReply()
			if e != nil {
				return nil, c.fatal(e)
			}
			reply[i] = r
		}
		return reply, nil
	}

	var err error
	var reply interface{}
	for i := 0; i <= pending; i++ {
		var e error
		if reply, e = c.readReply(); e != nil {
			return nil, c.fatal(e)
		}
		if e, ok := reply.(Error); ok && err == nil {
			err = e
		}
	}
	return reply, err
}
