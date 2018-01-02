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

package redis_test

import (
	"bytes"
	"io"
	"math"
	"net"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/garyburd/redigo/redis"
)

type testConn struct {
	io.Reader
	io.Writer
}

func (*testConn) Close() error                       { return nil }
func (*testConn) LocalAddr() net.Addr                { return nil }
func (*testConn) RemoteAddr() net.Addr               { return nil }
func (*testConn) SetDeadline(t time.Time) error      { return nil }
func (*testConn) SetReadDeadline(t time.Time) error  { return nil }
func (*testConn) SetWriteDeadline(t time.Time) error { return nil }

func dialTestConn(r io.Reader, w io.Writer) redis.DialOption {
	return redis.DialNetDial(func(net, addr string) (net.Conn, error) {
		return &testConn{Reader: r, Writer: w}, nil
	})
}

var writeTests = []struct {
	args     []interface{}
	expected string
}{
	{
		[]interface{}{"SET", "key", "value"},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n",
	},
	{
		[]interface{}{"SET", "key", "value"},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n",
	},
	{
		[]interface{}{"SET", "key", byte(100)},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$3\r\n100\r\n",
	},
	{
		[]interface{}{"SET", "key", 100},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$3\r\n100\r\n",
	},
	{
		[]interface{}{"SET", "key", int64(math.MinInt64)},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$20\r\n-9223372036854775808\r\n",
	},
	{
		[]interface{}{"SET", "key", float64(1349673917.939762)},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$21\r\n1.349673917939762e+09\r\n",
	},
	{
		[]interface{}{"SET", "key", ""},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$0\r\n\r\n",
	},
	{
		[]interface{}{"SET", "key", nil},
		"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$0\r\n\r\n",
	},
	{
		[]interface{}{"ECHO", true, false},
		"*3\r\n$4\r\nECHO\r\n$1\r\n1\r\n$1\r\n0\r\n",
	},
}

func TestWrite(t *testing.T) {
	for _, tt := range writeTests {
		var buf bytes.Buffer
		c, _ := redis.Dial("", "", dialTestConn(nil, &buf))
		err := c.Send(tt.args[0].(string), tt.args[1:]...)
		if err != nil {
			t.Errorf("Send(%v) returned error %v", tt.args, err)
			continue
		}
		c.Flush()
		actual := buf.String()
		if actual != tt.expected {
			t.Errorf("Send(%v) = %q, want %q", tt.args, actual, tt.expected)
		}
	}
}

var errorSentinel = &struct{}{}

var readTests = []struct {
	reply    string
	expected interface{}
}{
	{
		"+OK\r\n",
		"OK",
	},
	{
		"+PONG\r\n",
		"PONG",
	},
	{
		"@OK\r\n",
		errorSentinel,
	},
	{
		"$6\r\nfoobar\r\n",
		[]byte("foobar"),
	},
	{
		"$-1\r\n",
		nil,
	},
	{
		":1\r\n",
		int64(1),
	},
	{
		":-2\r\n",
		int64(-2),
	},
	{
		"*0\r\n",
		[]interface{}{},
	},
	{
		"*-1\r\n",
		nil,
	},
	{
		"*4\r\n$3\r\nfoo\r\n$3\r\nbar\r\n$5\r\nHello\r\n$5\r\nWorld\r\n",
		[]interface{}{[]byte("foo"), []byte("bar"), []byte("Hello"), []byte("World")},
	},
	{
		"*3\r\n$3\r\nfoo\r\n$-1\r\n$3\r\nbar\r\n",
		[]interface{}{[]byte("foo"), nil, []byte("bar")},
	},

	{
		// "x" is not a valid length
		"$x\r\nfoobar\r\n",
		errorSentinel,
	},
	{
		// -2 is not a valid length
		"$-2\r\n",
		errorSentinel,
	},
	{
		// "x"  is not a valid integer
		":x\r\n",
		errorSentinel,
	},
	{
		// missing \r\n following value
		"$6\r\nfoobar",
		errorSentinel,
	},
	{
		// short value
		"$6\r\nxx",
		errorSentinel,
	},
	{
		// long value
		"$6\r\nfoobarx\r\n",
		errorSentinel,
	},
}

func TestRead(t *testing.T) {
	for _, tt := range readTests {
		c, _ := redis.Dial("", "", dialTestConn(strings.NewReader(tt.reply), nil))
		actual, err := c.Receive()
		if tt.expected == errorSentinel {
			if err == nil {
				t.Errorf("Receive(%q) did not return expected error", tt.reply)
			}
		} else {
			if err != nil {
				t.Errorf("Receive(%q) returned error %v", tt.reply, err)
				continue
			}
			if !reflect.DeepEqual(actual, tt.expected) {
				t.Errorf("Receive(%q) = %v, want %v", tt.reply, actual, tt.expected)
			}
		}
	}
}

var testCommands = []struct {
	args     []interface{}
	expected interface{}
}{
	{
		[]interface{}{"PING"},
		"PONG",
	},
	{
		[]interface{}{"SET", "foo", "bar"},
		"OK",
	},
	{
		[]interface{}{"GET", "foo"},
		[]byte("bar"),
	},
	{
		[]interface{}{"GET", "nokey"},
		nil,
	},
	{
		[]interface{}{"MGET", "nokey", "foo"},
		[]interface{}{nil, []byte("bar")},
	},
	{
		[]interface{}{"INCR", "mycounter"},
		int64(1),
	},
	{
		[]interface{}{"LPUSH", "mylist", "foo"},
		int64(1),
	},
	{
		[]interface{}{"LPUSH", "mylist", "bar"},
		int64(2),
	},
	{
		[]interface{}{"LRANGE", "mylist", 0, -1},
		[]interface{}{[]byte("bar"), []byte("foo")},
	},
	{
		[]interface{}{"MULTI"},
		"OK",
	},
	{
		[]interface{}{"LRANGE", "mylist", 0, -1},
		"QUEUED",
	},
	{
		[]interface{}{"PING"},
		"QUEUED",
	},
	{
		[]interface{}{"EXEC"},
		[]interface{}{
			[]interface{}{[]byte("bar"), []byte("foo")},
			"PONG",
		},
	},
}

func TestDoCommands(t *testing.T) {
	c, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	for _, cmd := range testCommands {
		actual, err := c.Do(cmd.args[0].(string), cmd.args[1:]...)
		if err != nil {
			t.Errorf("Do(%v) returned error %v", cmd.args, err)
			continue
		}
		if !reflect.DeepEqual(actual, cmd.expected) {
			t.Errorf("Do(%v) = %v, want %v", cmd.args, actual, cmd.expected)
		}
	}
}

func TestPipelineCommands(t *testing.T) {
	c, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	for _, cmd := range testCommands {
		if err := c.Send(cmd.args[0].(string), cmd.args[1:]...); err != nil {
			t.Fatalf("Send(%v) returned error %v", cmd.args, err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Errorf("Flush() returned error %v", err)
	}
	for _, cmd := range testCommands {
		actual, err := c.Receive()
		if err != nil {
			t.Fatalf("Receive(%v) returned error %v", cmd.args, err)
		}
		if !reflect.DeepEqual(actual, cmd.expected) {
			t.Errorf("Receive(%v) = %v, want %v", cmd.args, actual, cmd.expected)
		}
	}
}

func TestBlankCommmand(t *testing.T) {
	c, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	for _, cmd := range testCommands {
		if err := c.Send(cmd.args[0].(string), cmd.args[1:]...); err != nil {
			t.Fatalf("Send(%v) returned error %v", cmd.args, err)
		}
	}
	reply, err := redis.Values(c.Do(""))
	if err != nil {
		t.Fatalf("Do() returned error %v", err)
	}
	if len(reply) != len(testCommands) {
		t.Fatalf("len(reply)=%d, want %d", len(reply), len(testCommands))
	}
	for i, cmd := range testCommands {
		actual := reply[i]
		if !reflect.DeepEqual(actual, cmd.expected) {
			t.Errorf("Receive(%v) = %v, want %v", cmd.args, actual, cmd.expected)
		}
	}
}

func TestRecvBeforeSend(t *testing.T) {
	c, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()
	done := make(chan struct{})
	go func() {
		c.Receive()
		close(done)
	}()
	time.Sleep(time.Millisecond)
	c.Send("PING")
	c.Flush()
	<-done
	_, err = c.Do("")
	if err != nil {
		t.Fatalf("error=%v", err)
	}
}

func TestError(t *testing.T) {
	c, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	c.Do("SET", "key", "val")
	_, err = c.Do("HSET", "key", "fld", "val")
	if err == nil {
		t.Errorf("Expected err for HSET on string key.")
	}
	if c.Err() != nil {
		t.Errorf("Conn has Err()=%v, expect nil", c.Err())
	}
	_, err = c.Do("SET", "key", "val")
	if err != nil {
		t.Errorf("Do(SET, key, val) returned error %v, expected nil.", err)
	}
}

func TestReadTimeout(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("net.Listen returned %v", err)
	}
	defer l.Close()

	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			go func() {
				time.Sleep(time.Second)
				c.Write([]byte("+OK\r\n"))
				c.Close()
			}()
		}
	}()

	// Do

	c1, err := redis.Dial(l.Addr().Network(), l.Addr().String(), redis.DialReadTimeout(time.Millisecond))
	if err != nil {
		t.Fatalf("redis.Dial returned %v", err)
	}
	defer c1.Close()

	_, err = c1.Do("PING")
	if err == nil {
		t.Fatalf("c1.Do() returned nil, expect error")
	}
	if c1.Err() == nil {
		t.Fatalf("c1.Err() = nil, expect error")
	}

	// Send/Flush/Receive

	c2, err := redis.Dial(l.Addr().Network(), l.Addr().String(), redis.DialReadTimeout(time.Millisecond))
	if err != nil {
		t.Fatalf("redis.Dial returned %v", err)
	}
	defer c2.Close()

	c2.Send("PING")
	c2.Flush()
	_, err = c2.Receive()
	if err == nil {
		t.Fatalf("c2.Receive() returned nil, expect error")
	}
	if c2.Err() == nil {
		t.Fatalf("c2.Err() = nil, expect error")
	}
}

var dialErrors = []struct {
	rawurl        string
	expectedError string
}{
	{
		"localhost",
		"invalid redis URL scheme",
	},
	// The error message for invalid hosts is diffferent in different
	// versions of Go, so just check that there is an error message.
	{
		"redis://weird url",
		"",
	},
	{
		"redis://foo:bar:baz",
		"",
	},
	{
		"http://www.google.com",
		"invalid redis URL scheme: http",
	},
	{
		"redis://localhost:6379/abc123",
		"invalid database: abc123",
	},
}

func TestDialURLErrors(t *testing.T) {
	for _, d := range dialErrors {
		_, err := redis.DialURL(d.rawurl)
		if err == nil || !strings.Contains(err.Error(), d.expectedError) {
			t.Errorf("DialURL did not return expected error (expected %v to contain %s)", err, d.expectedError)
		}
	}
}

func TestDialURLPort(t *testing.T) {
	checkPort := func(network, address string) (net.Conn, error) {
		if address != "localhost:6379" {
			t.Errorf("DialURL did not set port to 6379 by default (got %v)", address)
		}
		return nil, nil
	}
	_, err := redis.DialURL("redis://localhost", redis.DialNetDial(checkPort))
	if err != nil {
		t.Error("dial error:", err)
	}
}

func TestDialURLHost(t *testing.T) {
	checkHost := func(network, address string) (net.Conn, error) {
		if address != "localhost:6379" {
			t.Errorf("DialURL did not set host to localhost by default (got %v)", address)
		}
		return nil, nil
	}
	_, err := redis.DialURL("redis://:6379", redis.DialNetDial(checkHost))
	if err != nil {
		t.Error("dial error:", err)
	}
}

func TestDialURLPassword(t *testing.T) {
	var buf bytes.Buffer
	_, err := redis.DialURL("redis://x:abc123@localhost", dialTestConn(strings.NewReader("+OK\r\n"), &buf))
	if err != nil {
		t.Error("dial error:", err)
	}
	expected := "*2\r\n$4\r\nAUTH\r\n$6\r\nabc123\r\n"
	actual := buf.String()
	if actual != expected {
		t.Errorf("commands = %q, want %q", actual, expected)
	}
}

func TestDialURLDatabase(t *testing.T) {
	var buf3 bytes.Buffer
	_, err3 := redis.DialURL("redis://localhost/3", dialTestConn(strings.NewReader("+OK\r\n"), &buf3))
	if err3 != nil {
		t.Error("dial error:", err3)
	}
	expected3 := "*2\r\n$6\r\nSELECT\r\n$1\r\n3\r\n"
	actual3 := buf3.String()
	if actual3 != expected3 {
		t.Errorf("commands = %q, want %q", actual3, expected3)
	}
	// empty DB means 0
	var buf0 bytes.Buffer
	_, err0 := redis.DialURL("redis://localhost/", dialTestConn(strings.NewReader("+OK\r\n"), &buf0))
	if err0 != nil {
		t.Error("dial error:", err0)
	}
	expected0 := ""
	actual0 := buf0.String()
	if actual0 != expected0 {
		t.Errorf("commands = %q, want %q", actual0, expected0)
	}
}

// Connect to local instance of Redis running on the default port.
func ExampleDial() {
	c, err := redis.Dial("tcp", ":6379")
	if err != nil {
		// handle error
	}
	defer c.Close()
}

// Connect to remote instance of Redis using a URL.
func ExampleDialURL() {
	c, err := redis.DialURL(os.Getenv("REDIS_URL"))
	if err != nil {
		// handle connection error
	}
	defer c.Close()
}

// TextExecError tests handling of errors in a transaction. See
// http://redis.io/topics/transactions for information on how Redis handles
// errors in a transaction.
func TestExecError(t *testing.T) {
	c, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	// Execute commands that fail before EXEC is called.

	c.Do("DEL", "k0")
	c.Do("ZADD", "k0", 0, 0)
	c.Send("MULTI")
	c.Send("NOTACOMMAND", "k0", 0, 0)
	c.Send("ZINCRBY", "k0", 0, 0)
	v, err := c.Do("EXEC")
	if err == nil {
		t.Fatalf("EXEC returned values %v, expected error", v)
	}

	// Execute commands that fail after EXEC is called. The first command
	// returns an error.

	c.Do("DEL", "k1")
	c.Do("ZADD", "k1", 0, 0)
	c.Send("MULTI")
	c.Send("HSET", "k1", 0, 0)
	c.Send("ZINCRBY", "k1", 0, 0)
	v, err = c.Do("EXEC")
	if err != nil {
		t.Fatalf("EXEC returned error %v", err)
	}

	vs, err := redis.Values(v, nil)
	if err != nil {
		t.Fatalf("Values(v) returned error %v", err)
	}

	if len(vs) != 2 {
		t.Fatalf("len(vs) == %d, want 2", len(vs))
	}

	if _, ok := vs[0].(error); !ok {
		t.Fatalf("first result is type %T, expected error", vs[0])
	}

	if _, ok := vs[1].([]byte); !ok {
		t.Fatalf("second result is type %T, expected []byte", vs[1])
	}

	// Execute commands that fail after EXEC is called. The second command
	// returns an error.

	c.Do("ZADD", "k2", 0, 0)
	c.Send("MULTI")
	c.Send("ZINCRBY", "k2", 0, 0)
	c.Send("HSET", "k2", 0, 0)
	v, err = c.Do("EXEC")
	if err != nil {
		t.Fatalf("EXEC returned error %v", err)
	}

	vs, err = redis.Values(v, nil)
	if err != nil {
		t.Fatalf("Values(v) returned error %v", err)
	}

	if len(vs) != 2 {
		t.Fatalf("len(vs) == %d, want 2", len(vs))
	}

	if _, ok := vs[0].([]byte); !ok {
		t.Fatalf("first result is type %T, expected []byte", vs[0])
	}

	if _, ok := vs[1].(error); !ok {
		t.Fatalf("second result is type %T, expected error", vs[2])
	}
}

func BenchmarkDoEmpty(b *testing.B) {
	b.StopTimer()
	c, err := redis.DialDefaultServer()
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Do(""); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDoPing(b *testing.B) {
	b.StopTimer()
	c, err := redis.DialDefaultServer()
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Do("PING"); err != nil {
			b.Fatal(err)
		}
	}
}
