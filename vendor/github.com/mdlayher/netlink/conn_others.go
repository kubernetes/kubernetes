//go:build !linux
// +build !linux

package netlink

import (
	"fmt"
	"runtime"
)

// errUnimplemented is returned by all functions on platforms that
// cannot make use of netlink sockets.
var errUnimplemented = fmt.Errorf("netlink: not implemented on %s/%s",
	runtime.GOOS, runtime.GOARCH)

var _ Socket = &conn{}

// A conn is the no-op implementation of a netlink sockets connection.
type conn struct{}

// All cross-platform functions and Socket methods are unimplemented outside
// of Linux.

func dial(_ int, _ *Config) (*conn, uint32, error) { return nil, 0, errUnimplemented }
func newError(_ int) error                         { return errUnimplemented }

func (c *conn) Send(_ Message) error           { return errUnimplemented }
func (c *conn) SendMessages(_ []Message) error { return errUnimplemented }
func (c *conn) Receive() ([]Message, error)    { return nil, errUnimplemented }
func (c *conn) Close() error                   { return errUnimplemented }
