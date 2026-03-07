// Package nltest provides utilities for netlink testing.
package nltest

import (
	"fmt"
	"io"
	"os"

	"github.com/mdlayher/netlink"
	"github.com/mdlayher/netlink/nlenc"
)

// PID is the netlink header PID value assigned by nltest.
const PID = 1

// MustMarshalAttributes marshals a slice of netlink.Attributes to their binary
// format, but panics if any errors occur.
func MustMarshalAttributes(attrs []netlink.Attribute) []byte {
	b, err := netlink.MarshalAttributes(attrs)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal attributes to binary: %v", err))
	}

	return b
}

// Multipart sends a slice of netlink.Messages to the caller as a
// netlink multi-part message. If less than two messages are present,
// the messages are not altered.
func Multipart(msgs []netlink.Message) ([]netlink.Message, error) {
	if len(msgs) < 2 {
		return msgs, nil
	}

	for i := range msgs {
		// Last message has header type "done" in addition to multi-part flag.
		if i == len(msgs)-1 {
			msgs[i].Header.Type = netlink.Done
		}

		msgs[i].Header.Flags |= netlink.Multi
	}

	return msgs, nil
}

// Error returns a netlink error to the caller with the specified error
// number, in the body of the specified request message.
func Error(number int, reqs []netlink.Message) ([]netlink.Message, error) {
	req := reqs[0]
	req.Header.Length += 4
	req.Header.Type = netlink.Error

	errno := -1 * int32(number)
	req.Data = append(nlenc.Int32Bytes(errno), req.Data...)

	return []netlink.Message{req}, nil
}

// A Func is a function that can be used to test netlink.Conn interactions.
// The function can choose to return zero or more netlink messages, or an
// error if needed.
//
// For a netlink request/response interaction, a request req is populated by
// netlink.Conn.Send and passed to the function.
//
// For multicast interactions, an empty request req is passed to the function
// when netlink.Conn.Receive is called.
//
// If a Func returns an error, the error will be returned as-is to the caller.
// If no messages and io.EOF are returned, no messages and no error will be
// returned to the caller, simulating a multi-part message with no data.
type Func func(req []netlink.Message) ([]netlink.Message, error)

// Dial sets up a netlink.Conn for testing using the specified Func. All requests
// sent from the connection will be passed to the Func.  The connection should be
// closed as usual when it is no longer needed.
func Dial(fn Func) *netlink.Conn {
	sock := &socket{
		fn: fn,
	}

	return netlink.NewConn(sock, PID)
}

// CheckRequest returns a Func that verifies that each message in an incoming
// request has the specified netlink header type and flags in the same slice
// position index, and then passes the request through to fn.
//
// The length of the types and flags slices must match the number of requests
// passed to the returned Func, or CheckRequest will panic.
//
// As an example:
//   - types[0] and flags[0] will be checked against reqs[0]
//   - types[1] and flags[1] will be checked against reqs[1]
//   - ... and so on
//
// If an element of types or flags is set to the zero value, that check will
// be skipped for the request message that occurs at the same index.
//
// As an example, if types[0] is 0 and reqs[0].Header.Type is 1, the check will
// succeed because types[0] was not specified.
func CheckRequest(types []netlink.HeaderType, flags []netlink.HeaderFlags, fn Func) Func {
	if len(types) != len(flags) {
		panicf("nltest: CheckRequest called with mismatched types and flags slice lengths: %d != %d",
			len(types), len(flags))
	}

	return func(req []netlink.Message) ([]netlink.Message, error) {
		if len(types) != len(req) {
			panicf("nltest: CheckRequest function invoked types/flags and request message slice lengths: %d != %d",
				len(types), len(req))
		}

		for i := range req {
			if want, got := types[i], req[i].Header.Type; types[i] != 0 && want != got {
				return nil, fmt.Errorf("nltest: unexpected netlink header type: %s, want: %s", got, want)
			}

			if want, got := flags[i], req[i].Header.Flags; flags[i] != 0 && want != got {
				return nil, fmt.Errorf("nltest: unexpected netlink header flags: %s, want: %s", got, want)
			}
		}

		return fn(req)
	}
}

// A socket is a netlink.Socket used for testing.
type socket struct {
	fn Func

	msgs []netlink.Message
	err  error
}

func (c *socket) Close() error { return nil }

func (c *socket) SendMessages(messages []netlink.Message) error {
	msgs, err := c.fn(messages)
	c.msgs = append(c.msgs, msgs...)
	c.err = err
	return nil
}

func (c *socket) Send(m netlink.Message) error {
	c.msgs, c.err = c.fn([]netlink.Message{m})
	return nil
}

func (c *socket) Receive() ([]netlink.Message, error) {
	// No messages set by Send means that we are emulating a
	// multicast response or an error occurred.
	if len(c.msgs) == 0 {
		switch c.err {
		case nil:
			// No error, simulate multicast, but also return EOF to simulate
			// no replies if needed.
			msgs, err := c.fn(nil)
			if err == io.EOF {
				err = nil
			}

			return msgs, err
		case io.EOF:
			// EOF, simulate no replies in multi-part message.
			return nil, nil
		}

		// If the error is a system call error, wrap it in os.NewSyscallError
		// to simulate what the Linux netlink.Conn does.
		if isSyscallError(c.err) {
			return nil, os.NewSyscallError("recvmsg", c.err)
		}

		// Some generic error occurred and should be passed to the caller.
		return nil, c.err
	}

	// Detect multi-part messages.
	var multi bool
	for _, m := range c.msgs {
		if m.Header.Flags&netlink.Multi != 0 && m.Header.Type != netlink.Done {
			multi = true
		}
	}

	// When a multi-part message is detected, return all messages except for the
	// final "multi-part done", so that a second call to Receive from netlink.Conn
	// will drain that message.
	if multi {
		last := c.msgs[len(c.msgs)-1]
		ret := c.msgs[:len(c.msgs)-1]
		c.msgs = []netlink.Message{last}

		return ret, c.err
	}

	msgs, err := c.msgs, c.err
	c.msgs, c.err = nil, nil

	return msgs, err
}

func panicf(format string, a ...interface{}) {
	panic(fmt.Sprintf(format, a...))
}
