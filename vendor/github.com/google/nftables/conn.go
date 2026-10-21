// Copyright 2018 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nftables

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"syscall"

	"github.com/google/nftables/binaryutil"
	"github.com/google/nftables/expr"
	"github.com/mdlayher/netlink"
	"github.com/mdlayher/netlink/nltest"
	"golang.org/x/sys/unix"
)

// A Conn represents a netlink connection of the nftables family.
//
// All methods return their input, so that variables can be defined from string
// literals when desired.
//
// Commands are buffered. Flush sends all buffered commands in a single batch.
type Conn struct {
	TestDial nltest.Func // for testing only; passed to nltest.Dial
	NetNS    int         // fd referencing the network namespace netlink will interact with.

	lasting     bool       // establish a lasting connection to be used across multiple netlink operations.
	mu          sync.Mutex // protects the following state
	messages    []netlink.Message
	err         error
	nlconn      *netlink.Conn // netlink socket using NETLINK_NETFILTER protocol.
	sockOptions []SockOption
}

// ConnOption is an option to change the behavior of the nftables Conn returned by Open.
type ConnOption func(*Conn)

// SockOption is an option to change the behavior of the netlink socket used by the nftables Conn.
type SockOption func(*netlink.Conn) error

// New returns a netlink connection for querying and modifying nftables. Some
// aspects of the new netlink connection can be configured using the options
// WithNetNSFd, WithTestDial, and AsLasting.
//
// A lasting netlink connection should be closed by calling CloseLasting() to
// close the underlying lasting netlink connection, cancelling all pending
// operations using this connection.
func New(opts ...ConnOption) (*Conn, error) {
	cc := &Conn{}
	for _, opt := range opts {
		opt(cc)
	}

	if !cc.lasting {
		return cc, nil
	}

	nlconn, err := cc.dialNetlink()
	if err != nil {
		return nil, err
	}
	cc.nlconn = nlconn
	return cc, nil
}

// AsLasting creates the new netlink connection as a lasting connection that is
// reused across multiple netlink operations, instead of opening and closing the
// underlying netlink connection only for the duration of a single netlink
// operation.
func AsLasting() ConnOption {
	return func(cc *Conn) {
		// We cannot create the underlying connection yet, as we are called
		// anywhere in the option processing chain and there might be later
		// options still modifying connection behavior.
		cc.lasting = true
	}
}

// WithNetNSFd sets the network namespace to create a new netlink connection to:
// the fd must reference a network namespace.
func WithNetNSFd(fd int) ConnOption {
	return func(cc *Conn) {
		cc.NetNS = fd
	}
}

// WithTestDial sets the specified nltest.Func when creating a new netlink
// connection.
func WithTestDial(f nltest.Func) ConnOption {
	return func(cc *Conn) {
		cc.TestDial = f
	}
}

// WithSockOptions sets the specified socket options when creating a new netlink
// connection.
func WithSockOptions(opts ...SockOption) ConnOption {
	return func(cc *Conn) {
		cc.sockOptions = append(cc.sockOptions, opts...)
	}
}

// netlinkCloser is returned by netlinkConn(UnderLock) and must be called after
// being done with the returned netlink connection in order to properly close
// this connection, if necessary.
type netlinkCloser func() error

// netlinkConn returns a netlink connection together with a netlinkCloser that
// later must be called by the caller when it doesn't need the returned netlink
// connection anymore. The netlinkCloser will close the netlink connection when
// necessary. If New has been told to create a lasting connection, then this
// lasting netlink connection will be returned, otherwise a new "transient"
// netlink connection will be opened and returned instead. netlinkConn must not
// be called while the Conn.mu lock is currently helt (this will cause a
// deadlock). Use netlinkConnUnderLock instead in such situations.
func (cc *Conn) netlinkConn() (*netlink.Conn, netlinkCloser, error) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.netlinkConnUnderLock()
}

// netlinkConnUnderLock works like netlinkConn but must be called while holding
// the Conn.mu lock.
func (cc *Conn) netlinkConnUnderLock() (*netlink.Conn, netlinkCloser, error) {
	if cc.nlconn != nil {
		return cc.nlconn, func() error { return nil }, nil
	}
	nlconn, err := cc.dialNetlink()
	if err != nil {
		return nil, nil, err
	}
	return nlconn, func() error { return nlconn.Close() }, nil
}

func receiveAckAware(nlconn *netlink.Conn, sentMsgFlags netlink.HeaderFlags) ([]netlink.Message, error) {
	if nlconn == nil {
		return nil, errors.New("netlink conn is not initialized")
	}

	// first receive will be the message that we expect
	reply, err := nlconn.Receive()
	if err != nil {
		return nil, err
	}

	if (sentMsgFlags & netlink.Acknowledge) == 0 {
		// we did not request an ack
		return reply, nil
	}

	if (sentMsgFlags & netlink.Dump) == netlink.Dump {
		// sent message has Dump flag set, there will be no acks
		// https://github.com/torvalds/linux/blob/7e062cda7d90543ac8c7700fc7c5527d0c0f22ad/net/netlink/af_netlink.c#L2387-L2390
		return reply, nil
	}

	if len(reply) != 0 {
		last := reply[len(reply)-1]
		for re := last.Header.Type; (re&netlink.Overrun) == netlink.Overrun && (re&netlink.Done) != netlink.Done; re = last.Header.Type {
			// we are not finished, the message is overrun
			r, err := nlconn.Receive()
			if err != nil {
				return nil, err
			}
			reply = append(reply, r...)
			last = reply[len(reply)-1]
		}

		if last.Header.Type == netlink.Error && binaryutil.BigEndian.Uint32(last.Data[:4]) == 0 {
			// we have already collected an ack
			return reply, nil
		}
	}

	// Now we expect an ack
	ack, err := nlconn.Receive()
	if err != nil {
		return nil, err
	}

	if len(ack) == 0 {
		// received an empty ack?
		return reply, nil
	}

	msg := ack[0]
	if msg.Header.Type != netlink.Error {
		// acks should be delivered as NLMSG_ERROR
		return nil, fmt.Errorf("expected header %v, but got %v", netlink.Error, msg.Header.Type)
	}

	if binaryutil.BigEndian.Uint32(msg.Data[:4]) != 0 {
		// if errno field is not set to 0 (success), this is an error
		return nil, fmt.Errorf("error delivered in message: %v", msg.Data)
	}

	return reply, nil
}

// CloseLasting closes the lasting netlink connection that has been opened using
// AsLasting option when creating this connection. If either no lasting netlink
// connection has been opened or the lasting connection is already in the
// process of closing or has been closed, CloseLasting will immediately return
// without any error.
//
// CloseLasting will terminate all pending netlink operations using the lasting
// connection.
//
// After closing a lasting connection, the connection will revert to using
// on-demand transient netlink connections when calling further netlink
// operations (such as GetTables).
func (cc *Conn) CloseLasting() error {
	// Don't acquire the lock for the whole duration of the CloseLasting
	// operation, but instead only so long as to make sure to only run the
	// netlink socket close on the first time with a lasting netlink socket. As
	// there is only the New() constructor, but no Open() method, it's
	// impossible to reopen a lasting connection.
	cc.mu.Lock()
	nlconn := cc.nlconn
	cc.nlconn = nil
	cc.mu.Unlock()
	if nlconn != nil {
		return nlconn.Close()
	}
	return nil
}

// Flush sends all buffered commands in a single batch to nftables.
func (cc *Conn) Flush() error {
	cc.mu.Lock()
	defer func() {
		cc.messages = nil
		cc.mu.Unlock()
	}()
	if len(cc.messages) == 0 {
		// Messages were already programmed, returning nil
		return nil
	}
	if cc.err != nil {
		return cc.err // serialization error
	}
	conn, closer, err := cc.netlinkConnUnderLock()
	if err != nil {
		return err
	}
	defer func() { _ = closer() }()

	if _, err := conn.SendMessages(batch(cc.messages)); err != nil {
		return fmt.Errorf("SendMessages: %w", err)
	}

	var errs error
	// Fetch the requested acknowledgement for each message we sent.
	for _, msg := range cc.messages {
		if _, err := receiveAckAware(conn, msg.Header.Flags); err != nil {
			if errors.Is(err, os.ErrPermission) || errors.Is(err, syscall.ENOBUFS) {
				// Kernel will only send one error to user space.
				return err
			}
			errs = errors.Join(errs, err)
		}
	}

	if errs != nil {
		return fmt.Errorf("conn.Receive: %w", errs)
	}

	return nil
}

// FlushRuleset flushes the entire ruleset. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Operations_at_ruleset_level
func (cc *Conn) FlushRuleset() {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELTABLE),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: extraHeader(0, 0),
	})
}

func (cc *Conn) dialNetlink() (*netlink.Conn, error) {
	var (
		conn *netlink.Conn
		err  error
	)

	if cc.TestDial != nil {
		conn = nltest.Dial(cc.TestDial)
	} else {
		conn, err = netlink.Dial(unix.NETLINK_NETFILTER, &netlink.Config{NetNS: cc.NetNS})
	}

	if err != nil {
		return nil, err
	}

	for _, opt := range cc.sockOptions {
		if err := opt(conn); err != nil {
			return nil, err
		}
	}

	return conn, nil
}

func (cc *Conn) setErr(err error) {
	if cc.err != nil {
		return
	}
	cc.err = err
}

func (cc *Conn) marshalAttr(attrs []netlink.Attribute) []byte {
	b, err := netlink.MarshalAttributes(attrs)
	if err != nil {
		cc.setErr(err)
		return nil
	}
	return b
}

func (cc *Conn) marshalExpr(fam byte, e expr.Any) []byte {
	b, err := expr.Marshal(fam, e)
	if err != nil {
		cc.setErr(err)
		return nil
	}
	return b
}

func batch(messages []netlink.Message) []netlink.Message {
	batch := []netlink.Message{
		{
			Header: netlink.Header{
				Type:  netlink.HeaderType(unix.NFNL_MSG_BATCH_BEGIN),
				Flags: netlink.Request,
			},
			Data: extraHeader(0, unix.NFNL_SUBSYS_NFTABLES),
		},
	}

	batch = append(batch, messages...)

	batch = append(batch, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType(unix.NFNL_MSG_BATCH_END),
			Flags: netlink.Request,
		},
		Data: extraHeader(0, unix.NFNL_SUBSYS_NFTABLES),
	})

	return batch
}
