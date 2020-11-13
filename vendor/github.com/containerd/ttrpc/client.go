/*
   Copyright The containerd Authors.

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

package ttrpc

import (
	"context"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gogo/protobuf/proto"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ErrClosed is returned by client methods when the underlying connection is
// closed.
var ErrClosed = errors.New("ttrpc: closed")

// Client for a ttrpc server
type Client struct {
	codec   codec
	conn    net.Conn
	channel *channel
	calls   chan *callRequest

	ctx    context.Context
	closed func()

	closeOnce       sync.Once
	userCloseFunc   func()
	userCloseWaitCh chan struct{}

	errOnce     sync.Once
	err         error
	interceptor UnaryClientInterceptor
}

// ClientOpts configures a client
type ClientOpts func(c *Client)

// WithOnClose sets the close func whenever the client's Close() method is called
func WithOnClose(onClose func()) ClientOpts {
	return func(c *Client) {
		c.userCloseFunc = onClose
	}
}

// WithUnaryClientInterceptor sets the provided client interceptor
func WithUnaryClientInterceptor(i UnaryClientInterceptor) ClientOpts {
	return func(c *Client) {
		c.interceptor = i
	}
}

func NewClient(conn net.Conn, opts ...ClientOpts) *Client {
	ctx, cancel := context.WithCancel(context.Background())
	c := &Client{
		codec:           codec{},
		conn:            conn,
		channel:         newChannel(conn),
		calls:           make(chan *callRequest),
		closed:          cancel,
		ctx:             ctx,
		userCloseFunc:   func() {},
		userCloseWaitCh: make(chan struct{}),
		interceptor:     defaultClientInterceptor,
	}

	for _, o := range opts {
		o(c)
	}

	go c.run()
	return c
}

type callRequest struct {
	ctx  context.Context
	req  *Request
	resp *Response  // response will be written back here
	errs chan error // error written here on completion
}

func (c *Client) Call(ctx context.Context, service, method string, req, resp interface{}) error {
	payload, err := c.codec.Marshal(req)
	if err != nil {
		return err
	}

	var (
		creq = &Request{
			Service: service,
			Method:  method,
			Payload: payload,
		}

		cresp = &Response{}
	)

	if metadata, ok := GetMetadata(ctx); ok {
		metadata.setRequest(creq)
	}

	if dl, ok := ctx.Deadline(); ok {
		creq.TimeoutNano = dl.Sub(time.Now()).Nanoseconds()
	}

	info := &UnaryClientInfo{
		FullMethod: fullPath(service, method),
	}
	if err := c.interceptor(ctx, creq, cresp, info, c.dispatch); err != nil {
		return err
	}

	if err := c.codec.Unmarshal(cresp.Payload, resp); err != nil {
		return err
	}

	if cresp.Status != nil && cresp.Status.Code != int32(codes.OK) {
		return status.ErrorProto(cresp.Status)
	}
	return nil
}

func (c *Client) dispatch(ctx context.Context, req *Request, resp *Response) error {
	errs := make(chan error, 1)
	call := &callRequest{
		ctx:  ctx,
		req:  req,
		resp: resp,
		errs: errs,
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case c.calls <- call:
	case <-c.ctx.Done():
		return c.error()
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-errs:
		return filterCloseErr(err)
	case <-c.ctx.Done():
		return c.error()
	}
}

func (c *Client) Close() error {
	c.closeOnce.Do(func() {
		c.closed()
	})
	return nil
}

// UserOnCloseWait is used to blocks untils the user's on-close callback
// finishes.
func (c *Client) UserOnCloseWait(ctx context.Context) error {
	select {
	case <-c.userCloseWaitCh:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type message struct {
	messageHeader
	p   []byte
	err error
}

type receiver struct {
	wg       *sync.WaitGroup
	messages chan *message
	err      error
}

func (r *receiver) run(ctx context.Context, c *channel) {
	defer r.wg.Done()

	for {
		select {
		case <-ctx.Done():
			r.err = ctx.Err()
			return
		default:
			mh, p, err := c.recv()
			if err != nil {
				_, ok := status.FromError(err)
				if !ok {
					// treat all errors that are not an rpc status as terminal.
					// all others poison the connection.
					r.err = filterCloseErr(err)
					return
				}
			}
			select {
			case r.messages <- &message{
				messageHeader: mh,
				p:             p[:mh.Length],
				err:           err,
			}:
			case <-ctx.Done():
				r.err = ctx.Err()
				return
			}
		}
	}
}

func (c *Client) run() {
	var (
		streamID      uint32 = 1
		waiters              = make(map[uint32]*callRequest)
		calls                = c.calls
		incoming             = make(chan *message)
		receiversDone        = make(chan struct{})
		wg            sync.WaitGroup
	)

	// broadcast the shutdown error to the remaining waiters.
	abortWaiters := func(wErr error) {
		for _, waiter := range waiters {
			waiter.errs <- wErr
		}
	}
	recv := &receiver{
		wg:       &wg,
		messages: incoming,
	}
	wg.Add(1)

	go func() {
		wg.Wait()
		close(receiversDone)
	}()
	go recv.run(c.ctx, c.channel)

	defer func() {
		c.conn.Close()
		c.userCloseFunc()
		close(c.userCloseWaitCh)
	}()

	for {
		select {
		case call := <-calls:
			if err := c.send(streamID, messageTypeRequest, call.req); err != nil {
				call.errs <- err
				continue
			}

			waiters[streamID] = call
			streamID += 2 // enforce odd client initiated request ids
		case msg := <-incoming:
			call, ok := waiters[msg.StreamID]
			if !ok {
				logrus.Errorf("ttrpc: received message for unknown channel %v", msg.StreamID)
				continue
			}

			call.errs <- c.recv(call.resp, msg)
			delete(waiters, msg.StreamID)
		case <-receiversDone:
			// all the receivers have exited
			if recv.err != nil {
				c.setError(recv.err)
			}
			// don't return out, let the close of the context trigger the abort of waiters
			c.Close()
		case <-c.ctx.Done():
			abortWaiters(c.error())
			return
		}
	}
}

func (c *Client) error() error {
	c.errOnce.Do(func() {
		if c.err == nil {
			c.err = ErrClosed
		}
	})
	return c.err
}

func (c *Client) setError(err error) {
	c.errOnce.Do(func() {
		c.err = err
	})
}

func (c *Client) send(streamID uint32, mtype messageType, msg interface{}) error {
	p, err := c.codec.Marshal(msg)
	if err != nil {
		return err
	}

	return c.channel.send(streamID, mtype, p)
}

func (c *Client) recv(resp *Response, msg *message) error {
	if msg.err != nil {
		return msg.err
	}

	if msg.Type != messageTypeResponse {
		return errors.New("unknown message type received")
	}

	defer c.channel.putmbuf(msg.p)
	return proto.Unmarshal(msg.p, resp)
}

// filterCloseErr rewrites EOF and EPIPE errors to ErrClosed. Use when
// returning from call or handling errors from main read loop.
//
// This purposely ignores errors with a wrapped cause.
func filterCloseErr(err error) error {
	switch {
	case err == nil:
		return nil
	case err == io.EOF:
		return ErrClosed
	case errors.Cause(err) == io.EOF:
		return ErrClosed
	case strings.Contains(err.Error(), "use of closed network connection"):
		return ErrClosed
	default:
		// if we have an epipe on a write or econnreset on a read , we cast to errclosed
		var oerr *net.OpError
		if errors.As(err, &oerr) && (oerr.Op == "write" || oerr.Op == "read") {
			serr, sok := oerr.Err.(*os.SyscallError)
			if sok && ((serr.Err == syscall.EPIPE && oerr.Op == "write") ||
				(serr.Err == syscall.ECONNRESET && oerr.Op == "read")) {

				return ErrClosed
			}
		}
	}

	return err
}
