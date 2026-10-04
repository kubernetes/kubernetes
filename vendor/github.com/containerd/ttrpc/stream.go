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
	"sync"
	"time"
)

type streamID uint32

type streamMessage struct {
	header  messageHeader
	payload []byte
}

type stream struct {
	id     streamID
	sender sender
	recv   chan *streamMessage

	closeOnce sync.Once
	recvErr   error
	recvClose chan struct{}
}

func newStream(id streamID, send sender, recvBuf int) *stream {
	return &stream{
		id:        id,
		sender:    send,
		recv:      make(chan *streamMessage, recvBuf),
		recvClose: make(chan struct{}),
	}
}

func (s *stream) closeWithError(err error) error {
	s.closeOnce.Do(func() {
		if err != nil {
			s.recvErr = err
		} else {
			s.recvErr = ErrClosed
		}
		close(s.recvClose)
	})
	return nil
}

func (s *stream) send(mt messageType, flags uint8, b []byte) error {
	return s.sender.send(uint32(s.id), mt, flags, b)
}

// receive delivers a message to this stream from the connection receive loop.
// If the stream's recv buffer is full, it waits up to 1 second for the
// consumer to make progress. This keeps the receive loop moving for other
// streams while still providing backpressure under normal operation. If the
// timeout expires the stream is closed with ErrStreamFull.
func (s *stream) receive(ctx context.Context, msg *streamMessage) error {
	select {
	case <-s.recvClose:
		return s.recvErr
	default:
	}
	select {
	case <-s.recvClose:
		return s.recvErr
	case s.recv <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	default:
		// If recv channel is full, wait up to a second for an item
		// to drain and unblock, otherwise close the stream.
		select {
		case <-s.recvClose:
			return s.recvErr
		case s.recv <- msg:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
			s.closeWithError(ErrStreamFull)
			return ErrStreamFull
		}
	}
}

type sender interface {
	send(uint32, messageType, uint8, []byte) error
}
