/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package transport

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/hpack"
)

var updateHeaderTblSize = func(e *hpack.Encoder, v uint32) {
	e.SetMaxDynamicTableSizeLimit(v)
}

type itemNode struct {
	it   interface{}
	next *itemNode
}

type itemList struct {
	head *itemNode
	tail *itemNode
}

func (il *itemList) enqueue(i interface{}) {
	n := &itemNode{it: i}
	if il.tail == nil {
		il.head, il.tail = n, n
		return
	}
	il.tail.next = n
	il.tail = n
}

// peek returns the first item in the list without removing it from the
// list.
func (il *itemList) peek() interface{} {
	return il.head.it
}

func (il *itemList) dequeue() interface{} {
	if il.head == nil {
		return nil
	}
	i := il.head.it
	il.head = il.head.next
	if il.head == nil {
		il.tail = nil
	}
	return i
}

func (il *itemList) dequeueAll() *itemNode {
	h := il.head
	il.head, il.tail = nil, nil
	return h
}

func (il *itemList) isEmpty() bool {
	return il.head == nil
}

// The following defines various control items which could flow through
// the control buffer of transport. They represent different aspects of
// control tasks, e.g., flow control, settings, streaming resetting, etc.

// registerStream is used to register an incoming stream with loopy writer.
type registerStream struct {
	streamID uint32
	wq       *writeQuota
}

// headerFrame is also used to register stream on the client-side.
type headerFrame struct {
	streamID   uint32
	hf         []hpack.HeaderField
	endStream  bool                       // Valid on server side.
	initStream func(uint32) (bool, error) // Used only on the client side.
	onWrite    func()
	wq         *writeQuota    // write quota for the stream created.
	cleanup    *cleanupStream // Valid on the server side.
	onOrphaned func(error)    // Valid on client-side
}

type cleanupStream struct {
	streamID uint32
	idPtr    *uint32
	rst      bool
	rstCode  http2.ErrCode
	onWrite  func()
}

type dataFrame struct {
	streamID  uint32
	endStream bool
	h         []byte
	d         []byte
	// onEachWrite is called every time
	// a part of d is written out.
	onEachWrite func()
}

type incomingWindowUpdate struct {
	streamID  uint32
	increment uint32
}

type outgoingWindowUpdate struct {
	streamID  uint32
	increment uint32
}

type incomingSettings struct {
	ss []http2.Setting
}

type outgoingSettings struct {
	ss []http2.Setting
}

type settingsAck struct {
}

type incomingGoAway struct {
}

type goAway struct {
	code      http2.ErrCode
	debugData []byte
	headsUp   bool
	closeConn bool
}

type ping struct {
	ack  bool
	data [8]byte
}

type outFlowControlSizeRequest struct {
	resp chan uint32
}

type outStreamState int

const (
	active outStreamState = iota
	empty
	waitingOnStreamQuota
)

type outStream struct {
	id               uint32
	state            outStreamState
	itl              *itemList
	bytesOutStanding int
	wq               *writeQuota

	next *outStream
	prev *outStream
}

func (s *outStream) deleteSelf() {
	if s.prev != nil {
		s.prev.next = s.next
	}
	if s.next != nil {
		s.next.prev = s.prev
	}
	s.next, s.prev = nil, nil
}

type outStreamList struct {
	// Following are sentinel objects that mark the
	// beginning and end of the list. They do not
	// contain any item lists. All valid objects are
	// inserted in between them.
	// This is needed so that an outStream object can
	// deleteSelf() in O(1) time without knowing which
	// list it belongs to.
	head *outStream
	tail *outStream
}

func newOutStreamList() *outStreamList {
	head, tail := new(outStream), new(outStream)
	head.next = tail
	tail.prev = head
	return &outStreamList{
		head: head,
		tail: tail,
	}
}

func (l *outStreamList) enqueue(s *outStream) {
	e := l.tail.prev
	e.next = s
	s.prev = e
	s.next = l.tail
	l.tail.prev = s
}

// remove from the beginning of the list.
func (l *outStreamList) dequeue() *outStream {
	b := l.head.next
	if b == l.tail {
		return nil
	}
	b.deleteSelf()
	return b
}

type controlBuffer struct {
	ch              chan struct{}
	done            <-chan struct{}
	mu              sync.Mutex
	consumerWaiting bool
	list            *itemList
	err             error
}

func newControlBuffer(done <-chan struct{}) *controlBuffer {
	return &controlBuffer{
		ch:   make(chan struct{}, 1),
		list: &itemList{},
		done: done,
	}
}

func (c *controlBuffer) put(it interface{}) error {
	_, err := c.executeAndPut(nil, it)
	return err
}

func (c *controlBuffer) executeAndPut(f func(it interface{}) bool, it interface{}) (bool, error) {
	var wakeUp bool
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return false, c.err
	}
	if f != nil {
		if !f(it) { // f wasn't successful
			c.mu.Unlock()
			return false, nil
		}
	}
	if c.consumerWaiting {
		wakeUp = true
		c.consumerWaiting = false
	}
	c.list.enqueue(it)
	c.mu.Unlock()
	if wakeUp {
		select {
		case c.ch <- struct{}{}:
		default:
		}
	}
	return true, nil
}

func (c *controlBuffer) get(block bool) (interface{}, error) {
	for {
		c.mu.Lock()
		if c.err != nil {
			c.mu.Unlock()
			return nil, c.err
		}
		if !c.list.isEmpty() {
			h := c.list.dequeue()
			c.mu.Unlock()
			return h, nil
		}
		if !block {
			c.mu.Unlock()
			return nil, nil
		}
		c.consumerWaiting = true
		c.mu.Unlock()
		select {
		case <-c.ch:
		case <-c.done:
			c.finish()
			return nil, ErrConnClosing
		}
	}
}

func (c *controlBuffer) finish() {
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return
	}
	c.err = ErrConnClosing
	// There may be headers for streams in the control buffer.
	// These streams need to be cleaned out since the transport
	// is still not aware of these yet.
	for head := c.list.dequeueAll(); head != nil; head = head.next {
		hdr, ok := head.it.(*headerFrame)
		if !ok {
			continue
		}
		if hdr.onOrphaned != nil { // It will be nil on the server-side.
			hdr.onOrphaned(ErrConnClosing)
		}
	}
	c.mu.Unlock()
}

type side int

const (
	clientSide side = iota
	serverSide
)

type loopyWriter struct {
	side          side
	cbuf          *controlBuffer
	sendQuota     uint32
	oiws          uint32                // outbound initial window size.
	estdStreams   map[uint32]*outStream // Established streams.
	activeStreams *outStreamList        // Streams that are sending data.
	framer        *framer
	hBuf          *bytes.Buffer  // The buffer for HPACK encoding.
	hEnc          *hpack.Encoder // HPACK encoder.
	bdpEst        *bdpEstimator
	draining      bool

	// Side-specific handlers
	ssGoAwayHandler func(*goAway) (bool, error)
}

func newLoopyWriter(s side, fr *framer, cbuf *controlBuffer, bdpEst *bdpEstimator) *loopyWriter {
	var buf bytes.Buffer
	l := &loopyWriter{
		side:          s,
		cbuf:          cbuf,
		sendQuota:     defaultWindowSize,
		oiws:          defaultWindowSize,
		estdStreams:   make(map[uint32]*outStream),
		activeStreams: newOutStreamList(),
		framer:        fr,
		hBuf:          &buf,
		hEnc:          hpack.NewEncoder(&buf),
		bdpEst:        bdpEst,
	}
	return l
}

const minBatchSize = 1000

// run should be run in a separate goroutine.
func (l *loopyWriter) run() (err error) {
	defer func() {
		if err == ErrConnClosing {
			// Don't log ErrConnClosing as error since it happens
			// 1. When the connection is closed by some other known issue.
			// 2. User closed the connection.
			// 3. A graceful close of connection.
			infof("transport: loopyWriter.run returning. %v", err)
			err = nil
		}
	}()
	for {
		it, err := l.cbuf.get(true)
		if err != nil {
			return err
		}
		if err = l.handle(it); err != nil {
			return err
		}
		if _, err = l.processData(); err != nil {
			return err
		}
		gosched := true
	hasdata:
		for {
			it, err := l.cbuf.get(false)
			if err != nil {
				return err
			}
			if it != nil {
				if err = l.handle(it); err != nil {
					return err
				}
				if _, err = l.processData(); err != nil {
					return err
				}
				continue hasdata
			}
			isEmpty, err := l.processData()
			if err != nil {
				return err
			}
			if !isEmpty {
				continue hasdata
			}
			if gosched {
				gosched = false
				if l.framer.writer.offset < minBatchSize {
					runtime.Gosched()
					continue hasdata
				}
			}
			l.framer.writer.Flush()
			break hasdata

		}
	}
}

func (l *loopyWriter) outgoingWindowUpdateHandler(w *outgoingWindowUpdate) error {
	return l.framer.fr.WriteWindowUpdate(w.streamID, w.increment)
}

func (l *loopyWriter) incomingWindowUpdateHandler(w *incomingWindowUpdate) error {
	// Otherwise update the quota.
	if w.streamID == 0 {
		l.sendQuota += w.increment
		return nil
	}
	// Find the stream and update it.
	if str, ok := l.estdStreams[w.streamID]; ok {
		str.bytesOutStanding -= int(w.increment)
		if strQuota := int(l.oiws) - str.bytesOutStanding; strQuota > 0 && str.state == waitingOnStreamQuota {
			str.state = active
			l.activeStreams.enqueue(str)
			return nil
		}
	}
	return nil
}

func (l *loopyWriter) outgoingSettingsHandler(s *outgoingSettings) error {
	return l.framer.fr.WriteSettings(s.ss...)
}

func (l *loopyWriter) incomingSettingsHandler(s *incomingSettings) error {
	if err := l.applySettings(s.ss); err != nil {
		return err
	}
	return l.framer.fr.WriteSettingsAck()
}

func (l *loopyWriter) registerStreamHandler(h *registerStream) error {
	str := &outStream{
		id:    h.streamID,
		state: empty,
		itl:   &itemList{},
		wq:    h.wq,
	}
	l.estdStreams[h.streamID] = str
	return nil
}

func (l *loopyWriter) headerHandler(h *headerFrame) error {
	if l.side == serverSide {
		str, ok := l.estdStreams[h.streamID]
		if !ok {
			warningf("transport: loopy doesn't recognize the stream: %d", h.streamID)
			return nil
		}
		// Case 1.A: Server is responding back with headers.
		if !h.endStream {
			return l.writeHeader(h.streamID, h.endStream, h.hf, h.onWrite)
		}
		// else:  Case 1.B: Server wants to close stream.

		if str.state != empty { // either active or waiting on stream quota.
			// add it str's list of items.
			str.itl.enqueue(h)
			return nil
		}
		if err := l.writeHeader(h.streamID, h.endStream, h.hf, h.onWrite); err != nil {
			return err
		}
		return l.cleanupStreamHandler(h.cleanup)
	}
	// Case 2: Client wants to originate stream.
	str := &outStream{
		id:    h.streamID,
		state: empty,
		itl:   &itemList{},
		wq:    h.wq,
	}
	str.itl.enqueue(h)
	return l.originateStream(str)
}

func (l *loopyWriter) originateStream(str *outStream) error {
	hdr := str.itl.dequeue().(*headerFrame)
	sendPing, err := hdr.initStream(str.id)
	if err != nil {
		if err == ErrConnClosing {
			return err
		}
		// Other errors(errStreamDrain) need not close transport.
		return nil
	}
	if err = l.writeHeader(str.id, hdr.endStream, hdr.hf, hdr.onWrite); err != nil {
		return err
	}
	l.estdStreams[str.id] = str
	if sendPing {
		return l.pingHandler(&ping{data: [8]byte{}})
	}
	return nil
}

func (l *loopyWriter) writeHeader(streamID uint32, endStream bool, hf []hpack.HeaderField, onWrite func()) error {
	if onWrite != nil {
		onWrite()
	}
	l.hBuf.Reset()
	for _, f := range hf {
		if err := l.hEnc.WriteField(f); err != nil {
			warningf("transport: loopyWriter.writeHeader encountered error while encoding headers:", err)
		}
	}
	var (
		err               error
		endHeaders, first bool
	)
	first = true
	for !endHeaders {
		size := l.hBuf.Len()
		if size > http2MaxFrameLen {
			size = http2MaxFrameLen
		} else {
			endHeaders = true
		}
		if first {
			first = false
			err = l.framer.fr.WriteHeaders(http2.HeadersFrameParam{
				StreamID:      streamID,
				BlockFragment: l.hBuf.Next(size),
				EndStream:     endStream,
				EndHeaders:    endHeaders,
			})
		} else {
			err = l.framer.fr.WriteContinuation(
				streamID,
				endHeaders,
				l.hBuf.Next(size),
			)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (l *loopyWriter) preprocessData(df *dataFrame) error {
	str, ok := l.estdStreams[df.streamID]
	if !ok {
		return nil
	}
	// If we got data for a stream it means that
	// stream was originated and the headers were sent out.
	str.itl.enqueue(df)
	if str.state == empty {
		str.state = active
		l.activeStreams.enqueue(str)
	}
	return nil
}

func (l *loopyWriter) pingHandler(p *ping) error {
	if !p.ack {
		l.bdpEst.timesnap(p.data)
	}
	return l.framer.fr.WritePing(p.ack, p.data)

}

func (l *loopyWriter) outFlowControlSizeRequestHandler(o *outFlowControlSizeRequest) error {
	o.resp <- l.sendQuota
	return nil
}

func (l *loopyWriter) cleanupStreamHandler(c *cleanupStream) error {
	c.onWrite()
	if str, ok := l.estdStreams[c.streamID]; ok {
		// On the server side it could be a trailers-only response or
		// a RST_STREAM before stream initialization thus the stream might
		// not be established yet.
		delete(l.estdStreams, c.streamID)
		str.deleteSelf()
	}
	if c.rst { // If RST_STREAM needs to be sent.
		if err := l.framer.fr.WriteRSTStream(c.streamID, c.rstCode); err != nil {
			return err
		}
	}
	if l.side == clientSide && l.draining && len(l.estdStreams) == 0 {
		return ErrConnClosing
	}
	return nil
}

func (l *loopyWriter) incomingGoAwayHandler(*incomingGoAway) error {
	if l.side == clientSide {
		l.draining = true
		if len(l.estdStreams) == 0 {
			return ErrConnClosing
		}
	}
	return nil
}

func (l *loopyWriter) goAwayHandler(g *goAway) error {
	// Handling of outgoing GoAway is very specific to side.
	if l.ssGoAwayHandler != nil {
		draining, err := l.ssGoAwayHandler(g)
		if err != nil {
			return err
		}
		l.draining = draining
	}
	return nil
}

func (l *loopyWriter) handle(i interface{}) error {
	switch i := i.(type) {
	case *incomingWindowUpdate:
		return l.incomingWindowUpdateHandler(i)
	case *outgoingWindowUpdate:
		return l.outgoingWindowUpdateHandler(i)
	case *incomingSettings:
		return l.incomingSettingsHandler(i)
	case *outgoingSettings:
		return l.outgoingSettingsHandler(i)
	case *headerFrame:
		return l.headerHandler(i)
	case *registerStream:
		return l.registerStreamHandler(i)
	case *cleanupStream:
		return l.cleanupStreamHandler(i)
	case *incomingGoAway:
		return l.incomingGoAwayHandler(i)
	case *dataFrame:
		return l.preprocessData(i)
	case *ping:
		return l.pingHandler(i)
	case *goAway:
		return l.goAwayHandler(i)
	case *outFlowControlSizeRequest:
		return l.outFlowControlSizeRequestHandler(i)
	default:
		return fmt.Errorf("transport: unknown control message type %T", i)
	}
}

func (l *loopyWriter) applySettings(ss []http2.Setting) error {
	for _, s := range ss {
		switch s.ID {
		case http2.SettingInitialWindowSize:
			o := l.oiws
			l.oiws = s.Val
			if o < l.oiws {
				// If the new limit is greater make all depleted streams active.
				for _, stream := range l.estdStreams {
					if stream.state == waitingOnStreamQuota {
						stream.state = active
						l.activeStreams.enqueue(stream)
					}
				}
			}
		case http2.SettingHeaderTableSize:
			updateHeaderTblSize(l.hEnc, s.Val)
		}
	}
	return nil
}

func (l *loopyWriter) processData() (bool, error) {
	if l.sendQuota == 0 {
		return true, nil
	}
	str := l.activeStreams.dequeue()
	if str == nil {
		return true, nil
	}
	dataItem := str.itl.peek().(*dataFrame)
	if len(dataItem.h) == 0 && len(dataItem.d) == 0 {
		// Client sends out empty data frame with endStream = true
		if err := l.framer.fr.WriteData(dataItem.streamID, dataItem.endStream, nil); err != nil {
			return false, err
		}
		str.itl.dequeue()
		if str.itl.isEmpty() {
			str.state = empty
		} else if trailer, ok := str.itl.peek().(*headerFrame); ok { // the next item is trailers.
			if err := l.writeHeader(trailer.streamID, trailer.endStream, trailer.hf, trailer.onWrite); err != nil {
				return false, err
			}
			if err := l.cleanupStreamHandler(trailer.cleanup); err != nil {
				return false, nil
			}
		} else {
			l.activeStreams.enqueue(str)
		}
		return false, nil
	}
	var (
		idx int
		buf []byte
	)
	if len(dataItem.h) != 0 { // data header has not been written out yet.
		buf = dataItem.h
	} else {
		idx = 1
		buf = dataItem.d
	}
	size := http2MaxFrameLen
	if len(buf) < size {
		size = len(buf)
	}
	if strQuota := int(l.oiws) - str.bytesOutStanding; strQuota <= 0 {
		str.state = waitingOnStreamQuota
		return false, nil
	} else if strQuota < size {
		size = strQuota
	}

	if l.sendQuota < uint32(size) {
		size = int(l.sendQuota)
	}
	// Now that outgoing flow controls are checked we can replenish str's write quota
	str.wq.replenish(size)
	var endStream bool
	// This last data message on this stream and all
	// of it can be written in this go.
	if dataItem.endStream && size == len(buf) {
		// buf contains either data or it contains header but data is empty.
		if idx == 1 || len(dataItem.d) == 0 {
			endStream = true
		}
	}
	if dataItem.onEachWrite != nil {
		dataItem.onEachWrite()
	}
	if err := l.framer.fr.WriteData(dataItem.streamID, endStream, buf[:size]); err != nil {
		return false, err
	}
	buf = buf[size:]
	str.bytesOutStanding += size
	l.sendQuota -= uint32(size)
	if idx == 0 {
		dataItem.h = buf
	} else {
		dataItem.d = buf
	}

	if len(dataItem.h) == 0 && len(dataItem.d) == 0 { // All the data from that message was written out.
		str.itl.dequeue()
	}
	if str.itl.isEmpty() {
		str.state = empty
	} else if trailer, ok := str.itl.peek().(*headerFrame); ok { // The next item is trailers.
		if err := l.writeHeader(trailer.streamID, trailer.endStream, trailer.hf, trailer.onWrite); err != nil {
			return false, err
		}
		if err := l.cleanupStreamHandler(trailer.cleanup); err != nil {
			return false, err
		}
	} else if int(l.oiws)-str.bytesOutStanding <= 0 { // Ran out of stream quota.
		str.state = waitingOnStreamQuota
	} else { // Otherwise add it back to the list of active streams.
		l.activeStreams.enqueue(str)
	}
	return false, nil
}
