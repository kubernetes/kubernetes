// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"sync"
	"sync/atomic"
)

// debugMux, if set, causes messages in the connection protocol to be
// logged.
const debugMux = false

// chanList is a thread safe channel list.
type chanList struct {
	// protects concurrent access to chans
	sync.Mutex

	// chans are indexed by the local id of the channel, which the
	// other side should send in the PeersId field.
	chans []*channel

	// This is a debugging aid: it offsets all IDs by this
	// amount. This helps distinguish otherwise identical
	// server/client muxes
	offset uint32
}

// Assigns a channel ID to the given channel.
func (c *chanList) add(ch *channel) uint32 {
	c.Lock()
	defer c.Unlock()
	for i := range c.chans {
		if c.chans[i] == nil {
			c.chans[i] = ch
			return uint32(i) + c.offset
		}
	}
	c.chans = append(c.chans, ch)
	return uint32(len(c.chans)-1) + c.offset
}

// getChan returns the channel for the given ID.
func (c *chanList) getChan(id uint32) *channel {
	id -= c.offset

	c.Lock()
	defer c.Unlock()
	if id < uint32(len(c.chans)) {
		return c.chans[id]
	}
	return nil
}

func (c *chanList) remove(id uint32) {
	id -= c.offset
	c.Lock()
	if id < uint32(len(c.chans)) {
		c.chans[id] = nil
	}
	c.Unlock()
}

// dropAll forgets all channels it knows, returning them in a slice.
func (c *chanList) dropAll() []*channel {
	c.Lock()
	defer c.Unlock()
	var r []*channel

	for _, ch := range c.chans {
		if ch == nil {
			continue
		}
		r = append(r, ch)
	}
	c.chans = nil
	return r
}

// mux represents the state for the SSH connection protocol, which
// multiplexes many channels onto a single packet transport.
type mux struct {
	conn     packetConn
	chanList chanList

	incomingChannels chan NewChannel

	globalSentMu     sync.Mutex
	globalResponses  chan interface{}
	incomingRequests chan *Request

	errCond *sync.Cond
	err     error
}

// When debugging, each new chanList instantiation has a different
// offset.
var globalOff uint32

func (m *mux) Wait() error {
	m.errCond.L.Lock()
	defer m.errCond.L.Unlock()
	for m.err == nil {
		m.errCond.Wait()
	}
	return m.err
}

// newMux returns a mux that runs over the given connection.
func newMux(p packetConn) *mux {
	m := &mux{
		conn:             p,
		incomingChannels: make(chan NewChannel, chanSize),
		globalResponses:  make(chan interface{}, 1),
		incomingRequests: make(chan *Request, chanSize),
		errCond:          newCond(),
	}
	if debugMux {
		m.chanList.offset = atomic.AddUint32(&globalOff, 1)
	}

	go m.loop()
	return m
}

func (m *mux) sendMessage(msg interface{}) error {
	p := Marshal(msg)
	if debugMux {
		log.Printf("send global(%d): %#v", m.chanList.offset, msg)
	}
	return m.conn.writePacket(p)
}

func (m *mux) SendRequest(name string, wantReply bool, payload []byte) (bool, []byte, error) {
	if wantReply {
		m.globalSentMu.Lock()
		defer m.globalSentMu.Unlock()
	}

	if err := m.sendMessage(globalRequestMsg{
		Type:      name,
		WantReply: wantReply,
		Data:      payload,
	}); err != nil {
		return false, nil, err
	}

	if !wantReply {
		return false, nil, nil
	}

	msg, ok := <-m.globalResponses
	if !ok {
		return false, nil, io.EOF
	}
	switch msg := msg.(type) {
	case *globalRequestFailureMsg:
		return false, msg.Data, nil
	case *globalRequestSuccessMsg:
		return true, msg.Data, nil
	default:
		return false, nil, fmt.Errorf("ssh: unexpected response to request: %#v", msg)
	}
}

// ackRequest must be called after processing a global request that
// has WantReply set.
func (m *mux) ackRequest(ok bool, data []byte) error {
	if ok {
		return m.sendMessage(globalRequestSuccessMsg{Data: data})
	}
	return m.sendMessage(globalRequestFailureMsg{Data: data})
}

func (m *mux) Close() error {
	return m.conn.Close()
}

// loop runs the connection machine. It will process packets until an
// error is encountered. To synchronize on loop exit, use mux.Wait.
func (m *mux) loop() {
	var err error
	for err == nil {
		err = m.onePacket()
	}

	for _, ch := range m.chanList.dropAll() {
		ch.close()
	}

	close(m.incomingChannels)
	close(m.incomingRequests)
	close(m.globalResponses)

	m.conn.Close()

	m.errCond.L.Lock()
	m.err = err
	m.errCond.Broadcast()
	m.errCond.L.Unlock()

	if debugMux {
		log.Println("loop exit", err)
	}
}

// onePacket reads and processes one packet.
func (m *mux) onePacket() error {
	packet, err := m.conn.readPacket()
	if err != nil {
		return err
	}

	if debugMux {
		if packet[0] == msgChannelData || packet[0] == msgChannelExtendedData {
			log.Printf("decoding(%d): data packet - %d bytes", m.chanList.offset, len(packet))
		} else {
			p, _ := decode(packet)
			log.Printf("decoding(%d): %d %#v - %d bytes", m.chanList.offset, packet[0], p, len(packet))
		}
	}

	switch packet[0] {
	case msgChannelOpen:
		return m.handleChannelOpen(packet)
	case msgGlobalRequest, msgRequestSuccess, msgRequestFailure:
		return m.handleGlobalPacket(packet)
	}

	// assume a channel packet.
	if len(packet) < 5 {
		return parseError(packet[0])
	}
	id := binary.BigEndian.Uint32(packet[1:])
	ch := m.chanList.getChan(id)
	if ch == nil {
		return fmt.Errorf("ssh: invalid channel %d", id)
	}

	return ch.handlePacket(packet)
}

func (m *mux) handleGlobalPacket(packet []byte) error {
	msg, err := decode(packet)
	if err != nil {
		return err
	}

	switch msg := msg.(type) {
	case *globalRequestMsg:
		m.incomingRequests <- &Request{
			Type:      msg.Type,
			WantReply: msg.WantReply,
			Payload:   msg.Data,
			mux:       m,
		}
	case *globalRequestSuccessMsg, *globalRequestFailureMsg:
		m.globalResponses <- msg
	default:
		panic(fmt.Sprintf("not a global message %#v", msg))
	}

	return nil
}

// handleChannelOpen schedules a channel to be Accept()ed.
func (m *mux) handleChannelOpen(packet []byte) error {
	var msg channelOpenMsg
	if err := Unmarshal(packet, &msg); err != nil {
		return err
	}

	if msg.MaxPacketSize < minPacketLength || msg.MaxPacketSize > 1<<31 {
		failMsg := channelOpenFailureMsg{
			PeersId:  msg.PeersId,
			Reason:   ConnectionFailed,
			Message:  "invalid request",
			Language: "en_US.UTF-8",
		}
		return m.sendMessage(failMsg)
	}

	c := m.newChannel(msg.ChanType, channelInbound, msg.TypeSpecificData)
	c.remoteId = msg.PeersId
	c.maxRemotePayload = msg.MaxPacketSize
	c.remoteWin.add(msg.PeersWindow)
	m.incomingChannels <- c
	return nil
}

func (m *mux) OpenChannel(chanType string, extra []byte) (Channel, <-chan *Request, error) {
	ch, err := m.openChannel(chanType, extra)
	if err != nil {
		return nil, nil, err
	}

	return ch, ch.incomingRequests, nil
}

func (m *mux) openChannel(chanType string, extra []byte) (*channel, error) {
	ch := m.newChannel(chanType, channelOutbound, extra)

	ch.maxIncomingPayload = channelMaxPacket

	open := channelOpenMsg{
		ChanType:         chanType,
		PeersWindow:      ch.myWindow,
		MaxPacketSize:    ch.maxIncomingPayload,
		TypeSpecificData: extra,
		PeersId:          ch.localId,
	}
	if err := m.sendMessage(open); err != nil {
		return nil, err
	}

	switch msg := (<-ch.msg).(type) {
	case *channelOpenConfirmMsg:
		return ch, nil
	case *channelOpenFailureMsg:
		return nil, &OpenChannelError{msg.Reason, msg.Message}
	default:
		return nil, fmt.Errorf("ssh: unexpected packet in response to channel open: %T", msg)
	}
}
