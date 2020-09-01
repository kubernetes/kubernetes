// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ldap

import (
	"crypto/tls"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"gopkg.in/asn1-ber.v1"
)

const (
	// MessageQuit causes the processMessages loop to exit
	MessageQuit = 0
	// MessageRequest sends a request to the server
	MessageRequest = 1
	// MessageResponse receives a response from the server
	MessageResponse = 2
	// MessageFinish indicates the client considers a particular message ID to be finished
	MessageFinish = 3
	// MessageTimeout indicates the client-specified timeout for a particular message ID has been reached
	MessageTimeout = 4
)

// PacketResponse contains the packet or error encountered reading a response
type PacketResponse struct {
	// Packet is the packet read from the server
	Packet *ber.Packet
	// Error is an error encountered while reading
	Error error
}

// ReadPacket returns the packet or an error
func (pr *PacketResponse) ReadPacket() (*ber.Packet, error) {
	if (pr == nil) || (pr.Packet == nil && pr.Error == nil) {
		return nil, NewError(ErrorNetwork, errors.New("ldap: could not retrieve response"))
	}
	return pr.Packet, pr.Error
}

type messageContext struct {
	id int64
	// close(done) should only be called from finishMessage()
	done chan struct{}
	// close(responses) should only be called from processMessages(), and only sent to from sendResponse()
	responses chan *PacketResponse
}

// sendResponse should only be called within the processMessages() loop which
// is also responsible for closing the responses channel.
func (msgCtx *messageContext) sendResponse(packet *PacketResponse) {
	select {
	case msgCtx.responses <- packet:
		// Successfully sent packet to message handler.
	case <-msgCtx.done:
		// The request handler is done and will not receive more
		// packets.
	}
}

type messagePacket struct {
	Op        int
	MessageID int64
	Packet    *ber.Packet
	Context   *messageContext
}

type sendMessageFlags uint

const (
	startTLS sendMessageFlags = 1 << iota
)

// Conn represents an LDAP Connection
type Conn struct {
	conn                net.Conn
	isTLS               bool
	closing             uint32
	closeErr            atomicValue
	isStartingTLS       bool
	Debug               debugging
	chanConfirm         chan struct{}
	messageContexts     map[int64]*messageContext
	chanMessage         chan *messagePacket
	chanMessageID       chan int64
	wgClose             sync.WaitGroup
	outstandingRequests uint
	messageMutex        sync.Mutex
	requestTimeout      int64
}

var _ Client = &Conn{}

// DefaultTimeout is a package-level variable that sets the timeout value
// used for the Dial and DialTLS methods.
//
// WARNING: since this is a package-level variable, setting this value from
// multiple places will probably result in undesired behaviour.
var DefaultTimeout = 60 * time.Second

// Dial connects to the given address on the given network using net.Dial
// and then returns a new Conn for the connection.
func Dial(network, addr string) (*Conn, error) {
	c, err := net.DialTimeout(network, addr, DefaultTimeout)
	if err != nil {
		return nil, NewError(ErrorNetwork, err)
	}
	conn := NewConn(c, false)
	conn.Start()
	return conn, nil
}

// DialTLS connects to the given address on the given network using tls.Dial
// and then returns a new Conn for the connection.
func DialTLS(network, addr string, config *tls.Config) (*Conn, error) {
	dc, err := net.DialTimeout(network, addr, DefaultTimeout)
	if err != nil {
		return nil, NewError(ErrorNetwork, err)
	}
	c := tls.Client(dc, config)
	err = c.Handshake()
	if err != nil {
		// Handshake error, close the established connection before we return an error
		dc.Close()
		return nil, NewError(ErrorNetwork, err)
	}
	conn := NewConn(c, true)
	conn.Start()
	return conn, nil
}

// NewConn returns a new Conn using conn for network I/O.
func NewConn(conn net.Conn, isTLS bool) *Conn {
	return &Conn{
		conn:            conn,
		chanConfirm:     make(chan struct{}),
		chanMessageID:   make(chan int64),
		chanMessage:     make(chan *messagePacket, 10),
		messageContexts: map[int64]*messageContext{},
		requestTimeout:  0,
		isTLS:           isTLS,
	}
}

// Start initializes goroutines to read responses and process messages
func (l *Conn) Start() {
	go l.reader()
	go l.processMessages()
	l.wgClose.Add(1)
}

// isClosing returns whether or not we're currently closing.
func (l *Conn) isClosing() bool {
	return atomic.LoadUint32(&l.closing) == 1
}

// setClosing sets the closing value to true
func (l *Conn) setClosing() bool {
	return atomic.CompareAndSwapUint32(&l.closing, 0, 1)
}

// Close closes the connection.
func (l *Conn) Close() {
	l.messageMutex.Lock()
	defer l.messageMutex.Unlock()

	if l.setClosing() {
		l.Debug.Printf("Sending quit message and waiting for confirmation")
		l.chanMessage <- &messagePacket{Op: MessageQuit}
		<-l.chanConfirm
		close(l.chanMessage)

		l.Debug.Printf("Closing network connection")
		if err := l.conn.Close(); err != nil {
			log.Println(err)
		}

		l.wgClose.Done()
	}
	l.wgClose.Wait()
}

// SetTimeout sets the time after a request is sent that a MessageTimeout triggers
func (l *Conn) SetTimeout(timeout time.Duration) {
	if timeout > 0 {
		atomic.StoreInt64(&l.requestTimeout, int64(timeout))
	}
}

// Returns the next available messageID
func (l *Conn) nextMessageID() int64 {
	if messageID, ok := <-l.chanMessageID; ok {
		return messageID
	}
	return 0
}

// StartTLS sends the command to start a TLS session and then creates a new TLS Client
func (l *Conn) StartTLS(config *tls.Config) error {
	if l.isTLS {
		return NewError(ErrorNetwork, errors.New("ldap: already encrypted"))
	}

	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationExtendedRequest, nil, "Start TLS")
	request.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, "1.3.6.1.4.1.1466.20037", "TLS Extended Command"))
	packet.AppendChild(request)
	l.Debug.PrintPacket(packet)

	msgCtx, err := l.sendMessageWithFlags(packet, startTLS)
	if err != nil {
		return err
	}
	defer l.finishMessage(msgCtx)

	l.Debug.Printf("%d: waiting for response", msgCtx.id)

	packetResponse, ok := <-msgCtx.responses
	if !ok {
		return NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
	}
	packet, err = packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if err != nil {
		return err
	}

	if l.Debug {
		if err := addLDAPDescriptions(packet); err != nil {
			l.Close()
			return err
		}
		ber.PrintPacket(packet)
	}

	if resultCode, message := getLDAPResultCode(packet); resultCode == LDAPResultSuccess {
		conn := tls.Client(l.conn, config)

		if err := conn.Handshake(); err != nil {
			l.Close()
			return NewError(ErrorNetwork, fmt.Errorf("TLS handshake failed (%v)", err))
		}

		l.isTLS = true
		l.conn = conn
	} else {
		return NewError(resultCode, fmt.Errorf("ldap: cannot StartTLS (%s)", message))
	}
	go l.reader()

	return nil
}

func (l *Conn) sendMessage(packet *ber.Packet) (*messageContext, error) {
	return l.sendMessageWithFlags(packet, 0)
}

func (l *Conn) sendMessageWithFlags(packet *ber.Packet, flags sendMessageFlags) (*messageContext, error) {
	if l.isClosing() {
		return nil, NewError(ErrorNetwork, errors.New("ldap: connection closed"))
	}
	l.messageMutex.Lock()
	l.Debug.Printf("flags&startTLS = %d", flags&startTLS)
	if l.isStartingTLS {
		l.messageMutex.Unlock()
		return nil, NewError(ErrorNetwork, errors.New("ldap: connection is in startls phase"))
	}
	if flags&startTLS != 0 {
		if l.outstandingRequests != 0 {
			l.messageMutex.Unlock()
			return nil, NewError(ErrorNetwork, errors.New("ldap: cannot StartTLS with outstanding requests"))
		}
		l.isStartingTLS = true
	}
	l.outstandingRequests++

	l.messageMutex.Unlock()

	responses := make(chan *PacketResponse)
	messageID := packet.Children[0].Value.(int64)
	message := &messagePacket{
		Op:        MessageRequest,
		MessageID: messageID,
		Packet:    packet,
		Context: &messageContext{
			id:        messageID,
			done:      make(chan struct{}),
			responses: responses,
		},
	}
	l.sendProcessMessage(message)
	return message.Context, nil
}

func (l *Conn) finishMessage(msgCtx *messageContext) {
	close(msgCtx.done)

	if l.isClosing() {
		return
	}

	l.messageMutex.Lock()
	l.outstandingRequests--
	if l.isStartingTLS {
		l.isStartingTLS = false
	}
	l.messageMutex.Unlock()

	message := &messagePacket{
		Op:        MessageFinish,
		MessageID: msgCtx.id,
	}
	l.sendProcessMessage(message)
}

func (l *Conn) sendProcessMessage(message *messagePacket) bool {
	l.messageMutex.Lock()
	defer l.messageMutex.Unlock()
	if l.isClosing() {
		return false
	}
	l.chanMessage <- message
	return true
}

func (l *Conn) processMessages() {
	defer func() {
		if err := recover(); err != nil {
			log.Printf("ldap: recovered panic in processMessages: %v", err)
		}
		for messageID, msgCtx := range l.messageContexts {
			// If we are closing due to an error, inform anyone who
			// is waiting about the error.
			if l.isClosing() && l.closeErr.Load() != nil {
				msgCtx.sendResponse(&PacketResponse{Error: l.closeErr.Load().(error)})
			}
			l.Debug.Printf("Closing channel for MessageID %d", messageID)
			close(msgCtx.responses)
			delete(l.messageContexts, messageID)
		}
		close(l.chanMessageID)
		close(l.chanConfirm)
	}()

	var messageID int64 = 1
	for {
		select {
		case l.chanMessageID <- messageID:
			messageID++
		case message := <-l.chanMessage:
			switch message.Op {
			case MessageQuit:
				l.Debug.Printf("Shutting down - quit message received")
				return
			case MessageRequest:
				// Add to message list and write to network
				l.Debug.Printf("Sending message %d", message.MessageID)

				buf := message.Packet.Bytes()
				_, err := l.conn.Write(buf)
				if err != nil {
					l.Debug.Printf("Error Sending Message: %s", err.Error())
					message.Context.sendResponse(&PacketResponse{Error: fmt.Errorf("unable to send request: %s", err)})
					close(message.Context.responses)
					break
				}

				// Only add to messageContexts if we were able to
				// successfully write the message.
				l.messageContexts[message.MessageID] = message.Context

				// Add timeout if defined
				requestTimeout := time.Duration(atomic.LoadInt64(&l.requestTimeout))
				if requestTimeout > 0 {
					go func() {
						defer func() {
							if err := recover(); err != nil {
								log.Printf("ldap: recovered panic in RequestTimeout: %v", err)
							}
						}()
						time.Sleep(requestTimeout)
						timeoutMessage := &messagePacket{
							Op:        MessageTimeout,
							MessageID: message.MessageID,
						}
						l.sendProcessMessage(timeoutMessage)
					}()
				}
			case MessageResponse:
				l.Debug.Printf("Receiving message %d", message.MessageID)
				if msgCtx, ok := l.messageContexts[message.MessageID]; ok {
					msgCtx.sendResponse(&PacketResponse{message.Packet, nil})
				} else {
					log.Printf("Received unexpected message %d, %v", message.MessageID, l.isClosing())
					ber.PrintPacket(message.Packet)
				}
			case MessageTimeout:
				// Handle the timeout by closing the channel
				// All reads will return immediately
				if msgCtx, ok := l.messageContexts[message.MessageID]; ok {
					l.Debug.Printf("Receiving message timeout for %d", message.MessageID)
					msgCtx.sendResponse(&PacketResponse{message.Packet, errors.New("ldap: connection timed out")})
					delete(l.messageContexts, message.MessageID)
					close(msgCtx.responses)
				}
			case MessageFinish:
				l.Debug.Printf("Finished message %d", message.MessageID)
				if msgCtx, ok := l.messageContexts[message.MessageID]; ok {
					delete(l.messageContexts, message.MessageID)
					close(msgCtx.responses)
				}
			}
		}
	}
}

func (l *Conn) reader() {
	cleanstop := false
	defer func() {
		if err := recover(); err != nil {
			log.Printf("ldap: recovered panic in reader: %v", err)
		}
		if !cleanstop {
			l.Close()
		}
	}()

	for {
		if cleanstop {
			l.Debug.Printf("reader clean stopping (without closing the connection)")
			return
		}
		packet, err := ber.ReadPacket(l.conn)
		if err != nil {
			// A read error is expected here if we are closing the connection...
			if !l.isClosing() {
				l.closeErr.Store(fmt.Errorf("unable to read LDAP response packet: %s", err))
				l.Debug.Printf("reader error: %s", err.Error())
			}
			return
		}
		addLDAPDescriptions(packet)
		if len(packet.Children) == 0 {
			l.Debug.Printf("Received bad ldap packet")
			continue
		}
		l.messageMutex.Lock()
		if l.isStartingTLS {
			cleanstop = true
		}
		l.messageMutex.Unlock()
		message := &messagePacket{
			Op:        MessageResponse,
			MessageID: packet.Children[0].Value.(int64),
			Packet:    packet,
		}
		if !l.sendProcessMessage(message) {
			return
		}
	}
}
