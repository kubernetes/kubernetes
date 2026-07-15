package ldap

import (
	"bufio"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	ber "github.com/go-asn1-ber/asn1-ber"
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

const (
	// DefaultLdapPort default ldap port for pure TCP connection
	DefaultLdapPort = "389"
	// DefaultLdapsPort default ldap port for SSL connection
	DefaultLdapsPort = "636"
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
func (msgCtx *messageContext) sendResponse(packet *PacketResponse, timeout time.Duration) {
	timeoutCtx := context.Background()
	if timeout > 0 {
		var cancelFunc context.CancelFunc
		timeoutCtx, cancelFunc = context.WithTimeout(context.Background(), timeout)
		defer cancelFunc()
	}
	select {
	case msgCtx.responses <- packet:
		// Successfully sent packet to message handler.
	case <-msgCtx.done:
		// The request handler is done and will not receive more
		// packets.
	case <-timeoutCtx.Done():
		// The timeout was reached before the packet was sent.
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
	// requestTimeout is loaded atomically
	// so we need to ensure 64-bit alignment on 32-bit platforms.
	// https://github.com/go-ldap/ldap/pull/199
	requestTimeout      int64
	conn                net.Conn
	isTLS               bool
	closing             uint32
	closeErr            atomic.Value
	isStartingTLS       bool
	Debug               debugging
	chanConfirm         chan struct{}
	messageContexts     map[int64]*messageContext
	chanMessage         chan *messagePacket
	chanMessageID       chan int64
	wgClose             sync.WaitGroup
	outstandingRequests uint
	messageMutex        sync.Mutex

	err error
}

var _ Client = &Conn{}

// DefaultTimeout is a package-level variable that sets the timeout value
// used for the Dial and DialTLS methods.
//
// WARNING: since this is a package-level variable, setting this value from
// multiple places will probably result in undesired behaviour.
var DefaultTimeout = 60 * time.Second

// DialOpt configures DialContext.
type DialOpt func(*DialContext)

// DialWithDialer updates net.Dialer in DialContext.
func DialWithDialer(d *net.Dialer) DialOpt {
	return func(dc *DialContext) {
		dc.dialer = d
	}
}

// DialWithTLSConfig updates tls.Config in DialContext.
func DialWithTLSConfig(tc *tls.Config) DialOpt {
	return func(dc *DialContext) {
		dc.tlsConfig = tc
	}
}

// DialWithTLSDialer is a wrapper for DialWithTLSConfig with the option to
// specify a net.Dialer to for example define a timeout or a custom resolver.
//
// Deprecated:  Use DialWithDialer and DialWithTLSConfig instead
func DialWithTLSDialer(tlsConfig *tls.Config, dialer *net.Dialer) DialOpt {
	return func(dc *DialContext) {
		dc.tlsConfig = tlsConfig
		dc.dialer = dialer
	}
}

// DialContext contains necessary parameters to dial the given ldap URL.
type DialContext struct {
	dialer    *net.Dialer
	tlsConfig *tls.Config
}

func (dc *DialContext) dial(u *url.URL) (net.Conn, error) {
	if u.Scheme == "ldapi" {
		if u.Path == "" || u.Path == "/" {
			u.Path = "/var/run/slapd/ldapi"
		}
		return dc.dialer.Dial("unix", u.Path)
	}

	host, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		// we assume that error is due to missing port
		host = u.Host
		port = ""
	}

	switch u.Scheme {
	case "cldap":
		if port == "" {
			port = DefaultLdapPort
		}
		return dc.dialer.Dial("udp", net.JoinHostPort(host, port))
	case "ldap":
		if port == "" {
			port = DefaultLdapPort
		}
		return dc.dialer.Dial("tcp", net.JoinHostPort(host, port))
	case "ldaps":
		if port == "" {
			port = DefaultLdapsPort
		}
		return tls.DialWithDialer(dc.dialer, "tcp", net.JoinHostPort(host, port), dc.tlsConfig)
	}

	return nil, fmt.Errorf("Unknown scheme '%s'", u.Scheme)
}

// Dial connects to the given address on the given network using net.Dial
// and then returns a new Conn for the connection.
//
// Deprecated:  Use DialURL instead.
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
//
// Deprecated:  Use DialURL instead.
func DialTLS(network, addr string, config *tls.Config) (*Conn, error) {
	c, err := tls.DialWithDialer(&net.Dialer{Timeout: DefaultTimeout}, network, addr, config)
	if err != nil {
		return nil, NewError(ErrorNetwork, err)
	}
	conn := NewConn(c, true)
	conn.Start()
	return conn, nil
}

// DialURL connects to the given ldap URL.
// The following schemas are supported: ldap://, ldaps://, ldapi://,
// and cldap:// (RFC1798, deprecated but used by Active Directory).
// On success a new Conn for the connection is returned.
func DialURL(addr string, opts ...DialOpt) (*Conn, error) {
	u, err := url.Parse(addr)
	if err != nil {
		return nil, NewError(ErrorNetwork, err)
	}

	var dc DialContext
	for _, opt := range opts {
		opt(&dc)
	}
	if dc.dialer == nil {
		dc.dialer = &net.Dialer{Timeout: DefaultTimeout}
	}

	c, err := dc.dial(u)
	if err != nil {
		return nil, NewError(ErrorNetwork, err)
	}

	conn := NewConn(c, u.Scheme == "ldaps")
	conn.Start()
	return conn, nil
}

// NewConn returns a new Conn using conn for network I/O.
func NewConn(conn net.Conn, isTLS bool) *Conn {
	l := &Conn{
		conn:            conn,
		chanConfirm:     make(chan struct{}),
		chanMessageID:   make(chan int64),
		chanMessage:     make(chan *messagePacket, 10),
		messageContexts: map[int64]*messageContext{},
		requestTimeout:  0,
		isTLS:           isTLS,
	}
	l.wgClose.Add(1)
	return l
}

// Start initialises goroutines to read replies and process messages.
// Warning: Calling this function in addition to Dial or DialURL
// may cause race conditions.
//
// See: https://github.com/go-ldap/ldap/issues/356
func (l *Conn) Start() {
	go l.reader()
	go l.processMessages()
}

// IsClosing returns whether or not we're currently closing.
func (l *Conn) IsClosing() bool {
	return atomic.LoadUint32(&l.closing) == 1
}

// setClosing sets the closing value to true
func (l *Conn) setClosing() bool {
	return atomic.CompareAndSwapUint32(&l.closing, 0, 1)
}

// Close closes the connection.
func (l *Conn) Close() (err error) {
	l.messageMutex.Lock()
	defer l.messageMutex.Unlock()

	if l.setClosing() {
		l.Debug.Printf("Sending quit message and waiting for confirmation")
		l.chanMessage <- &messagePacket{Op: MessageQuit}

		timeoutCtx := context.Background()
		if l.getTimeout() > 0 {
			var cancelFunc context.CancelFunc
			timeoutCtx, cancelFunc = context.WithTimeout(timeoutCtx, time.Duration(l.getTimeout()))
			defer cancelFunc()
		}
		select {
		case <-l.chanConfirm:
			// Confirmation was received.
		case <-timeoutCtx.Done():
			// The timeout was reached before confirmation was received.
		}

		close(l.chanMessage)

		l.Debug.Printf("Closing network connection")
		err = l.conn.Close()
		l.wgClose.Done()
	}
	l.wgClose.Wait()

	return err
}

// SetTimeout sets the time after a request is sent that a MessageTimeout triggers
func (l *Conn) SetTimeout(timeout time.Duration) {
	atomic.StoreInt64(&l.requestTimeout, int64(timeout))
}

func (l *Conn) getTimeout() int64 {
	return atomic.LoadInt64(&l.requestTimeout)
}

// Returns the next available messageID
func (l *Conn) nextMessageID() int64 {
	if messageID, ok := <-l.chanMessageID; ok {
		return messageID
	}
	return 0
}

// GetLastError returns the last recorded error from goroutines like processMessages and reader.
// Only the last recorded error will be returned.
func (l *Conn) GetLastError() error {
	l.messageMutex.Lock()
	defer l.messageMutex.Unlock()
	return l.err
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
		l.Debug.PrintPacket(packet)
	}

	if err := GetLDAPError(packet); err == nil {
		conn := tls.Client(l.conn, config)

		if connErr := conn.Handshake(); connErr != nil {
			l.Close()
			return NewError(ErrorNetwork, fmt.Errorf("TLS handshake failed (%v)", connErr))
		}

		l.isTLS = true
		l.conn = conn
	} else {
		return err
	}
	go l.reader()

	return nil
}

// TLSConnectionState returns the client's TLS connection state.
// The return values are their zero values if StartTLS did
// not succeed.
func (l *Conn) TLSConnectionState() (state tls.ConnectionState, ok bool) {
	tc, ok := l.conn.(*tls.Conn)
	if !ok {
		return
	}
	return tc.ConnectionState(), true
}

func (l *Conn) sendMessage(packet *ber.Packet) (*messageContext, error) {
	return l.sendMessageWithFlags(packet, 0)
}

func (l *Conn) sendMessageWithFlags(packet *ber.Packet, flags sendMessageFlags) (*messageContext, error) {
	if l.IsClosing() {
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
	if !l.sendProcessMessage(message) {
		if l.IsClosing() {
			return nil, NewError(ErrorNetwork, errors.New("ldap: connection closed"))
		}
		return nil, NewError(ErrorNetwork, errors.New("ldap: could not send message for unknown reason"))
	}
	return message.Context, nil
}

func (l *Conn) finishMessage(msgCtx *messageContext) {
	close(msgCtx.done)

	if l.IsClosing() {
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
	if l.IsClosing() {
		return false
	}
	l.chanMessage <- message
	return true
}

func (l *Conn) processMessages() {
	defer func() {
		if err := recover(); err != nil {
			l.err = fmt.Errorf("ldap: recovered panic in processMessages: %v", err)
		}
		for messageID, msgCtx := range l.messageContexts {
			// If we are closing due to an error, inform anyone who
			// is waiting about the error.
			if l.IsClosing() && l.closeErr.Load() != nil {
				msgCtx.sendResponse(&PacketResponse{Error: l.closeErr.Load().(error)}, time.Duration(l.getTimeout()))
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
					message.Context.sendResponse(&PacketResponse{Error: fmt.Errorf("unable to send request: %s", err)}, time.Duration(l.getTimeout()))
					close(message.Context.responses)
					break
				}

				// Only add to messageContexts if we were able to
				// successfully write the message.
				l.messageContexts[message.MessageID] = message.Context

				// Add timeout if defined
				requestTimeout := l.getTimeout()
				if requestTimeout > 0 {
					go func() {
						timer := time.NewTimer(time.Duration(requestTimeout))
						defer func() {
							if err := recover(); err != nil {
								l.err = fmt.Errorf("ldap: recovered panic in RequestTimeout: %v", err)
							}

							timer.Stop()
						}()

						select {
						case <-timer.C:
							timeoutMessage := &messagePacket{
								Op:        MessageTimeout,
								MessageID: message.MessageID,
							}
							l.sendProcessMessage(timeoutMessage)
						case <-message.Context.done:
						}
					}()
				}
			case MessageResponse:
				l.Debug.Printf("Receiving message %d", message.MessageID)
				if msgCtx, ok := l.messageContexts[message.MessageID]; ok {
					msgCtx.sendResponse(&PacketResponse{message.Packet, nil}, time.Duration(l.getTimeout()))
				} else {
					l.err = fmt.Errorf("ldap: received unexpected message %d, %v", message.MessageID, l.IsClosing())
					l.Debug.PrintPacket(message.Packet)
				}
			case MessageTimeout:
				// Handle the timeout by closing the channel
				// All reads will return immediately
				if msgCtx, ok := l.messageContexts[message.MessageID]; ok {
					l.Debug.Printf("Receiving message timeout for %d", message.MessageID)
					msgCtx.sendResponse(&PacketResponse{message.Packet, NewError(ErrorNetwork, errors.New("ldap: connection timed out"))}, time.Duration(l.getTimeout()))
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
			l.err = fmt.Errorf("ldap: recovered panic in reader: %v", err)
		}
		if !cleanstop {
			l.Close()
		}
	}()

	bufConn := bufio.NewReader(l.conn)
	for {
		if cleanstop {
			l.Debug.Printf("reader clean stopping (without closing the connection)")
			return
		}
		packet, err := ber.ReadPacket(bufConn)
		if err != nil {
			// A read error is expected here if we are closing the connection...
			if !l.IsClosing() {
				l.closeErr.Store(fmt.Errorf("unable to read LDAP response packet: %s", err))
				l.Debug.Printf("reader error: %s", err)
			}
			return
		}
		if err := addLDAPDescriptions(packet); err != nil {
			l.Debug.Printf("descriptions error: %s", err)
		}
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
