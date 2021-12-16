package quic

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lucas-clemente/quic-go/internal/handshake"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
	"github.com/lucas-clemente/quic-go/logging"
)

// packetHandler handles packets
type packetHandler interface {
	handlePacket(*receivedPacket)
	shutdown()
	destroy(error)
	getPerspective() protocol.Perspective
}

type unknownPacketHandler interface {
	handlePacket(*receivedPacket)
	setCloseError(error)
}

type packetHandlerManager interface {
	AddWithConnID(protocol.ConnectionID, protocol.ConnectionID, func() packetHandler) bool
	Destroy() error
	sessionRunner
	SetServer(unknownPacketHandler)
	CloseServer()
}

type quicSession interface {
	EarlySession
	earlySessionReady() <-chan struct{}
	handlePacket(*receivedPacket)
	GetVersion() protocol.VersionNumber
	getPerspective() protocol.Perspective
	run() error
	destroy(error)
	shutdown()
}

// A Listener of QUIC
type baseServer struct {
	mutex sync.Mutex

	acceptEarlySessions bool

	tlsConf *tls.Config
	config  *Config

	conn connection
	// If the server is started with ListenAddr, we create a packet conn.
	// If it is started with Listen, we take a packet conn as a parameter.
	createdPacketConn bool

	tokenGenerator *handshake.TokenGenerator

	sessionHandler packetHandlerManager

	receivedPackets chan *receivedPacket

	// set as a member, so they can be set in the tests
	newSession func(
		sendConn,
		sessionRunner,
		protocol.ConnectionID, /* original dest connection ID */
		*protocol.ConnectionID, /* retry src connection ID */
		protocol.ConnectionID, /* client dest connection ID */
		protocol.ConnectionID, /* destination connection ID */
		protocol.ConnectionID, /* source connection ID */
		protocol.StatelessResetToken,
		*Config,
		*tls.Config,
		*handshake.TokenGenerator,
		bool, /* enable 0-RTT */
		logging.ConnectionTracer,
		uint64,
		utils.Logger,
		protocol.VersionNumber,
	) quicSession

	serverError error
	errorChan   chan struct{}
	closed      bool
	running     chan struct{} // closed as soon as run() returns

	sessionQueue    chan quicSession
	sessionQueueLen int32 // to be used as an atomic

	logger utils.Logger
}

var (
	_ Listener             = &baseServer{}
	_ unknownPacketHandler = &baseServer{}
)

type earlyServer struct{ *baseServer }

var _ EarlyListener = &earlyServer{}

func (s *earlyServer) Accept(ctx context.Context) (EarlySession, error) {
	return s.baseServer.accept(ctx)
}

// ListenAddr creates a QUIC server listening on a given address.
// The tls.Config must not be nil and must contain a certificate configuration.
// The quic.Config may be nil, in that case the default values will be used.
func ListenAddr(addr string, tlsConf *tls.Config, config *Config) (Listener, error) {
	return listenAddr(addr, tlsConf, config, false)
}

// ListenAddrEarly works like ListenAddr, but it returns sessions before the handshake completes.
func ListenAddrEarly(addr string, tlsConf *tls.Config, config *Config) (EarlyListener, error) {
	s, err := listenAddr(addr, tlsConf, config, true)
	if err != nil {
		return nil, err
	}
	return &earlyServer{s}, nil
}

func listenAddr(addr string, tlsConf *tls.Config, config *Config, acceptEarly bool) (*baseServer, error) {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return nil, err
	}
	serv, err := listen(conn, tlsConf, config, acceptEarly)
	if err != nil {
		return nil, err
	}
	serv.createdPacketConn = true
	return serv, nil
}

// Listen listens for QUIC connections on a given net.PacketConn. If the
// PacketConn satisfies the OOBCapablePacketConn interface (as a net.UDPConn
// does), ECN and packet info support will be enabled. In this case, ReadMsgUDP
// and WriteMsgUDP will be used instead of ReadFrom and WriteTo to read/write
// packets. A single net.PacketConn only be used for a single call to Listen.
// The PacketConn can be used for simultaneous calls to Dial. QUIC connection
// IDs are used for demultiplexing the different connections. The tls.Config
// must not be nil and must contain a certificate configuration. The
// tls.Config.CipherSuites allows setting of TLS 1.3 cipher suites. Furthermore,
// it must define an application control (using NextProtos). The quic.Config may
// be nil, in that case the default values will be used.
func Listen(conn net.PacketConn, tlsConf *tls.Config, config *Config) (Listener, error) {
	return listen(conn, tlsConf, config, false)
}

// ListenEarly works like Listen, but it returns sessions before the handshake completes.
func ListenEarly(conn net.PacketConn, tlsConf *tls.Config, config *Config) (EarlyListener, error) {
	s, err := listen(conn, tlsConf, config, true)
	if err != nil {
		return nil, err
	}
	return &earlyServer{s}, nil
}

func listen(conn net.PacketConn, tlsConf *tls.Config, config *Config, acceptEarly bool) (*baseServer, error) {
	if tlsConf == nil {
		return nil, errors.New("quic: tls.Config not set")
	}
	if err := validateConfig(config); err != nil {
		return nil, err
	}
	config = populateServerConfig(config)
	for _, v := range config.Versions {
		if !protocol.IsValidVersion(v) {
			return nil, fmt.Errorf("%s is not a valid QUIC version", v)
		}
	}

	sessionHandler, err := getMultiplexer().AddConn(conn, config.ConnectionIDLength, config.StatelessResetKey, config.Tracer)
	if err != nil {
		return nil, err
	}
	tokenGenerator, err := handshake.NewTokenGenerator(rand.Reader)
	if err != nil {
		return nil, err
	}
	c, err := wrapConn(conn)
	if err != nil {
		return nil, err
	}
	s := &baseServer{
		conn:                c,
		tlsConf:             tlsConf,
		config:              config,
		tokenGenerator:      tokenGenerator,
		sessionHandler:      sessionHandler,
		sessionQueue:        make(chan quicSession),
		errorChan:           make(chan struct{}),
		running:             make(chan struct{}),
		receivedPackets:     make(chan *receivedPacket, protocol.MaxServerUnprocessedPackets),
		newSession:          newSession,
		logger:              utils.DefaultLogger.WithPrefix("server"),
		acceptEarlySessions: acceptEarly,
	}
	go s.run()
	sessionHandler.SetServer(s)
	s.logger.Debugf("Listening for %s connections on %s", conn.LocalAddr().Network(), conn.LocalAddr().String())
	return s, nil
}

func (s *baseServer) run() {
	defer close(s.running)
	for {
		select {
		case <-s.errorChan:
			return
		default:
		}
		select {
		case <-s.errorChan:
			return
		case p := <-s.receivedPackets:
			if bufferStillInUse := s.handlePacketImpl(p); !bufferStillInUse {
				p.buffer.Release()
			}
		}
	}
}

var defaultAcceptToken = func(clientAddr net.Addr, token *Token) bool {
	if token == nil {
		return false
	}
	validity := protocol.TokenValidity
	if token.IsRetryToken {
		validity = protocol.RetryTokenValidity
	}
	if time.Now().After(token.SentTime.Add(validity)) {
		return false
	}
	var sourceAddr string
	if udpAddr, ok := clientAddr.(*net.UDPAddr); ok {
		sourceAddr = udpAddr.IP.String()
	} else {
		sourceAddr = clientAddr.String()
	}
	return sourceAddr == token.RemoteAddr
}

// Accept returns sessions that already completed the handshake.
// It is only valid if acceptEarlySessions is false.
func (s *baseServer) Accept(ctx context.Context) (Session, error) {
	return s.accept(ctx)
}

func (s *baseServer) accept(ctx context.Context) (quicSession, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case sess := <-s.sessionQueue:
		atomic.AddInt32(&s.sessionQueueLen, -1)
		return sess, nil
	case <-s.errorChan:
		return nil, s.serverError
	}
}

// Close the server
func (s *baseServer) Close() error {
	s.mutex.Lock()
	if s.closed {
		s.mutex.Unlock()
		return nil
	}
	if s.serverError == nil {
		s.serverError = errors.New("server closed")
	}
	// If the server was started with ListenAddr, we created the packet conn.
	// We need to close it in order to make the go routine reading from that conn return.
	createdPacketConn := s.createdPacketConn
	s.closed = true
	close(s.errorChan)
	s.mutex.Unlock()

	<-s.running
	s.sessionHandler.CloseServer()
	if createdPacketConn {
		return s.sessionHandler.Destroy()
	}
	return nil
}

func (s *baseServer) setCloseError(e error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	s.serverError = e
	close(s.errorChan)
}

// Addr returns the server's network address
func (s *baseServer) Addr() net.Addr {
	return s.conn.LocalAddr()
}

func (s *baseServer) handlePacket(p *receivedPacket) {
	select {
	case s.receivedPackets <- p:
	default:
		s.logger.Debugf("Dropping packet from %s (%d bytes). Server receive queue full.", p.remoteAddr, p.Size())
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeNotDetermined, p.Size(), logging.PacketDropDOSPrevention)
		}
	}
}

func (s *baseServer) handlePacketImpl(p *receivedPacket) bool /* is the buffer still in use? */ {
	if wire.IsVersionNegotiationPacket(p.data) {
		s.logger.Debugf("Dropping Version Negotiation packet.")
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeVersionNegotiation, p.Size(), logging.PacketDropUnexpectedPacket)
		}
		return false
	}
	// If we're creating a new session, the packet will be passed to the session.
	// The header will then be parsed again.
	hdr, _, _, err := wire.ParsePacket(p.data, s.config.ConnectionIDLength)
	if err != nil && err != wire.ErrUnsupportedVersion {
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeNotDetermined, p.Size(), logging.PacketDropHeaderParseError)
		}
		s.logger.Debugf("Error parsing packet: %s", err)
		return false
	}
	// Short header packets should never end up here in the first place
	if !hdr.IsLongHeader {
		panic(fmt.Sprintf("misrouted packet: %#v", hdr))
	}
	if hdr.Type == protocol.PacketTypeInitial && p.Size() < protocol.MinInitialPacketSize {
		s.logger.Debugf("Dropping a packet that is too small to be a valid Initial (%d bytes)", p.Size())
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeInitial, p.Size(), logging.PacketDropUnexpectedPacket)
		}
		return false
	}
	// send a Version Negotiation Packet if the client is speaking a different protocol version
	if !protocol.IsSupportedVersion(s.config.Versions, hdr.Version) {
		if p.Size() < protocol.MinUnknownVersionPacketSize {
			s.logger.Debugf("Dropping a packet with an unknown version that is too small (%d bytes)", p.Size())
			if s.config.Tracer != nil {
				s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeNotDetermined, p.Size(), logging.PacketDropUnexpectedPacket)
			}
			return false
		}
		if !s.config.DisableVersionNegotiationPackets {
			go s.sendVersionNegotiationPacket(p, hdr)
		}
		return false
	}
	if hdr.IsLongHeader && hdr.Type != protocol.PacketTypeInitial {
		// Drop long header packets.
		// There's little point in sending a Stateless Reset, since the client
		// might not have received the token yet.
		s.logger.Debugf("Dropping long header packet of type %s (%d bytes)", hdr.Type, len(p.data))
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeFromHeader(hdr), p.Size(), logging.PacketDropUnexpectedPacket)
		}
		return false
	}

	s.logger.Debugf("<- Received Initial packet.")

	if err := s.handleInitialImpl(p, hdr); err != nil {
		s.logger.Errorf("Error occurred handling initial packet: %s", err)
	}
	// Don't put the packet buffer back.
	// handleInitialImpl deals with the buffer.
	return true
}

func (s *baseServer) handleInitialImpl(p *receivedPacket, hdr *wire.Header) error {
	if len(hdr.Token) == 0 && hdr.DestConnectionID.Len() < protocol.MinConnectionIDLenInitial {
		p.buffer.Release()
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeInitial, p.Size(), logging.PacketDropUnexpectedPacket)
		}
		return errors.New("too short connection ID")
	}

	var (
		token          *Token
		retrySrcConnID *protocol.ConnectionID
	)
	origDestConnID := hdr.DestConnectionID
	if len(hdr.Token) > 0 {
		c, err := s.tokenGenerator.DecodeToken(hdr.Token)
		if err == nil {
			token = &Token{
				IsRetryToken: c.IsRetryToken,
				RemoteAddr:   c.RemoteAddr,
				SentTime:     c.SentTime,
			}
			if token.IsRetryToken {
				origDestConnID = c.OriginalDestConnectionID
				retrySrcConnID = &c.RetrySrcConnectionID
			}
		}
	}
	if !s.config.AcceptToken(p.remoteAddr, token) {
		go func() {
			defer p.buffer.Release()
			if token != nil && token.IsRetryToken {
				if err := s.maybeSendInvalidToken(p, hdr); err != nil {
					s.logger.Debugf("Error sending INVALID_TOKEN error: %s", err)
				}
				return
			}
			if err := s.sendRetry(p.remoteAddr, hdr, p.info); err != nil {
				s.logger.Debugf("Error sending Retry: %s", err)
			}
		}()
		return nil
	}

	if queueLen := atomic.LoadInt32(&s.sessionQueueLen); queueLen >= protocol.MaxAcceptQueueSize {
		s.logger.Debugf("Rejecting new connection. Server currently busy. Accept queue length: %d (max %d)", queueLen, protocol.MaxAcceptQueueSize)
		go func() {
			defer p.buffer.Release()
			if err := s.sendConnectionRefused(p.remoteAddr, hdr, p.info); err != nil {
				s.logger.Debugf("Error rejecting connection: %s", err)
			}
		}()
		return nil
	}

	connID, err := protocol.GenerateConnectionID(s.config.ConnectionIDLength)
	if err != nil {
		return err
	}
	s.logger.Debugf("Changing connection ID to %s.", connID)
	var sess quicSession
	tracingID := nextSessionTracingID()
	if added := s.sessionHandler.AddWithConnID(hdr.DestConnectionID, connID, func() packetHandler {
		var tracer logging.ConnectionTracer
		if s.config.Tracer != nil {
			// Use the same connection ID that is passed to the client's GetLogWriter callback.
			connID := hdr.DestConnectionID
			if origDestConnID.Len() > 0 {
				connID = origDestConnID
			}
			tracer = s.config.Tracer.TracerForConnection(
				context.WithValue(context.Background(), SessionTracingKey, tracingID),
				protocol.PerspectiveServer,
				connID,
			)
		}
		sess = s.newSession(
			newSendConn(s.conn, p.remoteAddr, p.info),
			s.sessionHandler,
			origDestConnID,
			retrySrcConnID,
			hdr.DestConnectionID,
			hdr.SrcConnectionID,
			connID,
			s.sessionHandler.GetStatelessResetToken(connID),
			s.config,
			s.tlsConf,
			s.tokenGenerator,
			s.acceptEarlySessions,
			tracer,
			tracingID,
			s.logger,
			hdr.Version,
		)
		sess.handlePacket(p)
		return sess
	}); !added {
		return nil
	}
	go sess.run()
	go s.handleNewSession(sess)
	if sess == nil {
		p.buffer.Release()
		return nil
	}
	return nil
}

func (s *baseServer) handleNewSession(sess quicSession) {
	sessCtx := sess.Context()
	if s.acceptEarlySessions {
		// wait until the early session is ready (or the handshake fails)
		select {
		case <-sess.earlySessionReady():
		case <-sessCtx.Done():
			return
		}
	} else {
		// wait until the handshake is complete (or fails)
		select {
		case <-sess.HandshakeComplete().Done():
		case <-sessCtx.Done():
			return
		}
	}

	atomic.AddInt32(&s.sessionQueueLen, 1)
	select {
	case s.sessionQueue <- sess:
		// blocks until the session is accepted
	case <-sessCtx.Done():
		atomic.AddInt32(&s.sessionQueueLen, -1)
		// don't pass sessions that were already closed to Accept()
	}
}

func (s *baseServer) sendRetry(remoteAddr net.Addr, hdr *wire.Header, info *packetInfo) error {
	// Log the Initial packet now.
	// If no Retry is sent, the packet will be logged by the session.
	(&wire.ExtendedHeader{Header: *hdr}).Log(s.logger)
	srcConnID, err := protocol.GenerateConnectionID(s.config.ConnectionIDLength)
	if err != nil {
		return err
	}
	token, err := s.tokenGenerator.NewRetryToken(remoteAddr, hdr.DestConnectionID, srcConnID)
	if err != nil {
		return err
	}
	replyHdr := &wire.ExtendedHeader{}
	replyHdr.IsLongHeader = true
	replyHdr.Type = protocol.PacketTypeRetry
	replyHdr.Version = hdr.Version
	replyHdr.SrcConnectionID = srcConnID
	replyHdr.DestConnectionID = hdr.SrcConnectionID
	replyHdr.Token = token
	if s.logger.Debug() {
		s.logger.Debugf("Changing connection ID to %s.", srcConnID)
		s.logger.Debugf("-> Sending Retry")
		replyHdr.Log(s.logger)
	}

	packetBuffer := getPacketBuffer()
	defer packetBuffer.Release()
	buf := bytes.NewBuffer(packetBuffer.Data)
	if err := replyHdr.Write(buf, hdr.Version); err != nil {
		return err
	}
	// append the Retry integrity tag
	tag := handshake.GetRetryIntegrityTag(buf.Bytes(), hdr.DestConnectionID, hdr.Version)
	buf.Write(tag[:])
	if s.config.Tracer != nil {
		s.config.Tracer.SentPacket(remoteAddr, &replyHdr.Header, protocol.ByteCount(buf.Len()), nil)
	}
	_, err = s.conn.WritePacket(buf.Bytes(), remoteAddr, info.OOB())
	return err
}

func (s *baseServer) maybeSendInvalidToken(p *receivedPacket, hdr *wire.Header) error {
	// Only send INVALID_TOKEN if we can unprotect the packet.
	// This makes sure that we won't send it for packets that were corrupted.
	sealer, opener := handshake.NewInitialAEAD(hdr.DestConnectionID, protocol.PerspectiveServer, hdr.Version)
	data := p.data[:hdr.ParsedLen()+hdr.Length]
	extHdr, err := unpackHeader(opener, hdr, data, hdr.Version)
	if err != nil {
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeInitial, p.Size(), logging.PacketDropHeaderParseError)
		}
		// don't return the error here. Just drop the packet.
		return nil
	}
	hdrLen := extHdr.ParsedLen()
	if _, err := opener.Open(data[hdrLen:hdrLen], data[hdrLen:], extHdr.PacketNumber, data[:hdrLen]); err != nil {
		// don't return the error here. Just drop the packet.
		if s.config.Tracer != nil {
			s.config.Tracer.DroppedPacket(p.remoteAddr, logging.PacketTypeInitial, p.Size(), logging.PacketDropPayloadDecryptError)
		}
		return nil
	}
	if s.logger.Debug() {
		s.logger.Debugf("Client sent an invalid retry token. Sending INVALID_TOKEN to %s.", p.remoteAddr)
	}
	return s.sendError(p.remoteAddr, hdr, sealer, qerr.InvalidToken, p.info)
}

func (s *baseServer) sendConnectionRefused(remoteAddr net.Addr, hdr *wire.Header, info *packetInfo) error {
	sealer, _ := handshake.NewInitialAEAD(hdr.DestConnectionID, protocol.PerspectiveServer, hdr.Version)
	return s.sendError(remoteAddr, hdr, sealer, qerr.ConnectionRefused, info)
}

// sendError sends the error as a response to the packet received with header hdr
func (s *baseServer) sendError(remoteAddr net.Addr, hdr *wire.Header, sealer handshake.LongHeaderSealer, errorCode qerr.TransportErrorCode, info *packetInfo) error {
	packetBuffer := getPacketBuffer()
	defer packetBuffer.Release()
	buf := bytes.NewBuffer(packetBuffer.Data)

	ccf := &wire.ConnectionCloseFrame{ErrorCode: uint64(errorCode)}

	replyHdr := &wire.ExtendedHeader{}
	replyHdr.IsLongHeader = true
	replyHdr.Type = protocol.PacketTypeInitial
	replyHdr.Version = hdr.Version
	replyHdr.SrcConnectionID = hdr.DestConnectionID
	replyHdr.DestConnectionID = hdr.SrcConnectionID
	replyHdr.PacketNumberLen = protocol.PacketNumberLen4
	replyHdr.Length = 4 /* packet number len */ + ccf.Length(hdr.Version) + protocol.ByteCount(sealer.Overhead())
	if err := replyHdr.Write(buf, hdr.Version); err != nil {
		return err
	}
	payloadOffset := buf.Len()

	if err := ccf.Write(buf, hdr.Version); err != nil {
		return err
	}

	raw := buf.Bytes()
	_ = sealer.Seal(raw[payloadOffset:payloadOffset], raw[payloadOffset:], replyHdr.PacketNumber, raw[:payloadOffset])
	raw = raw[0 : buf.Len()+sealer.Overhead()]

	pnOffset := payloadOffset - int(replyHdr.PacketNumberLen)
	sealer.EncryptHeader(
		raw[pnOffset+4:pnOffset+4+16],
		&raw[0],
		raw[pnOffset:payloadOffset],
	)

	replyHdr.Log(s.logger)
	wire.LogFrame(s.logger, ccf, true)
	if s.config.Tracer != nil {
		s.config.Tracer.SentPacket(remoteAddr, &replyHdr.Header, protocol.ByteCount(len(raw)), []logging.Frame{ccf})
	}
	_, err := s.conn.WritePacket(raw, remoteAddr, info.OOB())
	return err
}

func (s *baseServer) sendVersionNegotiationPacket(p *receivedPacket, hdr *wire.Header) {
	s.logger.Debugf("Client offered version %s, sending Version Negotiation", hdr.Version)
	data, err := wire.ComposeVersionNegotiation(hdr.SrcConnectionID, hdr.DestConnectionID, s.config.Versions)
	if err != nil {
		s.logger.Debugf("Error composing Version Negotiation: %s", err)
		return
	}
	if s.config.Tracer != nil {
		s.config.Tracer.SentPacket(
			p.remoteAddr,
			&wire.Header{
				IsLongHeader:     true,
				DestConnectionID: hdr.SrcConnectionID,
				SrcConnectionID:  hdr.DestConnectionID,
			},
			protocol.ByteCount(len(data)),
			nil,
		)
	}
	if _, err := s.conn.WritePacket(data, p.remoteAddr, p.info.OOB()); err != nil {
		s.logger.Debugf("Error sending Version Negotiation: %s", err)
	}
}
