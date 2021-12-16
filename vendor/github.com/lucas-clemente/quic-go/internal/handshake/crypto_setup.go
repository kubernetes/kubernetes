package handshake

import (
	"bytes"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/qtls"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
	"github.com/lucas-clemente/quic-go/logging"
	"github.com/lucas-clemente/quic-go/quicvarint"
)

// TLS unexpected_message alert
const alertUnexpectedMessage uint8 = 10

type messageType uint8

// TLS handshake message types.
const (
	typeClientHello         messageType = 1
	typeServerHello         messageType = 2
	typeNewSessionTicket    messageType = 4
	typeEncryptedExtensions messageType = 8
	typeCertificate         messageType = 11
	typeCertificateRequest  messageType = 13
	typeCertificateVerify   messageType = 15
	typeFinished            messageType = 20
)

func (m messageType) String() string {
	switch m {
	case typeClientHello:
		return "ClientHello"
	case typeServerHello:
		return "ServerHello"
	case typeNewSessionTicket:
		return "NewSessionTicket"
	case typeEncryptedExtensions:
		return "EncryptedExtensions"
	case typeCertificate:
		return "Certificate"
	case typeCertificateRequest:
		return "CertificateRequest"
	case typeCertificateVerify:
		return "CertificateVerify"
	case typeFinished:
		return "Finished"
	default:
		return fmt.Sprintf("unknown message type: %d", m)
	}
}

const clientSessionStateRevision = 3

type conn struct {
	localAddr, remoteAddr net.Addr
	version               protocol.VersionNumber
}

var _ ConnWithVersion = &conn{}

func newConn(local, remote net.Addr, version protocol.VersionNumber) ConnWithVersion {
	return &conn{
		localAddr:  local,
		remoteAddr: remote,
		version:    version,
	}
}

var _ net.Conn = &conn{}

func (c *conn) Read([]byte) (int, error)               { return 0, nil }
func (c *conn) Write([]byte) (int, error)              { return 0, nil }
func (c *conn) Close() error                           { return nil }
func (c *conn) RemoteAddr() net.Addr                   { return c.remoteAddr }
func (c *conn) LocalAddr() net.Addr                    { return c.localAddr }
func (c *conn) SetReadDeadline(time.Time) error        { return nil }
func (c *conn) SetWriteDeadline(time.Time) error       { return nil }
func (c *conn) SetDeadline(time.Time) error            { return nil }
func (c *conn) GetQUICVersion() protocol.VersionNumber { return c.version }

type cryptoSetup struct {
	tlsConf   *tls.Config
	extraConf *qtls.ExtraConfig
	conn      *qtls.Conn

	version protocol.VersionNumber

	messageChan               chan []byte
	isReadingHandshakeMessage chan struct{}
	readFirstHandshakeMessage bool

	ourParams  *wire.TransportParameters
	peerParams *wire.TransportParameters
	paramsChan <-chan []byte

	runner handshakeRunner

	alertChan chan uint8
	// handshakeDone is closed as soon as the go routine running qtls.Handshake() returns
	handshakeDone chan struct{}
	// is closed when Close() is called
	closeChan chan struct{}

	zeroRTTParameters      *wire.TransportParameters
	clientHelloWritten     bool
	clientHelloWrittenChan chan *wire.TransportParameters

	rttStats *utils.RTTStats

	tracer logging.ConnectionTracer
	logger utils.Logger

	perspective protocol.Perspective

	mutex sync.Mutex // protects all members below

	handshakeCompleteTime time.Time

	readEncLevel  protocol.EncryptionLevel
	writeEncLevel protocol.EncryptionLevel

	zeroRTTOpener LongHeaderOpener // only set for the server
	zeroRTTSealer LongHeaderSealer // only set for the client

	initialStream io.Writer
	initialOpener LongHeaderOpener
	initialSealer LongHeaderSealer

	handshakeStream io.Writer
	handshakeOpener LongHeaderOpener
	handshakeSealer LongHeaderSealer

	aead          *updatableAEAD
	has1RTTSealer bool
	has1RTTOpener bool
}

var (
	_ qtls.RecordLayer = &cryptoSetup{}
	_ CryptoSetup      = &cryptoSetup{}
)

// NewCryptoSetupClient creates a new crypto setup for the client
func NewCryptoSetupClient(
	initialStream io.Writer,
	handshakeStream io.Writer,
	connID protocol.ConnectionID,
	localAddr net.Addr,
	remoteAddr net.Addr,
	tp *wire.TransportParameters,
	runner handshakeRunner,
	tlsConf *tls.Config,
	enable0RTT bool,
	rttStats *utils.RTTStats,
	tracer logging.ConnectionTracer,
	logger utils.Logger,
	version protocol.VersionNumber,
) (CryptoSetup, <-chan *wire.TransportParameters /* ClientHello written. Receive nil for non-0-RTT */) {
	cs, clientHelloWritten := newCryptoSetup(
		initialStream,
		handshakeStream,
		connID,
		tp,
		runner,
		tlsConf,
		enable0RTT,
		rttStats,
		tracer,
		logger,
		protocol.PerspectiveClient,
		version,
	)
	cs.conn = qtls.Client(newConn(localAddr, remoteAddr, version), cs.tlsConf, cs.extraConf)
	return cs, clientHelloWritten
}

// NewCryptoSetupServer creates a new crypto setup for the server
func NewCryptoSetupServer(
	initialStream io.Writer,
	handshakeStream io.Writer,
	connID protocol.ConnectionID,
	localAddr net.Addr,
	remoteAddr net.Addr,
	tp *wire.TransportParameters,
	runner handshakeRunner,
	tlsConf *tls.Config,
	enable0RTT bool,
	rttStats *utils.RTTStats,
	tracer logging.ConnectionTracer,
	logger utils.Logger,
	version protocol.VersionNumber,
) CryptoSetup {
	cs, _ := newCryptoSetup(
		initialStream,
		handshakeStream,
		connID,
		tp,
		runner,
		tlsConf,
		enable0RTT,
		rttStats,
		tracer,
		logger,
		protocol.PerspectiveServer,
		version,
	)
	cs.conn = qtls.Server(newConn(localAddr, remoteAddr, version), cs.tlsConf, cs.extraConf)
	return cs
}

func newCryptoSetup(
	initialStream io.Writer,
	handshakeStream io.Writer,
	connID protocol.ConnectionID,
	tp *wire.TransportParameters,
	runner handshakeRunner,
	tlsConf *tls.Config,
	enable0RTT bool,
	rttStats *utils.RTTStats,
	tracer logging.ConnectionTracer,
	logger utils.Logger,
	perspective protocol.Perspective,
	version protocol.VersionNumber,
) (*cryptoSetup, <-chan *wire.TransportParameters /* ClientHello written. Receive nil for non-0-RTT */) {
	initialSealer, initialOpener := NewInitialAEAD(connID, perspective, version)
	if tracer != nil {
		tracer.UpdatedKeyFromTLS(protocol.EncryptionInitial, protocol.PerspectiveClient)
		tracer.UpdatedKeyFromTLS(protocol.EncryptionInitial, protocol.PerspectiveServer)
	}
	extHandler := newExtensionHandler(tp.Marshal(perspective), perspective, version)
	cs := &cryptoSetup{
		tlsConf:                   tlsConf,
		initialStream:             initialStream,
		initialSealer:             initialSealer,
		initialOpener:             initialOpener,
		handshakeStream:           handshakeStream,
		aead:                      newUpdatableAEAD(rttStats, tracer, logger),
		readEncLevel:              protocol.EncryptionInitial,
		writeEncLevel:             protocol.EncryptionInitial,
		runner:                    runner,
		ourParams:                 tp,
		paramsChan:                extHandler.TransportParameters(),
		rttStats:                  rttStats,
		tracer:                    tracer,
		logger:                    logger,
		perspective:               perspective,
		handshakeDone:             make(chan struct{}),
		alertChan:                 make(chan uint8),
		clientHelloWrittenChan:    make(chan *wire.TransportParameters, 1),
		messageChan:               make(chan []byte, 100),
		isReadingHandshakeMessage: make(chan struct{}),
		closeChan:                 make(chan struct{}),
		version:                   version,
	}
	var maxEarlyData uint32
	if enable0RTT {
		maxEarlyData = 0xffffffff
	}
	cs.extraConf = &qtls.ExtraConfig{
		GetExtensions:              extHandler.GetExtensions,
		ReceivedExtensions:         extHandler.ReceivedExtensions,
		AlternativeRecordLayer:     cs,
		EnforceNextProtoSelection:  true,
		MaxEarlyData:               maxEarlyData,
		Accept0RTT:                 cs.accept0RTT,
		Rejected0RTT:               cs.rejected0RTT,
		Enable0RTT:                 enable0RTT,
		GetAppDataForSessionState:  cs.marshalDataForSessionState,
		SetAppDataFromSessionState: cs.handleDataFromSessionState,
	}
	return cs, cs.clientHelloWrittenChan
}

func (h *cryptoSetup) ChangeConnectionID(id protocol.ConnectionID) {
	initialSealer, initialOpener := NewInitialAEAD(id, h.perspective, h.version)
	h.initialSealer = initialSealer
	h.initialOpener = initialOpener
	if h.tracer != nil {
		h.tracer.UpdatedKeyFromTLS(protocol.EncryptionInitial, protocol.PerspectiveClient)
		h.tracer.UpdatedKeyFromTLS(protocol.EncryptionInitial, protocol.PerspectiveServer)
	}
}

func (h *cryptoSetup) SetLargest1RTTAcked(pn protocol.PacketNumber) error {
	return h.aead.SetLargestAcked(pn)
}

func (h *cryptoSetup) RunHandshake() {
	// Handle errors that might occur when HandleData() is called.
	handshakeComplete := make(chan struct{})
	handshakeErrChan := make(chan error, 1)
	go func() {
		defer close(h.handshakeDone)
		if err := h.conn.Handshake(); err != nil {
			handshakeErrChan <- err
			return
		}
		close(handshakeComplete)
	}()

	select {
	case <-handshakeComplete: // return when the handshake is done
		h.mutex.Lock()
		h.handshakeCompleteTime = time.Now()
		h.mutex.Unlock()
		h.runner.OnHandshakeComplete()
	case <-h.closeChan:
		// wait until the Handshake() go routine has returned
		<-h.handshakeDone
	case alert := <-h.alertChan:
		handshakeErr := <-handshakeErrChan
		h.onError(alert, handshakeErr.Error())
	}
}

func (h *cryptoSetup) onError(alert uint8, message string) {
	h.runner.OnError(qerr.NewCryptoError(alert, message))
}

// Close closes the crypto setup.
// It aborts the handshake, if it is still running.
// It must only be called once.
func (h *cryptoSetup) Close() error {
	close(h.closeChan)
	// wait until qtls.Handshake() actually returned
	<-h.handshakeDone
	return nil
}

// handleMessage handles a TLS handshake message.
// It is called by the crypto streams when a new message is available.
// It returns if it is done with messages on the same encryption level.
func (h *cryptoSetup) HandleMessage(data []byte, encLevel protocol.EncryptionLevel) bool /* stream finished */ {
	msgType := messageType(data[0])
	h.logger.Debugf("Received %s message (%d bytes, encryption level: %s)", msgType, len(data), encLevel)
	if err := h.checkEncryptionLevel(msgType, encLevel); err != nil {
		h.onError(alertUnexpectedMessage, err.Error())
		return false
	}
	h.messageChan <- data
	if encLevel == protocol.Encryption1RTT {
		h.handlePostHandshakeMessage()
		return false
	}
readLoop:
	for {
		select {
		case data := <-h.paramsChan:
			if data == nil {
				h.onError(0x6d, "missing quic_transport_parameters extension")
			} else {
				h.handleTransportParameters(data)
			}
		case <-h.isReadingHandshakeMessage:
			break readLoop
		case <-h.handshakeDone:
			break readLoop
		case <-h.closeChan:
			break readLoop
		}
	}
	// We're done with the Initial encryption level after processing a ClientHello / ServerHello,
	// but only if a handshake opener and sealer was created.
	// Otherwise, a HelloRetryRequest was performed.
	// We're done with the Handshake encryption level after processing the Finished message.
	return ((msgType == typeClientHello || msgType == typeServerHello) && h.handshakeOpener != nil && h.handshakeSealer != nil) ||
		msgType == typeFinished
}

func (h *cryptoSetup) checkEncryptionLevel(msgType messageType, encLevel protocol.EncryptionLevel) error {
	var expected protocol.EncryptionLevel
	switch msgType {
	case typeClientHello,
		typeServerHello:
		expected = protocol.EncryptionInitial
	case typeEncryptedExtensions,
		typeCertificate,
		typeCertificateRequest,
		typeCertificateVerify,
		typeFinished:
		expected = protocol.EncryptionHandshake
	case typeNewSessionTicket:
		expected = protocol.Encryption1RTT
	default:
		return fmt.Errorf("unexpected handshake message: %d", msgType)
	}
	if encLevel != expected {
		return fmt.Errorf("expected handshake message %s to have encryption level %s, has %s", msgType, expected, encLevel)
	}
	return nil
}

func (h *cryptoSetup) handleTransportParameters(data []byte) {
	var tp wire.TransportParameters
	if err := tp.Unmarshal(data, h.perspective.Opposite()); err != nil {
		h.runner.OnError(&qerr.TransportError{
			ErrorCode:    qerr.TransportParameterError,
			ErrorMessage: err.Error(),
		})
	}
	h.peerParams = &tp
	h.runner.OnReceivedParams(h.peerParams)
}

// must be called after receiving the transport parameters
func (h *cryptoSetup) marshalDataForSessionState() []byte {
	buf := &bytes.Buffer{}
	quicvarint.Write(buf, clientSessionStateRevision)
	quicvarint.Write(buf, uint64(h.rttStats.SmoothedRTT().Microseconds()))
	h.peerParams.MarshalForSessionTicket(buf)
	return buf.Bytes()
}

func (h *cryptoSetup) handleDataFromSessionState(data []byte) {
	tp, err := h.handleDataFromSessionStateImpl(data)
	if err != nil {
		h.logger.Debugf("Restoring of transport parameters from session ticket failed: %s", err.Error())
		return
	}
	h.zeroRTTParameters = tp
}

func (h *cryptoSetup) handleDataFromSessionStateImpl(data []byte) (*wire.TransportParameters, error) {
	r := bytes.NewReader(data)
	ver, err := quicvarint.Read(r)
	if err != nil {
		return nil, err
	}
	if ver != clientSessionStateRevision {
		return nil, fmt.Errorf("mismatching version. Got %d, expected %d", ver, clientSessionStateRevision)
	}
	rtt, err := quicvarint.Read(r)
	if err != nil {
		return nil, err
	}
	h.rttStats.SetInitialRTT(time.Duration(rtt) * time.Microsecond)
	var tp wire.TransportParameters
	if err := tp.UnmarshalFromSessionTicket(r); err != nil {
		return nil, err
	}
	return &tp, nil
}

// only valid for the server
func (h *cryptoSetup) GetSessionTicket() ([]byte, error) {
	var appData []byte
	// Save transport parameters to the session ticket if we're allowing 0-RTT.
	if h.extraConf.MaxEarlyData > 0 {
		appData = (&sessionTicket{
			Parameters: h.ourParams,
			RTT:        h.rttStats.SmoothedRTT(),
		}).Marshal()
	}
	return h.conn.GetSessionTicket(appData)
}

// accept0RTT is called for the server when receiving the client's session ticket.
// It decides whether to accept 0-RTT.
func (h *cryptoSetup) accept0RTT(sessionTicketData []byte) bool {
	var t sessionTicket
	if err := t.Unmarshal(sessionTicketData); err != nil {
		h.logger.Debugf("Unmarshalling transport parameters from session ticket failed: %s", err.Error())
		return false
	}
	valid := h.ourParams.ValidFor0RTT(t.Parameters)
	if valid {
		h.logger.Debugf("Accepting 0-RTT. Restoring RTT from session ticket: %s", t.RTT)
		h.rttStats.SetInitialRTT(t.RTT)
	} else {
		h.logger.Debugf("Transport parameters changed. Rejecting 0-RTT.")
	}
	return valid
}

// rejected0RTT is called for the client when the server rejects 0-RTT.
func (h *cryptoSetup) rejected0RTT() {
	h.logger.Debugf("0-RTT was rejected. Dropping 0-RTT keys.")

	h.mutex.Lock()
	had0RTTKeys := h.zeroRTTSealer != nil
	h.zeroRTTSealer = nil
	h.mutex.Unlock()

	if had0RTTKeys {
		h.runner.DropKeys(protocol.Encryption0RTT)
	}
}

func (h *cryptoSetup) handlePostHandshakeMessage() {
	// make sure the handshake has already completed
	<-h.handshakeDone

	done := make(chan struct{})
	defer close(done)

	// h.alertChan is an unbuffered channel.
	// If an error occurs during conn.HandlePostHandshakeMessage,
	// it will be sent on this channel.
	// Read it from a go-routine so that HandlePostHandshakeMessage doesn't deadlock.
	alertChan := make(chan uint8, 1)
	go func() {
		<-h.isReadingHandshakeMessage
		select {
		case alert := <-h.alertChan:
			alertChan <- alert
		case <-done:
		}
	}()

	if err := h.conn.HandlePostHandshakeMessage(); err != nil {
		select {
		case <-h.closeChan:
		case alert := <-alertChan:
			h.onError(alert, err.Error())
		}
	}
}

// ReadHandshakeMessage is called by TLS.
// It blocks until a new handshake message is available.
func (h *cryptoSetup) ReadHandshakeMessage() ([]byte, error) {
	if !h.readFirstHandshakeMessage {
		h.readFirstHandshakeMessage = true
	} else {
		select {
		case h.isReadingHandshakeMessage <- struct{}{}:
		case <-h.closeChan:
			return nil, errors.New("error while handling the handshake message")
		}
	}
	select {
	case msg := <-h.messageChan:
		return msg, nil
	case <-h.closeChan:
		return nil, errors.New("error while handling the handshake message")
	}
}

func (h *cryptoSetup) SetReadKey(encLevel qtls.EncryptionLevel, suite *qtls.CipherSuiteTLS13, trafficSecret []byte) {
	h.mutex.Lock()
	switch encLevel {
	case qtls.Encryption0RTT:
		if h.perspective == protocol.PerspectiveClient {
			panic("Received 0-RTT read key for the client")
		}
		h.zeroRTTOpener = newLongHeaderOpener(
			createAEAD(suite, trafficSecret),
			newHeaderProtector(suite, trafficSecret, true),
		)
		h.mutex.Unlock()
		h.logger.Debugf("Installed 0-RTT Read keys (using %s)", tls.CipherSuiteName(suite.ID))
		if h.tracer != nil {
			h.tracer.UpdatedKeyFromTLS(protocol.Encryption0RTT, h.perspective.Opposite())
		}
		return
	case qtls.EncryptionHandshake:
		h.readEncLevel = protocol.EncryptionHandshake
		h.handshakeOpener = newHandshakeOpener(
			createAEAD(suite, trafficSecret),
			newHeaderProtector(suite, trafficSecret, true),
			h.dropInitialKeys,
			h.perspective,
		)
		h.logger.Debugf("Installed Handshake Read keys (using %s)", tls.CipherSuiteName(suite.ID))
	case qtls.EncryptionApplication:
		h.readEncLevel = protocol.Encryption1RTT
		h.aead.SetReadKey(suite, trafficSecret)
		h.has1RTTOpener = true
		h.logger.Debugf("Installed 1-RTT Read keys (using %s)", tls.CipherSuiteName(suite.ID))
	default:
		panic("unexpected read encryption level")
	}
	h.mutex.Unlock()
	if h.tracer != nil {
		h.tracer.UpdatedKeyFromTLS(h.readEncLevel, h.perspective.Opposite())
	}
}

func (h *cryptoSetup) SetWriteKey(encLevel qtls.EncryptionLevel, suite *qtls.CipherSuiteTLS13, trafficSecret []byte) {
	h.mutex.Lock()
	switch encLevel {
	case qtls.Encryption0RTT:
		if h.perspective == protocol.PerspectiveServer {
			panic("Received 0-RTT write key for the server")
		}
		h.zeroRTTSealer = newLongHeaderSealer(
			createAEAD(suite, trafficSecret),
			newHeaderProtector(suite, trafficSecret, true),
		)
		h.mutex.Unlock()
		h.logger.Debugf("Installed 0-RTT Write keys (using %s)", tls.CipherSuiteName(suite.ID))
		if h.tracer != nil {
			h.tracer.UpdatedKeyFromTLS(protocol.Encryption0RTT, h.perspective)
		}
		return
	case qtls.EncryptionHandshake:
		h.writeEncLevel = protocol.EncryptionHandshake
		h.handshakeSealer = newHandshakeSealer(
			createAEAD(suite, trafficSecret),
			newHeaderProtector(suite, trafficSecret, true),
			h.dropInitialKeys,
			h.perspective,
		)
		h.logger.Debugf("Installed Handshake Write keys (using %s)", tls.CipherSuiteName(suite.ID))
	case qtls.EncryptionApplication:
		h.writeEncLevel = protocol.Encryption1RTT
		h.aead.SetWriteKey(suite, trafficSecret)
		h.has1RTTSealer = true
		h.logger.Debugf("Installed 1-RTT Write keys (using %s)", tls.CipherSuiteName(suite.ID))
		if h.zeroRTTSealer != nil {
			h.zeroRTTSealer = nil
			h.logger.Debugf("Dropping 0-RTT keys.")
			if h.tracer != nil {
				h.tracer.DroppedEncryptionLevel(protocol.Encryption0RTT)
			}
		}
	default:
		panic("unexpected write encryption level")
	}
	h.mutex.Unlock()
	if h.tracer != nil {
		h.tracer.UpdatedKeyFromTLS(h.writeEncLevel, h.perspective)
	}
}

// WriteRecord is called when TLS writes data
func (h *cryptoSetup) WriteRecord(p []byte) (int, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	//nolint:exhaustive // LS records can only be written for Initial and Handshake.
	switch h.writeEncLevel {
	case protocol.EncryptionInitial:
		// assume that the first WriteRecord call contains the ClientHello
		n, err := h.initialStream.Write(p)
		if !h.clientHelloWritten && h.perspective == protocol.PerspectiveClient {
			h.clientHelloWritten = true
			if h.zeroRTTSealer != nil && h.zeroRTTParameters != nil {
				h.logger.Debugf("Doing 0-RTT.")
				h.clientHelloWrittenChan <- h.zeroRTTParameters
			} else {
				h.logger.Debugf("Not doing 0-RTT.")
				h.clientHelloWrittenChan <- nil
			}
		}
		return n, err
	case protocol.EncryptionHandshake:
		return h.handshakeStream.Write(p)
	default:
		panic(fmt.Sprintf("unexpected write encryption level: %s", h.writeEncLevel))
	}
}

func (h *cryptoSetup) SendAlert(alert uint8) {
	select {
	case h.alertChan <- alert:
	case <-h.closeChan:
		// no need to send an alert when we've already closed
	}
}

// used a callback in the handshakeSealer and handshakeOpener
func (h *cryptoSetup) dropInitialKeys() {
	h.mutex.Lock()
	h.initialOpener = nil
	h.initialSealer = nil
	h.mutex.Unlock()
	h.runner.DropKeys(protocol.EncryptionInitial)
	h.logger.Debugf("Dropping Initial keys.")
}

func (h *cryptoSetup) SetHandshakeConfirmed() {
	h.aead.SetHandshakeConfirmed()
	// drop Handshake keys
	var dropped bool
	h.mutex.Lock()
	if h.handshakeOpener != nil {
		h.handshakeOpener = nil
		h.handshakeSealer = nil
		dropped = true
	}
	h.mutex.Unlock()
	if dropped {
		h.runner.DropKeys(protocol.EncryptionHandshake)
		h.logger.Debugf("Dropping Handshake keys.")
	}
}

func (h *cryptoSetup) GetInitialSealer() (LongHeaderSealer, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.initialSealer == nil {
		return nil, ErrKeysDropped
	}
	return h.initialSealer, nil
}

func (h *cryptoSetup) Get0RTTSealer() (LongHeaderSealer, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.zeroRTTSealer == nil {
		return nil, ErrKeysDropped
	}
	return h.zeroRTTSealer, nil
}

func (h *cryptoSetup) GetHandshakeSealer() (LongHeaderSealer, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.handshakeSealer == nil {
		if h.initialSealer == nil {
			return nil, ErrKeysDropped
		}
		return nil, ErrKeysNotYetAvailable
	}
	return h.handshakeSealer, nil
}

func (h *cryptoSetup) Get1RTTSealer() (ShortHeaderSealer, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if !h.has1RTTSealer {
		return nil, ErrKeysNotYetAvailable
	}
	return h.aead, nil
}

func (h *cryptoSetup) GetInitialOpener() (LongHeaderOpener, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.initialOpener == nil {
		return nil, ErrKeysDropped
	}
	return h.initialOpener, nil
}

func (h *cryptoSetup) Get0RTTOpener() (LongHeaderOpener, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.zeroRTTOpener == nil {
		if h.initialOpener != nil {
			return nil, ErrKeysNotYetAvailable
		}
		// if the initial opener is also not available, the keys were already dropped
		return nil, ErrKeysDropped
	}
	return h.zeroRTTOpener, nil
}

func (h *cryptoSetup) GetHandshakeOpener() (LongHeaderOpener, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.handshakeOpener == nil {
		if h.initialOpener != nil {
			return nil, ErrKeysNotYetAvailable
		}
		// if the initial opener is also not available, the keys were already dropped
		return nil, ErrKeysDropped
	}
	return h.handshakeOpener, nil
}

func (h *cryptoSetup) Get1RTTOpener() (ShortHeaderOpener, error) {
	h.mutex.Lock()
	defer h.mutex.Unlock()

	if h.zeroRTTOpener != nil && time.Since(h.handshakeCompleteTime) > 3*h.rttStats.PTO(true) {
		h.zeroRTTOpener = nil
		h.logger.Debugf("Dropping 0-RTT keys.")
		if h.tracer != nil {
			h.tracer.DroppedEncryptionLevel(protocol.Encryption0RTT)
		}
	}

	if !h.has1RTTOpener {
		return nil, ErrKeysNotYetAvailable
	}
	return h.aead, nil
}

func (h *cryptoSetup) ConnectionState() ConnectionState {
	return qtls.GetConnectionState(h.conn)
}
