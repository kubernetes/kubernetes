package quic

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"strings"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/logging"
)

type client struct {
	conn sendConn
	// If the client is created with DialAddr, we create a packet conn.
	// If it is started with Dial, we take a packet conn as a parameter.
	createdPacketConn bool

	use0RTT bool

	packetHandlers packetHandlerManager

	tlsConf *tls.Config
	config  *Config

	srcConnID  protocol.ConnectionID
	destConnID protocol.ConnectionID

	initialPacketNumber  protocol.PacketNumber
	hasNegotiatedVersion bool
	version              protocol.VersionNumber

	handshakeChan chan struct{}

	session quicSession

	tracer    logging.ConnectionTracer
	tracingID uint64
	logger    utils.Logger
}

var (
	// make it possible to mock connection ID generation in the tests
	generateConnectionID           = protocol.GenerateConnectionID
	generateConnectionIDForInitial = protocol.GenerateConnectionIDForInitial
)

// DialAddr establishes a new QUIC connection to a server.
// It uses a new UDP connection and closes this connection when the QUIC session is closed.
// The hostname for SNI is taken from the given address.
// The tls.Config.CipherSuites allows setting of TLS 1.3 cipher suites.
func DialAddr(
	addr string,
	tlsConf *tls.Config,
	config *Config,
) (Session, error) {
	return DialAddrContext(context.Background(), addr, tlsConf, config)
}

// DialAddrEarly establishes a new 0-RTT QUIC connection to a server.
// It uses a new UDP connection and closes this connection when the QUIC session is closed.
// The hostname for SNI is taken from the given address.
// The tls.Config.CipherSuites allows setting of TLS 1.3 cipher suites.
func DialAddrEarly(
	addr string,
	tlsConf *tls.Config,
	config *Config,
) (EarlySession, error) {
	return DialAddrEarlyContext(context.Background(), addr, tlsConf, config)
}

// DialAddrEarlyContext establishes a new 0-RTT QUIC connection to a server using provided context.
// See DialAddrEarly for details
func DialAddrEarlyContext(
	ctx context.Context,
	addr string,
	tlsConf *tls.Config,
	config *Config,
) (EarlySession, error) {
	sess, err := dialAddrContext(ctx, addr, tlsConf, config, true)
	if err != nil {
		return nil, err
	}
	utils.Logger.WithPrefix(utils.DefaultLogger, "client").Debugf("Returning early session")
	return sess, nil
}

// DialAddrContext establishes a new QUIC connection to a server using the provided context.
// See DialAddr for details.
func DialAddrContext(
	ctx context.Context,
	addr string,
	tlsConf *tls.Config,
	config *Config,
) (Session, error) {
	return dialAddrContext(ctx, addr, tlsConf, config, false)
}

func dialAddrContext(
	ctx context.Context,
	addr string,
	tlsConf *tls.Config,
	config *Config,
	use0RTT bool,
) (quicSession, error) {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil, err
	}
	udpConn, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.IPv4zero, Port: 0})
	if err != nil {
		return nil, err
	}
	return dialContext(ctx, udpConn, udpAddr, addr, tlsConf, config, use0RTT, true)
}

// Dial establishes a new QUIC connection to a server using a net.PacketConn. If
// the PacketConn satisfies the OOBCapablePacketConn interface (as a net.UDPConn
// does), ECN and packet info support will be enabled. In this case, ReadMsgUDP
// and WriteMsgUDP will be used instead of ReadFrom and WriteTo to read/write
// packets. The same PacketConn can be used for multiple calls to Dial and
// Listen, QUIC connection IDs are used for demultiplexing the different
// connections. The host parameter is used for SNI. The tls.Config must define
// an application protocol (using NextProtos).
func Dial(
	pconn net.PacketConn,
	remoteAddr net.Addr,
	host string,
	tlsConf *tls.Config,
	config *Config,
) (Session, error) {
	return dialContext(context.Background(), pconn, remoteAddr, host, tlsConf, config, false, false)
}

// DialEarly establishes a new 0-RTT QUIC connection to a server using a net.PacketConn.
// The same PacketConn can be used for multiple calls to Dial and Listen,
// QUIC connection IDs are used for demultiplexing the different connections.
// The host parameter is used for SNI.
// The tls.Config must define an application protocol (using NextProtos).
func DialEarly(
	pconn net.PacketConn,
	remoteAddr net.Addr,
	host string,
	tlsConf *tls.Config,
	config *Config,
) (EarlySession, error) {
	return DialEarlyContext(context.Background(), pconn, remoteAddr, host, tlsConf, config)
}

// DialEarlyContext establishes a new 0-RTT QUIC connection to a server using a net.PacketConn using the provided context.
// See DialEarly for details.
func DialEarlyContext(
	ctx context.Context,
	pconn net.PacketConn,
	remoteAddr net.Addr,
	host string,
	tlsConf *tls.Config,
	config *Config,
) (EarlySession, error) {
	return dialContext(ctx, pconn, remoteAddr, host, tlsConf, config, true, false)
}

// DialContext establishes a new QUIC connection to a server using a net.PacketConn using the provided context.
// See Dial for details.
func DialContext(
	ctx context.Context,
	pconn net.PacketConn,
	remoteAddr net.Addr,
	host string,
	tlsConf *tls.Config,
	config *Config,
) (Session, error) {
	return dialContext(ctx, pconn, remoteAddr, host, tlsConf, config, false, false)
}

func dialContext(
	ctx context.Context,
	pconn net.PacketConn,
	remoteAddr net.Addr,
	host string,
	tlsConf *tls.Config,
	config *Config,
	use0RTT bool,
	createdPacketConn bool,
) (quicSession, error) {
	if tlsConf == nil {
		return nil, errors.New("quic: tls.Config not set")
	}
	if err := validateConfig(config); err != nil {
		return nil, err
	}
	config = populateClientConfig(config, createdPacketConn)
	packetHandlers, err := getMultiplexer().AddConn(pconn, config.ConnectionIDLength, config.StatelessResetKey, config.Tracer)
	if err != nil {
		return nil, err
	}
	c, err := newClient(pconn, remoteAddr, config, tlsConf, host, use0RTT, createdPacketConn)
	if err != nil {
		return nil, err
	}
	c.packetHandlers = packetHandlers

	c.tracingID = nextSessionTracingID()
	if c.config.Tracer != nil {
		c.tracer = c.config.Tracer.TracerForConnection(
			context.WithValue(ctx, SessionTracingKey, c.tracingID),
			protocol.PerspectiveClient,
			c.destConnID,
		)
	}
	if c.tracer != nil {
		c.tracer.StartedConnection(c.conn.LocalAddr(), c.conn.RemoteAddr(), c.srcConnID, c.destConnID)
	}
	if err := c.dial(ctx); err != nil {
		return nil, err
	}
	return c.session, nil
}

func newClient(
	pconn net.PacketConn,
	remoteAddr net.Addr,
	config *Config,
	tlsConf *tls.Config,
	host string,
	use0RTT bool,
	createdPacketConn bool,
) (*client, error) {
	if tlsConf == nil {
		tlsConf = &tls.Config{}
	}
	if tlsConf.ServerName == "" {
		sni := host
		if strings.IndexByte(sni, ':') != -1 {
			var err error
			sni, _, err = net.SplitHostPort(sni)
			if err != nil {
				return nil, err
			}
		}

		tlsConf.ServerName = sni
	}

	// check that all versions are actually supported
	if config != nil {
		for _, v := range config.Versions {
			if !protocol.IsValidVersion(v) {
				return nil, fmt.Errorf("%s is not a valid QUIC version", v)
			}
		}
	}

	srcConnID, err := generateConnectionID(config.ConnectionIDLength)
	if err != nil {
		return nil, err
	}
	destConnID, err := generateConnectionIDForInitial()
	if err != nil {
		return nil, err
	}
	c := &client{
		srcConnID:         srcConnID,
		destConnID:        destConnID,
		conn:              newSendPconn(pconn, remoteAddr),
		createdPacketConn: createdPacketConn,
		use0RTT:           use0RTT,
		tlsConf:           tlsConf,
		config:            config,
		version:           config.Versions[0],
		handshakeChan:     make(chan struct{}),
		logger:            utils.DefaultLogger.WithPrefix("client"),
	}
	return c, nil
}

func (c *client) dial(ctx context.Context) error {
	c.logger.Infof("Starting new connection to %s (%s -> %s), source connection ID %s, destination connection ID %s, version %s", c.tlsConf.ServerName, c.conn.LocalAddr(), c.conn.RemoteAddr(), c.srcConnID, c.destConnID, c.version)

	c.session = newClientSession(
		c.conn,
		c.packetHandlers,
		c.destConnID,
		c.srcConnID,
		c.config,
		c.tlsConf,
		c.initialPacketNumber,
		c.use0RTT,
		c.hasNegotiatedVersion,
		c.tracer,
		c.tracingID,
		c.logger,
		c.version,
	)
	c.packetHandlers.Add(c.srcConnID, c.session)

	errorChan := make(chan error, 1)
	go func() {
		err := c.session.run() // returns as soon as the session is closed

		if e := (&errCloseForRecreating{}); !errors.As(err, &e) && c.createdPacketConn {
			c.packetHandlers.Destroy()
		}
		errorChan <- err
	}()

	// only set when we're using 0-RTT
	// Otherwise, earlySessionChan will be nil. Receiving from a nil chan blocks forever.
	var earlySessionChan <-chan struct{}
	if c.use0RTT {
		earlySessionChan = c.session.earlySessionReady()
	}

	select {
	case <-ctx.Done():
		c.session.shutdown()
		return ctx.Err()
	case err := <-errorChan:
		var recreateErr *errCloseForRecreating
		if errors.As(err, &recreateErr) {
			c.initialPacketNumber = recreateErr.nextPacketNumber
			c.version = recreateErr.nextVersion
			c.hasNegotiatedVersion = true
			return c.dial(ctx)
		}
		return err
	case <-earlySessionChan:
		// ready to send 0-RTT data
		return nil
	case <-c.session.HandshakeComplete().Done():
		// handshake successfully completed
		return nil
	}
}
