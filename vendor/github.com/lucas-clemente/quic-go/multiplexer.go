package quic

import (
	"bytes"
	"fmt"
	"net"
	"sync"

	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/logging"
)

var (
	connMuxerOnce sync.Once
	connMuxer     multiplexer
)

type indexableConn interface {
	LocalAddr() net.Addr
}

type multiplexer interface {
	AddConn(c net.PacketConn, connIDLen int, statelessResetKey []byte, tracer logging.Tracer) (packetHandlerManager, error)
	RemoveConn(indexableConn) error
}

type connManager struct {
	connIDLen         int
	statelessResetKey []byte
	tracer            logging.Tracer
	manager           packetHandlerManager
}

// The connMultiplexer listens on multiple net.PacketConns and dispatches
// incoming packets to the session handler.
type connMultiplexer struct {
	mutex sync.Mutex

	conns                   map[string] /* LocalAddr().String() */ connManager
	newPacketHandlerManager func(net.PacketConn, int, []byte, logging.Tracer, utils.Logger) (packetHandlerManager, error) // so it can be replaced in the tests

	logger utils.Logger
}

var _ multiplexer = &connMultiplexer{}

func getMultiplexer() multiplexer {
	connMuxerOnce.Do(func() {
		connMuxer = &connMultiplexer{
			conns:                   make(map[string]connManager),
			logger:                  utils.DefaultLogger.WithPrefix("muxer"),
			newPacketHandlerManager: newPacketHandlerMap,
		}
	})
	return connMuxer
}

func (m *connMultiplexer) AddConn(
	c net.PacketConn,
	connIDLen int,
	statelessResetKey []byte,
	tracer logging.Tracer,
) (packetHandlerManager, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	addr := c.LocalAddr()
	connIndex := addr.Network() + " " + addr.String()
	p, ok := m.conns[connIndex]
	if !ok {
		manager, err := m.newPacketHandlerManager(c, connIDLen, statelessResetKey, tracer, m.logger)
		if err != nil {
			return nil, err
		}
		p = connManager{
			connIDLen:         connIDLen,
			statelessResetKey: statelessResetKey,
			manager:           manager,
			tracer:            tracer,
		}
		m.conns[connIndex] = p
	} else {
		if p.connIDLen != connIDLen {
			return nil, fmt.Errorf("cannot use %d byte connection IDs on a connection that is already using %d byte connction IDs", connIDLen, p.connIDLen)
		}
		if statelessResetKey != nil && !bytes.Equal(p.statelessResetKey, statelessResetKey) {
			return nil, fmt.Errorf("cannot use different stateless reset keys on the same packet conn")
		}
		if tracer != p.tracer {
			return nil, fmt.Errorf("cannot use different tracers on the same packet conn")
		}
	}
	return p.manager, nil
}

func (m *connMultiplexer) RemoveConn(c indexableConn) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	connIndex := c.LocalAddr().Network() + " " + c.LocalAddr().String()
	if _, ok := m.conns[connIndex]; !ok {
		return fmt.Errorf("cannote remove connection, connection is unknown")
	}

	delete(m.conns, connIndex)
	return nil
}
