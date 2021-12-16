package quic

import (
	"sync"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

// A closedLocalSession is a session that we closed locally.
// When receiving packets for such a session, we need to retransmit the packet containing the CONNECTION_CLOSE frame,
// with an exponential backoff.
type closedLocalSession struct {
	conn            sendConn
	connClosePacket []byte

	closeOnce sync.Once
	closeChan chan struct{} // is closed when the session is closed or destroyed

	receivedPackets chan *receivedPacket
	counter         uint64 // number of packets received

	perspective protocol.Perspective

	logger utils.Logger
}

var _ packetHandler = &closedLocalSession{}

// newClosedLocalSession creates a new closedLocalSession and runs it.
func newClosedLocalSession(
	conn sendConn,
	connClosePacket []byte,
	perspective protocol.Perspective,
	logger utils.Logger,
) packetHandler {
	s := &closedLocalSession{
		conn:            conn,
		connClosePacket: connClosePacket,
		perspective:     perspective,
		logger:          logger,
		closeChan:       make(chan struct{}),
		receivedPackets: make(chan *receivedPacket, 64),
	}
	go s.run()
	return s
}

func (s *closedLocalSession) run() {
	for {
		select {
		case p := <-s.receivedPackets:
			s.handlePacketImpl(p)
		case <-s.closeChan:
			return
		}
	}
}

func (s *closedLocalSession) handlePacket(p *receivedPacket) {
	select {
	case s.receivedPackets <- p:
	default:
	}
}

func (s *closedLocalSession) handlePacketImpl(_ *receivedPacket) {
	s.counter++
	// exponential backoff
	// only send a CONNECTION_CLOSE for the 1st, 2nd, 4th, 8th, 16th, ... packet arriving
	for n := s.counter; n > 1; n = n / 2 {
		if n%2 != 0 {
			return
		}
	}
	s.logger.Debugf("Received %d packets after sending CONNECTION_CLOSE. Retransmitting.", s.counter)
	if err := s.conn.Write(s.connClosePacket); err != nil {
		s.logger.Debugf("Error retransmitting CONNECTION_CLOSE: %s", err)
	}
}

func (s *closedLocalSession) shutdown() {
	s.destroy(nil)
}

func (s *closedLocalSession) destroy(error) {
	s.closeOnce.Do(func() {
		close(s.closeChan)
	})
}

func (s *closedLocalSession) getPerspective() protocol.Perspective {
	return s.perspective
}

// A closedRemoteSession is a session that was closed remotely.
// For such a session, we might receive reordered packets that were sent before the CONNECTION_CLOSE.
// We can just ignore those packets.
type closedRemoteSession struct {
	perspective protocol.Perspective
}

var _ packetHandler = &closedRemoteSession{}

func newClosedRemoteSession(pers protocol.Perspective) packetHandler {
	return &closedRemoteSession{perspective: pers}
}

func (s *closedRemoteSession) handlePacket(*receivedPacket)         {}
func (s *closedRemoteSession) shutdown()                            {}
func (s *closedRemoteSession) destroy(error)                        {}
func (s *closedRemoteSession) getPerspective() protocol.Perspective { return s.perspective }
