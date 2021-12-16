package quic

import (
	"fmt"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

type connIDGenerator struct {
	connIDLen  int
	highestSeq uint64

	activeSrcConnIDs        map[uint64]protocol.ConnectionID
	initialClientDestConnID protocol.ConnectionID

	addConnectionID        func(protocol.ConnectionID)
	getStatelessResetToken func(protocol.ConnectionID) protocol.StatelessResetToken
	removeConnectionID     func(protocol.ConnectionID)
	retireConnectionID     func(protocol.ConnectionID)
	replaceWithClosed      func(protocol.ConnectionID, packetHandler)
	queueControlFrame      func(wire.Frame)

	version protocol.VersionNumber
}

func newConnIDGenerator(
	initialConnectionID protocol.ConnectionID,
	initialClientDestConnID protocol.ConnectionID, // nil for the client
	addConnectionID func(protocol.ConnectionID),
	getStatelessResetToken func(protocol.ConnectionID) protocol.StatelessResetToken,
	removeConnectionID func(protocol.ConnectionID),
	retireConnectionID func(protocol.ConnectionID),
	replaceWithClosed func(protocol.ConnectionID, packetHandler),
	queueControlFrame func(wire.Frame),
	version protocol.VersionNumber,
) *connIDGenerator {
	m := &connIDGenerator{
		connIDLen:              initialConnectionID.Len(),
		activeSrcConnIDs:       make(map[uint64]protocol.ConnectionID),
		addConnectionID:        addConnectionID,
		getStatelessResetToken: getStatelessResetToken,
		removeConnectionID:     removeConnectionID,
		retireConnectionID:     retireConnectionID,
		replaceWithClosed:      replaceWithClosed,
		queueControlFrame:      queueControlFrame,
		version:                version,
	}
	m.activeSrcConnIDs[0] = initialConnectionID
	m.initialClientDestConnID = initialClientDestConnID
	return m
}

func (m *connIDGenerator) SetMaxActiveConnIDs(limit uint64) error {
	if m.connIDLen == 0 {
		return nil
	}
	// The active_connection_id_limit transport parameter is the number of
	// connection IDs the peer will store. This limit includes the connection ID
	// used during the handshake, and the one sent in the preferred_address
	// transport parameter.
	// We currently don't send the preferred_address transport parameter,
	// so we can issue (limit - 1) connection IDs.
	for i := uint64(len(m.activeSrcConnIDs)); i < utils.MinUint64(limit, protocol.MaxIssuedConnectionIDs); i++ {
		if err := m.issueNewConnID(); err != nil {
			return err
		}
	}
	return nil
}

func (m *connIDGenerator) Retire(seq uint64, sentWithDestConnID protocol.ConnectionID) error {
	if seq > m.highestSeq {
		return &qerr.TransportError{
			ErrorCode:    qerr.ProtocolViolation,
			ErrorMessage: fmt.Sprintf("retired connection ID %d (highest issued: %d)", seq, m.highestSeq),
		}
	}
	connID, ok := m.activeSrcConnIDs[seq]
	// We might already have deleted this connection ID, if this is a duplicate frame.
	if !ok {
		return nil
	}
	if connID.Equal(sentWithDestConnID) {
		return &qerr.TransportError{
			ErrorCode:    qerr.ProtocolViolation,
			ErrorMessage: fmt.Sprintf("retired connection ID %d (%s), which was used as the Destination Connection ID on this packet", seq, connID),
		}
	}
	m.retireConnectionID(connID)
	delete(m.activeSrcConnIDs, seq)
	// Don't issue a replacement for the initial connection ID.
	if seq == 0 {
		return nil
	}
	return m.issueNewConnID()
}

func (m *connIDGenerator) issueNewConnID() error {
	connID, err := protocol.GenerateConnectionID(m.connIDLen)
	if err != nil {
		return err
	}
	m.activeSrcConnIDs[m.highestSeq+1] = connID
	m.addConnectionID(connID)
	m.queueControlFrame(&wire.NewConnectionIDFrame{
		SequenceNumber:      m.highestSeq + 1,
		ConnectionID:        connID,
		StatelessResetToken: m.getStatelessResetToken(connID),
	})
	m.highestSeq++
	return nil
}

func (m *connIDGenerator) SetHandshakeComplete() {
	if m.initialClientDestConnID != nil {
		m.retireConnectionID(m.initialClientDestConnID)
		m.initialClientDestConnID = nil
	}
}

func (m *connIDGenerator) RemoveAll() {
	if m.initialClientDestConnID != nil {
		m.removeConnectionID(m.initialClientDestConnID)
	}
	for _, connID := range m.activeSrcConnIDs {
		m.removeConnectionID(connID)
	}
}

func (m *connIDGenerator) ReplaceWithClosed(handler packetHandler) {
	if m.initialClientDestConnID != nil {
		m.replaceWithClosed(m.initialClientDestConnID, handler)
	}
	for _, connID := range m.activeSrcConnIDs {
		m.replaceWithClosed(connID, handler)
	}
}
