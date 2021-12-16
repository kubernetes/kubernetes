package quic

import (
	"fmt"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

type cryptoDataHandler interface {
	HandleMessage([]byte, protocol.EncryptionLevel) bool
}

type cryptoStreamManager struct {
	cryptoHandler cryptoDataHandler

	initialStream   cryptoStream
	handshakeStream cryptoStream
	oneRTTStream    cryptoStream
}

func newCryptoStreamManager(
	cryptoHandler cryptoDataHandler,
	initialStream cryptoStream,
	handshakeStream cryptoStream,
	oneRTTStream cryptoStream,
) *cryptoStreamManager {
	return &cryptoStreamManager{
		cryptoHandler:   cryptoHandler,
		initialStream:   initialStream,
		handshakeStream: handshakeStream,
		oneRTTStream:    oneRTTStream,
	}
}

func (m *cryptoStreamManager) HandleCryptoFrame(frame *wire.CryptoFrame, encLevel protocol.EncryptionLevel) (bool /* encryption level changed */, error) {
	var str cryptoStream
	//nolint:exhaustive // CRYPTO frames cannot be sent in 0-RTT packets.
	switch encLevel {
	case protocol.EncryptionInitial:
		str = m.initialStream
	case protocol.EncryptionHandshake:
		str = m.handshakeStream
	case protocol.Encryption1RTT:
		str = m.oneRTTStream
	default:
		return false, fmt.Errorf("received CRYPTO frame with unexpected encryption level: %s", encLevel)
	}
	if err := str.HandleCryptoFrame(frame); err != nil {
		return false, err
	}
	for {
		data := str.GetCryptoData()
		if data == nil {
			return false, nil
		}
		if encLevelFinished := m.cryptoHandler.HandleMessage(data, encLevel); encLevelFinished {
			return true, str.Finish()
		}
	}
}
