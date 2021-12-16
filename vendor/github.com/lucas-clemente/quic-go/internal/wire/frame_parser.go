package wire

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
)

type frameParser struct {
	ackDelayExponent uint8

	supportsDatagrams bool

	version protocol.VersionNumber
}

// NewFrameParser creates a new frame parser.
func NewFrameParser(supportsDatagrams bool, v protocol.VersionNumber) FrameParser {
	return &frameParser{
		supportsDatagrams: supportsDatagrams,
		version:           v,
	}
}

// ParseNext parses the next frame.
// It skips PADDING frames.
func (p *frameParser) ParseNext(r *bytes.Reader, encLevel protocol.EncryptionLevel) (Frame, error) {
	for r.Len() != 0 {
		typeByte, _ := r.ReadByte()
		if typeByte == 0x0 { // PADDING frame
			continue
		}
		r.UnreadByte()

		f, err := p.parseFrame(r, typeByte, encLevel)
		if err != nil {
			return nil, &qerr.TransportError{
				FrameType:    uint64(typeByte),
				ErrorCode:    qerr.FrameEncodingError,
				ErrorMessage: err.Error(),
			}
		}
		return f, nil
	}
	return nil, nil
}

func (p *frameParser) parseFrame(r *bytes.Reader, typeByte byte, encLevel protocol.EncryptionLevel) (Frame, error) {
	var frame Frame
	var err error
	if typeByte&0xf8 == 0x8 {
		frame, err = parseStreamFrame(r, p.version)
	} else {
		switch typeByte {
		case 0x1:
			frame, err = parsePingFrame(r, p.version)
		case 0x2, 0x3:
			ackDelayExponent := p.ackDelayExponent
			if encLevel != protocol.Encryption1RTT {
				ackDelayExponent = protocol.DefaultAckDelayExponent
			}
			frame, err = parseAckFrame(r, ackDelayExponent, p.version)
		case 0x4:
			frame, err = parseResetStreamFrame(r, p.version)
		case 0x5:
			frame, err = parseStopSendingFrame(r, p.version)
		case 0x6:
			frame, err = parseCryptoFrame(r, p.version)
		case 0x7:
			frame, err = parseNewTokenFrame(r, p.version)
		case 0x10:
			frame, err = parseMaxDataFrame(r, p.version)
		case 0x11:
			frame, err = parseMaxStreamDataFrame(r, p.version)
		case 0x12, 0x13:
			frame, err = parseMaxStreamsFrame(r, p.version)
		case 0x14:
			frame, err = parseDataBlockedFrame(r, p.version)
		case 0x15:
			frame, err = parseStreamDataBlockedFrame(r, p.version)
		case 0x16, 0x17:
			frame, err = parseStreamsBlockedFrame(r, p.version)
		case 0x18:
			frame, err = parseNewConnectionIDFrame(r, p.version)
		case 0x19:
			frame, err = parseRetireConnectionIDFrame(r, p.version)
		case 0x1a:
			frame, err = parsePathChallengeFrame(r, p.version)
		case 0x1b:
			frame, err = parsePathResponseFrame(r, p.version)
		case 0x1c, 0x1d:
			frame, err = parseConnectionCloseFrame(r, p.version)
		case 0x1e:
			frame, err = parseHandshakeDoneFrame(r, p.version)
		case 0x30, 0x31:
			if p.supportsDatagrams {
				frame, err = parseDatagramFrame(r, p.version)
				break
			}
			fallthrough
		default:
			err = errors.New("unknown frame type")
		}
	}
	if err != nil {
		return nil, err
	}
	if !p.isAllowedAtEncLevel(frame, encLevel) {
		return nil, fmt.Errorf("%s not allowed at encryption level %s", reflect.TypeOf(frame).Elem().Name(), encLevel)
	}
	return frame, nil
}

func (p *frameParser) isAllowedAtEncLevel(f Frame, encLevel protocol.EncryptionLevel) bool {
	switch encLevel {
	case protocol.EncryptionInitial, protocol.EncryptionHandshake:
		switch f.(type) {
		case *CryptoFrame, *AckFrame, *ConnectionCloseFrame, *PingFrame:
			return true
		default:
			return false
		}
	case protocol.Encryption0RTT:
		switch f.(type) {
		case *CryptoFrame, *AckFrame, *ConnectionCloseFrame, *NewTokenFrame, *PathResponseFrame, *RetireConnectionIDFrame:
			return false
		default:
			return true
		}
	case protocol.Encryption1RTT:
		return true
	default:
		panic("unknown encryption level")
	}
}

func (p *frameParser) SetAckDelayExponent(exp uint8) {
	p.ackDelayExponent = exp
}
