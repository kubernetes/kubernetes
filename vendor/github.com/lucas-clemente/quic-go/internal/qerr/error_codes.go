package qerr

import (
	"fmt"

	"github.com/lucas-clemente/quic-go/internal/qtls"
)

// TransportErrorCode is a QUIC transport error.
type TransportErrorCode uint64

// The error codes defined by QUIC
const (
	NoError                   TransportErrorCode = 0x0
	InternalError             TransportErrorCode = 0x1
	ConnectionRefused         TransportErrorCode = 0x2
	FlowControlError          TransportErrorCode = 0x3
	StreamLimitError          TransportErrorCode = 0x4
	StreamStateError          TransportErrorCode = 0x5
	FinalSizeError            TransportErrorCode = 0x6
	FrameEncodingError        TransportErrorCode = 0x7
	TransportParameterError   TransportErrorCode = 0x8
	ConnectionIDLimitError    TransportErrorCode = 0x9
	ProtocolViolation         TransportErrorCode = 0xa
	InvalidToken              TransportErrorCode = 0xb
	ApplicationErrorErrorCode TransportErrorCode = 0xc
	CryptoBufferExceeded      TransportErrorCode = 0xd
	KeyUpdateError            TransportErrorCode = 0xe
	AEADLimitReached          TransportErrorCode = 0xf
	NoViablePathError         TransportErrorCode = 0x10
)

func (e TransportErrorCode) IsCryptoError() bool {
	return e >= 0x100 && e < 0x200
}

// Message is a description of the error.
// It only returns a non-empty string for crypto errors.
func (e TransportErrorCode) Message() string {
	if !e.IsCryptoError() {
		return ""
	}
	return qtls.Alert(e - 0x100).Error()
}

func (e TransportErrorCode) String() string {
	switch e {
	case NoError:
		return "NO_ERROR"
	case InternalError:
		return "INTERNAL_ERROR"
	case ConnectionRefused:
		return "CONNECTION_REFUSED"
	case FlowControlError:
		return "FLOW_CONTROL_ERROR"
	case StreamLimitError:
		return "STREAM_LIMIT_ERROR"
	case StreamStateError:
		return "STREAM_STATE_ERROR"
	case FinalSizeError:
		return "FINAL_SIZE_ERROR"
	case FrameEncodingError:
		return "FRAME_ENCODING_ERROR"
	case TransportParameterError:
		return "TRANSPORT_PARAMETER_ERROR"
	case ConnectionIDLimitError:
		return "CONNECTION_ID_LIMIT_ERROR"
	case ProtocolViolation:
		return "PROTOCOL_VIOLATION"
	case InvalidToken:
		return "INVALID_TOKEN"
	case ApplicationErrorErrorCode:
		return "APPLICATION_ERROR"
	case CryptoBufferExceeded:
		return "CRYPTO_BUFFER_EXCEEDED"
	case KeyUpdateError:
		return "KEY_UPDATE_ERROR"
	case AEADLimitReached:
		return "AEAD_LIMIT_REACHED"
	case NoViablePathError:
		return "NO_VIABLE_PATH"
	default:
		if e.IsCryptoError() {
			return fmt.Sprintf("CRYPTO_ERROR (%#x)", uint16(e))
		}
		return fmt.Sprintf("unknown error code: %#x", uint16(e))
	}
}
