package wire

import (
	"bytes"
	"errors"
	"fmt"
	"io"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/quicvarint"
)

// ErrInvalidReservedBits is returned when the reserved bits are incorrect.
// When this error is returned, parsing continues, and an ExtendedHeader is returned.
// This is necessary because we need to decrypt the packet in that case,
// in order to avoid a timing side-channel.
var ErrInvalidReservedBits = errors.New("invalid reserved bits")

// ExtendedHeader is the header of a QUIC packet.
type ExtendedHeader struct {
	Header

	typeByte byte

	KeyPhase protocol.KeyPhaseBit

	PacketNumberLen protocol.PacketNumberLen
	PacketNumber    protocol.PacketNumber

	parsedLen protocol.ByteCount
}

func (h *ExtendedHeader) parse(b *bytes.Reader, v protocol.VersionNumber) (bool /* reserved bits valid */, error) {
	startLen := b.Len()
	// read the (now unencrypted) first byte
	var err error
	h.typeByte, err = b.ReadByte()
	if err != nil {
		return false, err
	}
	if _, err := b.Seek(int64(h.Header.ParsedLen())-1, io.SeekCurrent); err != nil {
		return false, err
	}
	var reservedBitsValid bool
	if h.IsLongHeader {
		reservedBitsValid, err = h.parseLongHeader(b, v)
	} else {
		reservedBitsValid, err = h.parseShortHeader(b, v)
	}
	if err != nil {
		return false, err
	}
	h.parsedLen = protocol.ByteCount(startLen - b.Len())
	return reservedBitsValid, err
}

func (h *ExtendedHeader) parseLongHeader(b *bytes.Reader, _ protocol.VersionNumber) (bool /* reserved bits valid */, error) {
	if err := h.readPacketNumber(b); err != nil {
		return false, err
	}
	if h.typeByte&0xc != 0 {
		return false, nil
	}
	return true, nil
}

func (h *ExtendedHeader) parseShortHeader(b *bytes.Reader, _ protocol.VersionNumber) (bool /* reserved bits valid */, error) {
	h.KeyPhase = protocol.KeyPhaseZero
	if h.typeByte&0x4 > 0 {
		h.KeyPhase = protocol.KeyPhaseOne
	}

	if err := h.readPacketNumber(b); err != nil {
		return false, err
	}
	if h.typeByte&0x18 != 0 {
		return false, nil
	}
	return true, nil
}

func (h *ExtendedHeader) readPacketNumber(b *bytes.Reader) error {
	h.PacketNumberLen = protocol.PacketNumberLen(h.typeByte&0x3) + 1
	switch h.PacketNumberLen {
	case protocol.PacketNumberLen1:
		n, err := b.ReadByte()
		if err != nil {
			return err
		}
		h.PacketNumber = protocol.PacketNumber(n)
	case protocol.PacketNumberLen2:
		n, err := utils.BigEndian.ReadUint16(b)
		if err != nil {
			return err
		}
		h.PacketNumber = protocol.PacketNumber(n)
	case protocol.PacketNumberLen3:
		n, err := utils.BigEndian.ReadUint24(b)
		if err != nil {
			return err
		}
		h.PacketNumber = protocol.PacketNumber(n)
	case protocol.PacketNumberLen4:
		n, err := utils.BigEndian.ReadUint32(b)
		if err != nil {
			return err
		}
		h.PacketNumber = protocol.PacketNumber(n)
	default:
		return fmt.Errorf("invalid packet number length: %d", h.PacketNumberLen)
	}
	return nil
}

// Write writes the Header.
func (h *ExtendedHeader) Write(b *bytes.Buffer, ver protocol.VersionNumber) error {
	if h.DestConnectionID.Len() > protocol.MaxConnIDLen {
		return fmt.Errorf("invalid connection ID length: %d bytes", h.DestConnectionID.Len())
	}
	if h.SrcConnectionID.Len() > protocol.MaxConnIDLen {
		return fmt.Errorf("invalid connection ID length: %d bytes", h.SrcConnectionID.Len())
	}
	if h.IsLongHeader {
		return h.writeLongHeader(b, ver)
	}
	return h.writeShortHeader(b, ver)
}

func (h *ExtendedHeader) writeLongHeader(b *bytes.Buffer, _ protocol.VersionNumber) error {
	var packetType uint8
	//nolint:exhaustive
	switch h.Type {
	case protocol.PacketTypeInitial:
		packetType = 0x0
	case protocol.PacketType0RTT:
		packetType = 0x1
	case protocol.PacketTypeHandshake:
		packetType = 0x2
	case protocol.PacketTypeRetry:
		packetType = 0x3
	}
	firstByte := 0xc0 | packetType<<4
	if h.Type != protocol.PacketTypeRetry {
		// Retry packets don't have a packet number
		firstByte |= uint8(h.PacketNumberLen - 1)
	}

	b.WriteByte(firstByte)
	utils.BigEndian.WriteUint32(b, uint32(h.Version))
	b.WriteByte(uint8(h.DestConnectionID.Len()))
	b.Write(h.DestConnectionID.Bytes())
	b.WriteByte(uint8(h.SrcConnectionID.Len()))
	b.Write(h.SrcConnectionID.Bytes())

	//nolint:exhaustive
	switch h.Type {
	case protocol.PacketTypeRetry:
		b.Write(h.Token)
		return nil
	case protocol.PacketTypeInitial:
		quicvarint.Write(b, uint64(len(h.Token)))
		b.Write(h.Token)
	}
	quicvarint.WriteWithLen(b, uint64(h.Length), 2)
	return h.writePacketNumber(b)
}

func (h *ExtendedHeader) writeShortHeader(b *bytes.Buffer, _ protocol.VersionNumber) error {
	typeByte := 0x40 | uint8(h.PacketNumberLen-1)
	if h.KeyPhase == protocol.KeyPhaseOne {
		typeByte |= byte(1 << 2)
	}

	b.WriteByte(typeByte)
	b.Write(h.DestConnectionID.Bytes())
	return h.writePacketNumber(b)
}

func (h *ExtendedHeader) writePacketNumber(b *bytes.Buffer) error {
	switch h.PacketNumberLen {
	case protocol.PacketNumberLen1:
		b.WriteByte(uint8(h.PacketNumber))
	case protocol.PacketNumberLen2:
		utils.BigEndian.WriteUint16(b, uint16(h.PacketNumber))
	case protocol.PacketNumberLen3:
		utils.BigEndian.WriteUint24(b, uint32(h.PacketNumber))
	case protocol.PacketNumberLen4:
		utils.BigEndian.WriteUint32(b, uint32(h.PacketNumber))
	default:
		return fmt.Errorf("invalid packet number length: %d", h.PacketNumberLen)
	}
	return nil
}

// ParsedLen returns the number of bytes that were consumed when parsing the header
func (h *ExtendedHeader) ParsedLen() protocol.ByteCount {
	return h.parsedLen
}

// GetLength determines the length of the Header.
func (h *ExtendedHeader) GetLength(v protocol.VersionNumber) protocol.ByteCount {
	if h.IsLongHeader {
		length := 1 /* type byte */ + 4 /* version */ + 1 /* dest conn ID len */ + protocol.ByteCount(h.DestConnectionID.Len()) + 1 /* src conn ID len */ + protocol.ByteCount(h.SrcConnectionID.Len()) + protocol.ByteCount(h.PacketNumberLen) + 2 /* length */
		if h.Type == protocol.PacketTypeInitial {
			length += quicvarint.Len(uint64(len(h.Token))) + protocol.ByteCount(len(h.Token))
		}
		return length
	}

	length := protocol.ByteCount(1 /* type byte */ + h.DestConnectionID.Len())
	length += protocol.ByteCount(h.PacketNumberLen)
	return length
}

// Log logs the Header
func (h *ExtendedHeader) Log(logger utils.Logger) {
	if h.IsLongHeader {
		var token string
		if h.Type == protocol.PacketTypeInitial || h.Type == protocol.PacketTypeRetry {
			if len(h.Token) == 0 {
				token = "Token: (empty), "
			} else {
				token = fmt.Sprintf("Token: %#x, ", h.Token)
			}
			if h.Type == protocol.PacketTypeRetry {
				logger.Debugf("\tLong Header{Type: %s, DestConnectionID: %s, SrcConnectionID: %s, %sVersion: %s}", h.Type, h.DestConnectionID, h.SrcConnectionID, token, h.Version)
				return
			}
		}
		logger.Debugf("\tLong Header{Type: %s, DestConnectionID: %s, SrcConnectionID: %s, %sPacketNumber: %d, PacketNumberLen: %d, Length: %d, Version: %s}", h.Type, h.DestConnectionID, h.SrcConnectionID, token, h.PacketNumber, h.PacketNumberLen, h.Length, h.Version)
	} else {
		logger.Debugf("\tShort Header{DestConnectionID: %s, PacketNumber: %d, PacketNumberLen: %d, KeyPhase: %s}", h.DestConnectionID, h.PacketNumber, h.PacketNumberLen, h.KeyPhase)
	}
}
