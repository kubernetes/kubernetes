package wire

import (
	"bytes"
	"crypto/rand"
	"errors"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

// ParseVersionNegotiationPacket parses a Version Negotiation packet.
func ParseVersionNegotiationPacket(b *bytes.Reader) (*Header, []protocol.VersionNumber, error) {
	hdr, err := parseHeader(b, 0)
	if err != nil {
		return nil, nil, err
	}
	if b.Len() == 0 {
		//nolint:stylecheck
		return nil, nil, errors.New("Version Negotiation packet has empty version list")
	}
	if b.Len()%4 != 0 {
		//nolint:stylecheck
		return nil, nil, errors.New("Version Negotiation packet has a version list with an invalid length")
	}
	versions := make([]protocol.VersionNumber, b.Len()/4)
	for i := 0; b.Len() > 0; i++ {
		v, err := utils.BigEndian.ReadUint32(b)
		if err != nil {
			return nil, nil, err
		}
		versions[i] = protocol.VersionNumber(v)
	}
	return hdr, versions, nil
}

// ComposeVersionNegotiation composes a Version Negotiation
func ComposeVersionNegotiation(destConnID, srcConnID protocol.ConnectionID, versions []protocol.VersionNumber) ([]byte, error) {
	greasedVersions := protocol.GetGreasedVersions(versions)
	expectedLen := 1 /* type byte */ + 4 /* version field */ + 1 /* dest connection ID length field */ + destConnID.Len() + 1 /* src connection ID length field */ + srcConnID.Len() + len(greasedVersions)*4
	buf := bytes.NewBuffer(make([]byte, 0, expectedLen))
	r := make([]byte, 1)
	_, _ = rand.Read(r) // ignore the error here. It is not critical to have perfect random here.
	buf.WriteByte(r[0] | 0x80)
	utils.BigEndian.WriteUint32(buf, 0) // version 0
	buf.WriteByte(uint8(destConnID.Len()))
	buf.Write(destConnID)
	buf.WriteByte(uint8(srcConnID.Len()))
	buf.Write(srcConnID)
	for _, v := range greasedVersions {
		utils.BigEndian.WriteUint32(buf, uint32(v))
	}
	return buf.Bytes(), nil
}
