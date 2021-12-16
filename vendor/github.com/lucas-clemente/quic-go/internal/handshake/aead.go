package handshake

import (
	"crypto/cipher"
	"encoding/binary"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qtls"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

func createAEAD(suite *qtls.CipherSuiteTLS13, trafficSecret []byte) cipher.AEAD {
	key := hkdfExpandLabel(suite.Hash, trafficSecret, []byte{}, "quic key", suite.KeyLen)
	iv := hkdfExpandLabel(suite.Hash, trafficSecret, []byte{}, "quic iv", suite.IVLen())
	return suite.AEAD(key, iv)
}

type longHeaderSealer struct {
	aead            cipher.AEAD
	headerProtector headerProtector

	// use a single slice to avoid allocations
	nonceBuf []byte
}

var _ LongHeaderSealer = &longHeaderSealer{}

func newLongHeaderSealer(aead cipher.AEAD, headerProtector headerProtector) LongHeaderSealer {
	return &longHeaderSealer{
		aead:            aead,
		headerProtector: headerProtector,
		nonceBuf:        make([]byte, aead.NonceSize()),
	}
}

func (s *longHeaderSealer) Seal(dst, src []byte, pn protocol.PacketNumber, ad []byte) []byte {
	binary.BigEndian.PutUint64(s.nonceBuf[len(s.nonceBuf)-8:], uint64(pn))
	// The AEAD we're using here will be the qtls.aeadAESGCM13.
	// It uses the nonce provided here and XOR it with the IV.
	return s.aead.Seal(dst, s.nonceBuf, src, ad)
}

func (s *longHeaderSealer) EncryptHeader(sample []byte, firstByte *byte, pnBytes []byte) {
	s.headerProtector.EncryptHeader(sample, firstByte, pnBytes)
}

func (s *longHeaderSealer) Overhead() int {
	return s.aead.Overhead()
}

type longHeaderOpener struct {
	aead            cipher.AEAD
	headerProtector headerProtector
	highestRcvdPN   protocol.PacketNumber // highest packet number received (which could be successfully unprotected)

	// use a single slice to avoid allocations
	nonceBuf []byte
}

var _ LongHeaderOpener = &longHeaderOpener{}

func newLongHeaderOpener(aead cipher.AEAD, headerProtector headerProtector) LongHeaderOpener {
	return &longHeaderOpener{
		aead:            aead,
		headerProtector: headerProtector,
		nonceBuf:        make([]byte, aead.NonceSize()),
	}
}

func (o *longHeaderOpener) DecodePacketNumber(wirePN protocol.PacketNumber, wirePNLen protocol.PacketNumberLen) protocol.PacketNumber {
	return protocol.DecodePacketNumber(wirePNLen, o.highestRcvdPN, wirePN)
}

func (o *longHeaderOpener) Open(dst, src []byte, pn protocol.PacketNumber, ad []byte) ([]byte, error) {
	binary.BigEndian.PutUint64(o.nonceBuf[len(o.nonceBuf)-8:], uint64(pn))
	// The AEAD we're using here will be the qtls.aeadAESGCM13.
	// It uses the nonce provided here and XOR it with the IV.
	dec, err := o.aead.Open(dst, o.nonceBuf, src, ad)
	if err == nil {
		o.highestRcvdPN = utils.MaxPacketNumber(o.highestRcvdPN, pn)
	} else {
		err = ErrDecryptionFailed
	}
	return dec, err
}

func (o *longHeaderOpener) DecryptHeader(sample []byte, firstByte *byte, pnBytes []byte) {
	o.headerProtector.DecryptHeader(sample, firstByte, pnBytes)
}

type handshakeSealer struct {
	LongHeaderSealer

	dropInitialKeys func()
	dropped         bool
}

func newHandshakeSealer(
	aead cipher.AEAD,
	headerProtector headerProtector,
	dropInitialKeys func(),
	perspective protocol.Perspective,
) LongHeaderSealer {
	sealer := newLongHeaderSealer(aead, headerProtector)
	// The client drops Initial keys when sending the first Handshake packet.
	if perspective == protocol.PerspectiveServer {
		return sealer
	}
	return &handshakeSealer{
		LongHeaderSealer: sealer,
		dropInitialKeys:  dropInitialKeys,
	}
}

func (s *handshakeSealer) Seal(dst, src []byte, pn protocol.PacketNumber, ad []byte) []byte {
	data := s.LongHeaderSealer.Seal(dst, src, pn, ad)
	if !s.dropped {
		s.dropInitialKeys()
		s.dropped = true
	}
	return data
}

type handshakeOpener struct {
	LongHeaderOpener

	dropInitialKeys func()
	dropped         bool
}

func newHandshakeOpener(
	aead cipher.AEAD,
	headerProtector headerProtector,
	dropInitialKeys func(),
	perspective protocol.Perspective,
) LongHeaderOpener {
	opener := newLongHeaderOpener(aead, headerProtector)
	// The server drops Initial keys when first successfully processing a Handshake packet.
	if perspective == protocol.PerspectiveClient {
		return opener
	}
	return &handshakeOpener{
		LongHeaderOpener: opener,
		dropInitialKeys:  dropInitialKeys,
	}
}

func (o *handshakeOpener) Open(dst, src []byte, pn protocol.PacketNumber, ad []byte) ([]byte, error) {
	dec, err := o.LongHeaderOpener.Open(dst, src, pn, ad)
	if err == nil && !o.dropped {
		o.dropInitialKeys()
		o.dropped = true
	}
	return dec, err
}
