package handshake

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/tls"
	"encoding/binary"
	"fmt"

	"golang.org/x/crypto/chacha20"

	"github.com/lucas-clemente/quic-go/internal/qtls"
)

type headerProtector interface {
	EncryptHeader(sample []byte, firstByte *byte, hdrBytes []byte)
	DecryptHeader(sample []byte, firstByte *byte, hdrBytes []byte)
}

func newHeaderProtector(suite *qtls.CipherSuiteTLS13, trafficSecret []byte, isLongHeader bool) headerProtector {
	switch suite.ID {
	case tls.TLS_AES_128_GCM_SHA256, tls.TLS_AES_256_GCM_SHA384:
		return newAESHeaderProtector(suite, trafficSecret, isLongHeader)
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		return newChaChaHeaderProtector(suite, trafficSecret, isLongHeader)
	default:
		panic(fmt.Sprintf("Invalid cipher suite id: %d", suite.ID))
	}
}

type aesHeaderProtector struct {
	mask         []byte
	block        cipher.Block
	isLongHeader bool
}

var _ headerProtector = &aesHeaderProtector{}

func newAESHeaderProtector(suite *qtls.CipherSuiteTLS13, trafficSecret []byte, isLongHeader bool) headerProtector {
	hpKey := hkdfExpandLabel(suite.Hash, trafficSecret, []byte{}, "quic hp", suite.KeyLen)
	block, err := aes.NewCipher(hpKey)
	if err != nil {
		panic(fmt.Sprintf("error creating new AES cipher: %s", err))
	}
	return &aesHeaderProtector{
		block:        block,
		mask:         make([]byte, block.BlockSize()),
		isLongHeader: isLongHeader,
	}
}

func (p *aesHeaderProtector) DecryptHeader(sample []byte, firstByte *byte, hdrBytes []byte) {
	p.apply(sample, firstByte, hdrBytes)
}

func (p *aesHeaderProtector) EncryptHeader(sample []byte, firstByte *byte, hdrBytes []byte) {
	p.apply(sample, firstByte, hdrBytes)
}

func (p *aesHeaderProtector) apply(sample []byte, firstByte *byte, hdrBytes []byte) {
	if len(sample) != len(p.mask) {
		panic("invalid sample size")
	}
	p.block.Encrypt(p.mask, sample)
	if p.isLongHeader {
		*firstByte ^= p.mask[0] & 0xf
	} else {
		*firstByte ^= p.mask[0] & 0x1f
	}
	for i := range hdrBytes {
		hdrBytes[i] ^= p.mask[i+1]
	}
}

type chachaHeaderProtector struct {
	mask [5]byte

	key          [32]byte
	isLongHeader bool
}

var _ headerProtector = &chachaHeaderProtector{}

func newChaChaHeaderProtector(suite *qtls.CipherSuiteTLS13, trafficSecret []byte, isLongHeader bool) headerProtector {
	hpKey := hkdfExpandLabel(suite.Hash, trafficSecret, []byte{}, "quic hp", suite.KeyLen)

	p := &chachaHeaderProtector{
		isLongHeader: isLongHeader,
	}
	copy(p.key[:], hpKey)
	return p
}

func (p *chachaHeaderProtector) DecryptHeader(sample []byte, firstByte *byte, hdrBytes []byte) {
	p.apply(sample, firstByte, hdrBytes)
}

func (p *chachaHeaderProtector) EncryptHeader(sample []byte, firstByte *byte, hdrBytes []byte) {
	p.apply(sample, firstByte, hdrBytes)
}

func (p *chachaHeaderProtector) apply(sample []byte, firstByte *byte, hdrBytes []byte) {
	if len(sample) != 16 {
		panic("invalid sample size")
	}
	for i := 0; i < 5; i++ {
		p.mask[i] = 0
	}
	cipher, err := chacha20.NewUnauthenticatedCipher(p.key[:], sample[4:])
	if err != nil {
		panic(err)
	}
	cipher.SetCounter(binary.LittleEndian.Uint32(sample[:4]))
	cipher.XORKeyStream(p.mask[:], p.mask[:])
	p.applyMask(firstByte, hdrBytes)
}

func (p *chachaHeaderProtector) applyMask(firstByte *byte, hdrBytes []byte) {
	if p.isLongHeader {
		*firstByte ^= p.mask[0] & 0xf
	} else {
		*firstByte ^= p.mask[0] & 0x1f
	}
	for i := range hdrBytes {
		hdrBytes[i] ^= p.mask[i+1]
	}
}
