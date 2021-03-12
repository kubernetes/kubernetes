package dns

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/rsa"
	"encoding/binary"
	"math/big"
	"strings"
	"time"
)

// Sign signs a dns.Msg. It fills the signature with the appropriate data.
// The SIG record should have the SignerName, KeyTag, Algorithm, Inception
// and Expiration set.
func (rr *SIG) Sign(k crypto.Signer, m *Msg) ([]byte, error) {
	if k == nil {
		return nil, ErrPrivKey
	}
	if rr.KeyTag == 0 || len(rr.SignerName) == 0 || rr.Algorithm == 0 {
		return nil, ErrKey
	}

	rr.Hdr = RR_Header{Name: ".", Rrtype: TypeSIG, Class: ClassANY, Ttl: 0}
	rr.OrigTtl, rr.TypeCovered, rr.Labels = 0, 0, 0

	buf := make([]byte, m.Len()+Len(rr))
	mbuf, err := m.PackBuffer(buf)
	if err != nil {
		return nil, err
	}
	if &buf[0] != &mbuf[0] {
		return nil, ErrBuf
	}
	off, err := PackRR(rr, buf, len(mbuf), nil, false)
	if err != nil {
		return nil, err
	}
	buf = buf[:off:cap(buf)]

	hash, ok := AlgorithmToHash[rr.Algorithm]
	if !ok {
		return nil, ErrAlg
	}

	hasher := hash.New()
	// Write SIG rdata
	hasher.Write(buf[len(mbuf)+1+2+2+4+2:])
	// Write message
	hasher.Write(buf[:len(mbuf)])

	signature, err := sign(k, hasher.Sum(nil), hash, rr.Algorithm)
	if err != nil {
		return nil, err
	}

	rr.Signature = toBase64(signature)

	buf = append(buf, signature...)
	if len(buf) > int(^uint16(0)) {
		return nil, ErrBuf
	}
	// Adjust sig data length
	rdoff := len(mbuf) + 1 + 2 + 2 + 4
	rdlen := binary.BigEndian.Uint16(buf[rdoff:])
	rdlen += uint16(len(signature))
	binary.BigEndian.PutUint16(buf[rdoff:], rdlen)
	// Adjust additional count
	adc := binary.BigEndian.Uint16(buf[10:])
	adc++
	binary.BigEndian.PutUint16(buf[10:], adc)
	return buf, nil
}

// Verify validates the message buf using the key k.
// It's assumed that buf is a valid message from which rr was unpacked.
func (rr *SIG) Verify(k *KEY, buf []byte) error {
	if k == nil {
		return ErrKey
	}
	if rr.KeyTag == 0 || len(rr.SignerName) == 0 || rr.Algorithm == 0 {
		return ErrKey
	}

	var hash crypto.Hash
	switch rr.Algorithm {
	case RSASHA1:
		hash = crypto.SHA1
	case RSASHA256, ECDSAP256SHA256:
		hash = crypto.SHA256
	case ECDSAP384SHA384:
		hash = crypto.SHA384
	case RSASHA512:
		hash = crypto.SHA512
	default:
		return ErrAlg
	}
	hasher := hash.New()

	buflen := len(buf)
	qdc := binary.BigEndian.Uint16(buf[4:])
	anc := binary.BigEndian.Uint16(buf[6:])
	auc := binary.BigEndian.Uint16(buf[8:])
	adc := binary.BigEndian.Uint16(buf[10:])
	offset := headerSize
	var err error
	for i := uint16(0); i < qdc && offset < buflen; i++ {
		_, offset, err = UnpackDomainName(buf, offset)
		if err != nil {
			return err
		}
		// Skip past Type and Class
		offset += 2 + 2
	}
	for i := uint16(1); i < anc+auc+adc && offset < buflen; i++ {
		_, offset, err = UnpackDomainName(buf, offset)
		if err != nil {
			return err
		}
		// Skip past Type, Class and TTL
		offset += 2 + 2 + 4
		if offset+1 >= buflen {
			continue
		}
		rdlen := binary.BigEndian.Uint16(buf[offset:])
		offset += 2
		offset += int(rdlen)
	}
	if offset >= buflen {
		return &Error{err: "overflowing unpacking signed message"}
	}

	// offset should be just prior to SIG
	bodyend := offset
	// owner name SHOULD be root
	_, offset, err = UnpackDomainName(buf, offset)
	if err != nil {
		return err
	}
	// Skip Type, Class, TTL, RDLen
	offset += 2 + 2 + 4 + 2
	sigstart := offset
	// Skip Type Covered, Algorithm, Labels, Original TTL
	offset += 2 + 1 + 1 + 4
	if offset+4+4 >= buflen {
		return &Error{err: "overflow unpacking signed message"}
	}
	expire := binary.BigEndian.Uint32(buf[offset:])
	offset += 4
	incept := binary.BigEndian.Uint32(buf[offset:])
	offset += 4
	now := uint32(time.Now().Unix())
	if now < incept || now > expire {
		return ErrTime
	}
	// Skip key tag
	offset += 2
	var signername string
	signername, offset, err = UnpackDomainName(buf, offset)
	if err != nil {
		return err
	}
	// If key has come from the DNS name compression might
	// have mangled the case of the name
	if !strings.EqualFold(signername, k.Header().Name) {
		return &Error{err: "signer name doesn't match key name"}
	}
	sigend := offset
	hasher.Write(buf[sigstart:sigend])
	hasher.Write(buf[:10])
	hasher.Write([]byte{
		byte((adc - 1) << 8),
		byte(adc - 1),
	})
	hasher.Write(buf[12:bodyend])

	hashed := hasher.Sum(nil)
	sig := buf[sigend:]
	switch k.Algorithm {
	case RSASHA1, RSASHA256, RSASHA512:
		pk := k.publicKeyRSA()
		if pk != nil {
			return rsa.VerifyPKCS1v15(pk, hash, hashed, sig)
		}
	case ECDSAP256SHA256, ECDSAP384SHA384:
		pk := k.publicKeyECDSA()
		r := new(big.Int).SetBytes(sig[:len(sig)/2])
		s := new(big.Int).SetBytes(sig[len(sig)/2:])
		if pk != nil {
			if ecdsa.Verify(pk, hashed, r, s) {
				return nil
			}
			return ErrSig
		}
	}
	return ErrKeyAlg
}
