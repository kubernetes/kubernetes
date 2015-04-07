// SIG(0)
//
// From RFC 2931:
//
//     SIG(0) provides protection for DNS transactions and requests ....
//     ... protection for glue records, DNS requests, protection for message headers
//     on requests and responses, and protection of the overall integrity of a response.
//
// It works like TSIG, except that SIG(0) uses public key cryptography, instead of the shared
// secret approach in TSIG.
// Supported algorithms: DSA, ECDSAP256SHA256, ECDSAP384SHA384, RSASHA1, RSASHA256 and
// RSASHA512.
//
// Signing subsequent messages in multi-message sessions is not implemented.
//
package dns

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rand"
	"crypto/rsa"
	"math/big"
	"strings"
	"time"
)

// Sign signs a dns.Msg. It fills the signature with the appropriate data.
// The SIG record should have the SignerName, KeyTag, Algorithm, Inception
// and Expiration set.
func (rr *SIG) Sign(k PrivateKey, m *Msg) ([]byte, error) {
	if k == nil {
		return nil, ErrPrivKey
	}
	if rr.KeyTag == 0 || len(rr.SignerName) == 0 || rr.Algorithm == 0 {
		return nil, ErrKey
	}
	rr.Header().Rrtype = TypeSIG
	rr.Header().Class = ClassANY
	rr.Header().Ttl = 0
	rr.Header().Name = "."
	rr.OrigTtl = 0
	rr.TypeCovered = 0
	rr.Labels = 0

	buf := make([]byte, m.Len()+rr.len())
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
	var hash crypto.Hash
	var intlen int
	switch rr.Algorithm {
	case DSA, RSASHA1:
		hash = crypto.SHA1
	case RSASHA256, ECDSAP256SHA256:
		hash = crypto.SHA256
		intlen = 32
	case ECDSAP384SHA384:
		hash = crypto.SHA384
		intlen = 48
	case RSASHA512:
		hash = crypto.SHA512
	default:
		return nil, ErrAlg
	}
	hasher := hash.New()
	// Write SIG rdata
	hasher.Write(buf[len(mbuf)+1+2+2+4+2:])
	// Write message
	hasher.Write(buf[:len(mbuf)])
	hashed := hasher.Sum(nil)

	var sig []byte
	switch p := k.(type) {
	case *dsa.PrivateKey:
		t := divRoundUp(divRoundUp(p.PublicKey.Y.BitLen(), 8)-64, 8)
		r1, s1, err := dsa.Sign(rand.Reader, p, hashed)
		if err != nil {
			return nil, err
		}
		sig = append(sig, byte(t))
		sig = append(sig, intToBytes(r1, 20)...)
		sig = append(sig, intToBytes(s1, 20)...)
	case *rsa.PrivateKey:
		sig, err = rsa.SignPKCS1v15(rand.Reader, p, hash, hashed)
		if err != nil {
			return nil, err
		}
	case *ecdsa.PrivateKey:
		r1, s1, err := ecdsa.Sign(rand.Reader, p, hashed)
		if err != nil {
			return nil, err
		}
		sig = intToBytes(r1, intlen)
		sig = append(sig, intToBytes(s1, intlen)...)
	default:
		return nil, ErrAlg
	}
	rr.Signature = toBase64(sig)
	buf = append(buf, sig...)
	if len(buf) > int(^uint16(0)) {
		return nil, ErrBuf
	}
	// Adjust sig data length
	rdoff := len(mbuf) + 1 + 2 + 2 + 4
	rdlen, _ := unpackUint16(buf, rdoff)
	rdlen += uint16(len(sig))
	buf[rdoff], buf[rdoff+1] = packUint16(rdlen)
	// Adjust additional count
	adc, _ := unpackUint16(buf, 10)
	adc += 1
	buf[10], buf[11] = packUint16(adc)
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
	case DSA, RSASHA1:
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
	qdc, _ := unpackUint16(buf, 4)
	anc, _ := unpackUint16(buf, 6)
	auc, _ := unpackUint16(buf, 8)
	adc, offset := unpackUint16(buf, 10)
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
		var rdlen uint16
		rdlen, offset = unpackUint16(buf, offset)
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
	expire := uint32(buf[offset])<<24 | uint32(buf[offset+1])<<16 | uint32(buf[offset+2])<<8 | uint32(buf[offset+3])
	offset += 4
	incept := uint32(buf[offset])<<24 | uint32(buf[offset+1])<<16 | uint32(buf[offset+2])<<8 | uint32(buf[offset+3])
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
	if strings.ToLower(signername) != strings.ToLower(k.Header().Name) {
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
	case DSA:
		pk := k.publicKeyDSA()
		sig = sig[1:]
		r := big.NewInt(0)
		r.SetBytes(sig[:len(sig)/2])
		s := big.NewInt(0)
		s.SetBytes(sig[len(sig)/2:])
		if pk != nil {
			if dsa.Verify(pk, hashed, r, s) {
				return nil
			}
			return ErrSig
		}
	case RSASHA1, RSASHA256, RSASHA512:
		pk := k.publicKeyRSA()
		if pk != nil {
			return rsa.VerifyPKCS1v15(pk, hash, hashed, sig)
		}
	case ECDSAP256SHA256, ECDSAP384SHA384:
		pk := k.publicKeyCurve()
		r := big.NewInt(0)
		r.SetBytes(sig[:len(sig)/2])
		s := big.NewInt(0)
		s.SetBytes(sig[len(sig)/2:])
		if pk != nil {
			if ecdsa.Verify(pk, hashed, r, s) {
				return nil
			}
			return ErrSig
		}
	}
	return ErrKeyAlg
}
