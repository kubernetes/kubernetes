package dns

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"math/big"

	"golang.org/x/crypto/ed25519"
)

// Generate generates a DNSKEY of the given bit size.
// The public part is put inside the DNSKEY record.
// The Algorithm in the key must be set as this will define
// what kind of DNSKEY will be generated.
// The ECDSA algorithms imply a fixed keysize, in that case
// bits should be set to the size of the algorithm.
func (k *DNSKEY) Generate(bits int) (crypto.PrivateKey, error) {
	switch k.Algorithm {
	case DSA, DSANSEC3SHA1:
		if bits != 1024 {
			return nil, ErrKeySize
		}
	case RSAMD5, RSASHA1, RSASHA256, RSASHA1NSEC3SHA1:
		if bits < 512 || bits > 4096 {
			return nil, ErrKeySize
		}
	case RSASHA512:
		if bits < 1024 || bits > 4096 {
			return nil, ErrKeySize
		}
	case ECDSAP256SHA256:
		if bits != 256 {
			return nil, ErrKeySize
		}
	case ECDSAP384SHA384:
		if bits != 384 {
			return nil, ErrKeySize
		}
	case ED25519:
		if bits != 256 {
			return nil, ErrKeySize
		}
	}

	switch k.Algorithm {
	case DSA, DSANSEC3SHA1:
		params := new(dsa.Parameters)
		if err := dsa.GenerateParameters(params, rand.Reader, dsa.L1024N160); err != nil {
			return nil, err
		}
		priv := new(dsa.PrivateKey)
		priv.PublicKey.Parameters = *params
		err := dsa.GenerateKey(priv, rand.Reader)
		if err != nil {
			return nil, err
		}
		k.setPublicKeyDSA(params.Q, params.P, params.G, priv.PublicKey.Y)
		return priv, nil
	case RSAMD5, RSASHA1, RSASHA256, RSASHA512, RSASHA1NSEC3SHA1:
		priv, err := rsa.GenerateKey(rand.Reader, bits)
		if err != nil {
			return nil, err
		}
		k.setPublicKeyRSA(priv.PublicKey.E, priv.PublicKey.N)
		return priv, nil
	case ECDSAP256SHA256, ECDSAP384SHA384:
		var c elliptic.Curve
		switch k.Algorithm {
		case ECDSAP256SHA256:
			c = elliptic.P256()
		case ECDSAP384SHA384:
			c = elliptic.P384()
		}
		priv, err := ecdsa.GenerateKey(c, rand.Reader)
		if err != nil {
			return nil, err
		}
		k.setPublicKeyECDSA(priv.PublicKey.X, priv.PublicKey.Y)
		return priv, nil
	case ED25519:
		pub, priv, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return nil, err
		}
		k.setPublicKeyED25519(pub)
		return priv, nil
	default:
		return nil, ErrAlg
	}
}

// Set the public key (the value E and N)
func (k *DNSKEY) setPublicKeyRSA(_E int, _N *big.Int) bool {
	if _E == 0 || _N == nil {
		return false
	}
	buf := exponentToBuf(_E)
	buf = append(buf, _N.Bytes()...)
	k.PublicKey = toBase64(buf)
	return true
}

// Set the public key for Elliptic Curves
func (k *DNSKEY) setPublicKeyECDSA(_X, _Y *big.Int) bool {
	if _X == nil || _Y == nil {
		return false
	}
	var intlen int
	switch k.Algorithm {
	case ECDSAP256SHA256:
		intlen = 32
	case ECDSAP384SHA384:
		intlen = 48
	}
	k.PublicKey = toBase64(curveToBuf(_X, _Y, intlen))
	return true
}

// Set the public key for DSA
func (k *DNSKEY) setPublicKeyDSA(_Q, _P, _G, _Y *big.Int) bool {
	if _Q == nil || _P == nil || _G == nil || _Y == nil {
		return false
	}
	buf := dsaToBuf(_Q, _P, _G, _Y)
	k.PublicKey = toBase64(buf)
	return true
}

// Set the public key for Ed25519
func (k *DNSKEY) setPublicKeyED25519(_K ed25519.PublicKey) bool {
	if _K == nil {
		return false
	}
	k.PublicKey = toBase64(_K)
	return true
}

// Set the public key (the values E and N) for RSA
// RFC 3110: Section 2. RSA Public KEY Resource Records
func exponentToBuf(_E int) []byte {
	var buf []byte
	i := big.NewInt(int64(_E)).Bytes()
	if len(i) < 256 {
		buf = make([]byte, 1, 1+len(i))
		buf[0] = uint8(len(i))
	} else {
		buf = make([]byte, 3, 3+len(i))
		buf[0] = 0
		buf[1] = uint8(len(i) >> 8)
		buf[2] = uint8(len(i))
	}
	buf = append(buf, i...)
	return buf
}

// Set the public key for X and Y for Curve. The two
// values are just concatenated.
func curveToBuf(_X, _Y *big.Int, intlen int) []byte {
	buf := intToBytes(_X, intlen)
	buf = append(buf, intToBytes(_Y, intlen)...)
	return buf
}

// Set the public key for X and Y for Curve. The two
// values are just concatenated.
func dsaToBuf(_Q, _P, _G, _Y *big.Int) []byte {
	t := divRoundUp(divRoundUp(_G.BitLen(), 8)-64, 8)
	buf := []byte{byte(t)}
	buf = append(buf, intToBytes(_Q, 20)...)
	buf = append(buf, intToBytes(_P, 64+t*8)...)
	buf = append(buf, intToBytes(_G, 64+t*8)...)
	buf = append(buf, intToBytes(_Y, 64+t*8)...)
	return buf
}
