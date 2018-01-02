package ubiquity

// In this file, we include chain ranking functions based on security and performance
import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"crypto/x509"
	"time"

	"github.com/cloudflare/cfssl/helpers"
)

// Compute the priority of different hash algorithm based on security
// SHA2 > SHA1 >> MD = Others = Unknown
func hashPriority(cert *x509.Certificate) int {
	switch cert.SignatureAlgorithm {
	case x509.ECDSAWithSHA1, x509.DSAWithSHA1, x509.SHA1WithRSA:
		return 10
	case x509.ECDSAWithSHA256, x509.ECDSAWithSHA384, x509.ECDSAWithSHA512,
		x509.DSAWithSHA256, x509.SHA256WithRSA, x509.SHA384WithRSA,
		x509.SHA512WithRSA:
		return 100
	default:
		return 0
	}
}

// Compute the priority of different key algorithm based performance and security
// ECDSA>RSA>DSA>Unknown
func keyAlgoPriority(cert *x509.Certificate) int {
	switch cert.PublicKeyAlgorithm {
	case x509.ECDSA:
		switch cert.PublicKey.(*ecdsa.PublicKey).Curve {
		case elliptic.P256():
			return 100
		case elliptic.P384():
			return 120
		case elliptic.P521():
			return 140
		default:
			return 100
		}
	case x509.RSA:
		switch cert.PublicKey.(*rsa.PublicKey).N.BitLen() {
		case 4096:
			return 70
		case 3072:
			return 50
		case 2048:
			return 30
		// key size <= 1024 are discouraged.
		default:
			return 0
		}
	// we do not want to bundle a DSA cert.
	case x509.DSA:
		return 0
	default:
		return 0
	}
}

// HashPriority returns the hash priority of the chain as the average of hash priority of certs in it.
func HashPriority(certs []*x509.Certificate) int {
	ret := 0.0
	for i, cert := range certs {
		f1 := 1.0 / (float64(i) + 1.0)
		f2 := 1.0 - f1
		ret = ret*f2 + float64(hashPriority(cert))*f1
	}
	return int(ret)
}

// KeyAlgoPriority returns the key algorithm priority of the chain as the average of key algorithm priority of certs in it.
func KeyAlgoPriority(certs []*x509.Certificate) int {
	ret := 0.0
	for i, cert := range certs {
		f1 := 1.0 / (float64(i) + 1.0)
		f2 := 1.0 - f1
		ret = float64(keyAlgoPriority(cert))*f1 + ret*f2
	}
	return int(ret)
}

// CompareChainHashPriority ranks chains with more current hash functions higher.
func CompareChainHashPriority(chain1, chain2 []*x509.Certificate) int {
	hp1 := HashPriority(chain1)
	hp2 := HashPriority(chain2)
	return hp1 - hp2
}

// CompareChainKeyAlgoPriority ranks chains with more current key algorithm higher.
func CompareChainKeyAlgoPriority(chain1, chain2 []*x509.Certificate) int {
	kap1 := KeyAlgoPriority(chain1)
	kap2 := KeyAlgoPriority(chain2)
	return kap1 - kap2
}

// CompareChainCryptoSuite ranks chains with more current crypto suite higher.
func CompareChainCryptoSuite(chain1, chain2 []*x509.Certificate) int {
	cs1 := HashPriority(chain1) + KeyAlgoPriority(chain1)
	cs2 := HashPriority(chain2) + KeyAlgoPriority(chain2)
	return cs1 - cs2
}

// CompareChainLength ranks shorter chain higher.
func CompareChainLength(chain1, chain2 []*x509.Certificate) int {
	return len(chain2) - len(chain1)
}

func compareTime(t1, t2 time.Time) int {
	if t1.After(t2) {
		return 1
	} else if t1.Before(t2) {
		return -1
	}
	return 0
}

// CompareChainExpiry ranks chain that lasts longer higher.
func CompareChainExpiry(chain1, chain2 []*x509.Certificate) int {
	t1 := helpers.ExpiryTime(chain1)
	t2 := helpers.ExpiryTime(chain2)
	return compareTime(t1, t2)
}
