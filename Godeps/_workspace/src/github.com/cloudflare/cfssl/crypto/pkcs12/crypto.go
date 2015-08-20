package pkcs12

// Much credit to at Microsoft's Azure project https://github.com/Azure/go-pkcs12/blob/master/pkcs12.go,
// from which the following decryption code was adapted under the MIT License.  The functions in this
// file implement the decryption of pkcs12 formatted data as described in
// https://tools.ietf.org/html/rfc2898#section-6.1.2

import (
	"bytes"
	"crypto/cipher"
	"crypto/des"
	"crypto/sha1"
	"crypto/x509/pkix"
	"encoding/asn1"
	"errors"

	"github.com/cloudflare/cfssl/crypto/pkcs12/pbkdf"
	rc2 "github.com/dgryski/go-rc2"
)

type pbeParams struct {
	Salt       []byte
	Iterations int
}

const (
	pbeWithSHAAnd3KeyTripleDESCBC = "pbeWithSHAAnd3-KeyTripleDES-CBC"
	pbewithSHAAnd40BitRC2CBC      = "pbewithSHAAnd40BitRC2-CBC"
)

var algByOID = map[string]string{
	"1.2.840.113549.1.12.1.3": pbeWithSHAAnd3KeyTripleDESCBC,
	"1.2.840.113549.1.12.1.6": pbewithSHAAnd40BitRC2CBC,
}

var blockcodeByAlg = map[string]func(key []byte) (cipher.Block, error){
	pbeWithSHAAnd3KeyTripleDESCBC: des.NewTripleDESCipher,
	pbewithSHAAnd40BitRC2CBC: func(key []byte) (cipher.Block, error) {
		return rc2.New(key, len(key)*8)
	},
}

var (
	deriveKeyByAlg = map[string]func(salt, password []byte, iterations int) []byte{
		pbeWithSHAAnd3KeyTripleDESCBC: func(salt, password []byte, iterations int) []byte {
			return pbkdf.PBKDF(sha1Sum, 20, 64, salt, password, iterations, 1, 24)
		},
		pbewithSHAAnd40BitRC2CBC: func(salt, password []byte, iterations int) []byte {
			return pbkdf.PBKDF(sha1Sum, 20, 64, salt, password, iterations, 1, 5)
		},
	}
	deriveIVByAlg = map[string]func(salt, password []byte, iterations int) []byte{
		pbeWithSHAAnd3KeyTripleDESCBC: func(salt, password []byte, iterations int) []byte {
			return pbkdf.PBKDF(sha1Sum, 20, 64, salt, password, iterations, 2, 8)
		},
		pbewithSHAAnd40BitRC2CBC: func(salt, password []byte, iterations int) []byte {
			return pbkdf.PBKDF(sha1Sum, 20, 64, salt, password, iterations, 2, 8)
		},
	}
)

func sha1Sum(in []byte) []byte {
	sum := sha1.Sum(in)
	return sum[:]
}

//For use decrypting key and certificates, apart from a few minor changes
//this entire function was taken from Azure: https://github.com/Azure/go-pkcs12/blob/master/crypto.go
func decrypt(algorithm pkix.AlgorithmIdentifier, encrypted, password []byte) (decrypted []byte, err error) {
	// Generate a CBC Decrypter
	cbc, err := cbcGen(algorithm, password)
	if err != nil {
		return nil, err
	}
	// Decrypting the secret data
	decrypted = make([]byte, len(encrypted))
	cbc.CryptBlocks(decrypted, encrypted)

	if psLen := int(decrypted[len(decrypted)-1]); psLen > 0 && psLen < 9 {
		m := decrypted[:len(decrypted)-psLen]
		ps := decrypted[len(decrypted)-psLen:]
		if bytes.Compare(ps, bytes.Repeat([]byte{byte(psLen)}, psLen)) != 0 {
			return nil, errors.New("decryption error, incorrect padding")
		}
		decrypted = m
	} else {
		return nil, errors.New("decryption error, incorrect padding")
	}

	return
}

// Generating a cbc cipher decoder
func cbcGen(algorithm pkix.AlgorithmIdentifier, password []byte) (cipher.BlockMode, error) {
	algorithmName, supported := algByOID[algorithm.Algorithm.String()]
	if !supported {
		return nil, errors.New("Algorithm not supported")
	}
	var params pbeParams
	if _, err := asn1.Unmarshal(algorithm.Parameters.FullBytes, &params); err != nil {
		return nil, err
	}
	k := deriveKeyByAlg[algorithmName](params.Salt, password, params.Iterations)
	iv := deriveIVByAlg[algorithmName](params.Salt, password, params.Iterations)

	code, err := blockcodeByAlg[algorithmName](k)
	if err != nil {
		return nil, err
	}

	cbc := cipher.NewCBCDecrypter(code, iv)
	return cbc, nil

}
