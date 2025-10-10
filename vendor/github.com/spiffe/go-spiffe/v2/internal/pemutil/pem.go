package pemutil

import (
	"crypto"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
)

const (
	certType string = "CERTIFICATE"
	keyType  string = "PRIVATE KEY"
)

func ParseCertificates(certsBytes []byte) ([]*x509.Certificate, error) {
	objects, err := parseBlocks(certsBytes, certType)
	if err != nil {
		return nil, err
	}

	certs := []*x509.Certificate{}
	for _, object := range objects {
		cert, ok := object.(*x509.Certificate)
		if !ok {
			return nil, fmt.Errorf("expected *x509.Certificate; got %T", object)
		}
		certs = append(certs, cert)
	}

	return certs, nil
}

func ParsePrivateKey(keyBytes []byte) (crypto.PrivateKey, error) {
	objects, err := parseBlocks(keyBytes, keyType)
	if err != nil {
		return nil, err
	}
	if len(objects) == 0 {
		return nil, nil
	}

	privateKey, ok := objects[0].(crypto.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("expected crypto.PrivateKey; got %T", objects[0])
	}
	return privateKey, nil
}

func EncodePKCS8PrivateKey(privateKey interface{}) ([]byte, error) {
	keyBytes, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return nil, err
	}

	return pem.EncodeToMemory(&pem.Block{
		Type:  keyType,
		Bytes: keyBytes,
	}), nil
}

func EncodeCertificates(certificates []*x509.Certificate) []byte {
	pemBytes := []byte{}
	for _, cert := range certificates {
		pemBytes = append(pemBytes, pem.EncodeToMemory(&pem.Block{
			Type:  certType,
			Bytes: cert.Raw,
		})...)
	}
	return pemBytes
}

func parseBlocks(blocksBytes []byte, expectedType string) ([]interface{}, error) {
	objects := []interface{}{}
	var foundBlocks = false
	for {
		if len(blocksBytes) == 0 {
			if len(objects) == 0 && !foundBlocks {
				return nil, errors.New("no PEM blocks found")
			}
			return objects, nil
		}
		object, rest, foundBlock, err := parseBlock(blocksBytes, expectedType)
		blocksBytes = rest
		if foundBlock {
			foundBlocks = true
		}
		switch {
		case err != nil:
			return nil, err
		case object != nil:
			objects = append(objects, object)
		}
	}
}

func parseBlock(pemBytes []byte, pemType string) (interface{}, []byte, bool, error) {
	pemBlock, rest := pem.Decode(pemBytes)
	if pemBlock == nil {
		return nil, nil, false, nil
	}

	if pemBlock.Type != pemType {
		return nil, rest, true, nil
	}

	var object interface{}
	var err error
	switch pemType {
	case certType:
		object, err = x509.ParseCertificate(pemBlock.Bytes)
	case keyType:
		object, err = x509.ParsePKCS8PrivateKey(pemBlock.Bytes)
	default:
		err = fmt.Errorf("PEM type not supported: %q", pemType)
	}

	if err != nil {
		return nil, nil, false, err
	}

	return object, rest, true, nil
}
