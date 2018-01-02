package sign

import (
	"bytes"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"io"
	"math/rand"
	"strings"
	"testing"
)

func generatePEM(randReader io.Reader, password []byte) (buf *bytes.Buffer, err error) {
	k, err := rsa.GenerateKey(randReader, 1024)
	if err != nil {
		return nil, err
	}

	derBytes := x509.MarshalPKCS1PrivateKey(k)

	var block *pem.Block
	if password != nil {
		block, err = x509.EncryptPEMBlock(randReader, "RSA PRIVATE KEY", derBytes, password, x509.PEMCipherAES128)
	} else {
		block = &pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: derBytes,
		}
	}

	buf = &bytes.Buffer{}
	err = pem.Encode(buf, block)
	return buf, err
}

func TestLoadPemPrivKey(t *testing.T) {
	reader, err := generatePEM(newRandomReader(rand.New(rand.NewSource(1))), nil)
	if err != nil {
		t.Errorf("Unexpected pem generation err %s", err.Error())
	}

	privKey, err := LoadPEMPrivKey(reader)
	if err != nil {
		t.Errorf("Unexpected key load error, %s", err.Error())
	}
	if privKey == nil {
		t.Errorf("Expected valid privKey, but got nil")
	}
}

func TestLoadPemPrivKeyInvalidPEM(t *testing.T) {
	reader := strings.NewReader("invalid PEM data")
	privKey, err := LoadPEMPrivKey(reader)

	if err == nil {
		t.Errorf("Expected error invalid PEM data error")
	}
	if privKey != nil {
		t.Errorf("Expected nil privKey but got %#v", privKey)
	}
}

func TestLoadEncryptedPEMPrivKey(t *testing.T) {
	reader, err := generatePEM(newRandomReader(rand.New(rand.NewSource(1))), []byte("password"))
	if err != nil {
		t.Errorf("Unexpected pem generation err %s", err.Error())
	}

	privKey, err := LoadEncryptedPEMPrivKey(reader, []byte("password"))

	if err != nil {
		t.Errorf("Unexpected key load error, %s", err.Error())
	}
	if privKey == nil {
		t.Errorf("Expected valid privKey, but got nil")
	}
}

func TestLoadEncryptedPEMPrivKeyWrongPassword(t *testing.T) {
	reader, err := generatePEM(newRandomReader(rand.New(rand.NewSource(1))), []byte("password"))
	privKey, err := LoadEncryptedPEMPrivKey(reader, []byte("wrong password"))

	if err == nil {
		t.Errorf("Expected error invalid PEM data error")
	}
	if privKey != nil {
		t.Errorf("Expected nil privKey but got %#v", privKey)
	}
}
