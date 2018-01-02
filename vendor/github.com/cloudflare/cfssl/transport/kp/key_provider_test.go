package kp

import (
	"os"
	"testing"

	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/transport/core"
)

const (
	testKey  = "testdata/test.key"
	testCert = "testdata/test.pem"
)

var testIdentity = &core.Identity{
	Request: &csr.CertificateRequest{
		CN: "localhost test certificate",
	},
	Profiles: map[string]map[string]string{
		"paths": {
			"private_key": testKey,
			"certificate": testCert,
		},
	},
}

func removeIfPresent(path string) error {
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		return os.Remove(path)
	}
	return nil
}

func TestMain(m *testing.M) {
	exitCode := m.Run()

	err := removeIfPresent(testKey)
	if err == nil {
		err = removeIfPresent(testCert)
	}

	if err != nil {
		os.Exit(1)
	}
	os.Exit(exitCode)
}

var kp KeyProvider

func TestNewStandardProvider(t *testing.T) {
	var err error
	kp, err = NewStandardProvider(testIdentity)
	if err != nil {
		t.Fatalf("%v", err)
	}

	if kp.Ready() {
		t.Fatalf("key provider should not be ready yet")
	}

	if err = kp.Check(); err != nil {
		t.Fatalf("calling check should return no error")
	}

	if nil != kp.Certificate() {
		t.Fatal("key provider should not have a certificate yet")
	}

	if kp.Ready() {
		t.Fatal("key provider should not be ready")
	}

	if !kp.Persistent() {
		t.Fatal("key provider should be persistent")
	}
}

func TestGenerate(t *testing.T) {
	err := kp.Load()
	if err == nil {
		t.Fatal("key provider shouldn't have a key yet")
	}

	err = kp.Generate("ecdsa", 256)
	if err != nil {
		t.Fatalf("key provider couldn't generate key: %v", err)
	}
}
