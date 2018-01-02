package libtrust

import (
	"encoding/pem"
	"io/ioutil"
	"net"
	"os"
	"path"
	"testing"
)

func TestGenerateCertificates(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	_, err = GenerateSelfSignedServerCert(key, []string{"localhost"}, []net.IP{net.ParseIP("127.0.0.1")})
	if err != nil {
		t.Fatal(err)
	}

	_, err = GenerateSelfSignedClientCert(key)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGenerateCACertPool(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	caKey1, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	caKey2, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	_, err = GenerateCACertPool(key, []PublicKey{caKey1.PublicKey(), caKey2.PublicKey()})
	if err != nil {
		t.Fatal(err)
	}
}

func TestLoadCertificates(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	caKey1, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}
	caKey2, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	cert1, err := GenerateCACert(caKey1, key)
	if err != nil {
		t.Fatal(err)
	}
	cert2, err := GenerateCACert(caKey2, key)
	if err != nil {
		t.Fatal(err)
	}

	d, err := ioutil.TempDir("/tmp", "cert-test")
	if err != nil {
		t.Fatal(err)
	}
	caFile := path.Join(d, "ca.pem")
	f, err := os.OpenFile(caFile, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}

	err = pem.Encode(f, &pem.Block{Type: "CERTIFICATE", Bytes: cert1.Raw})
	if err != nil {
		t.Fatal(err)
	}
	err = pem.Encode(f, &pem.Block{Type: "CERTIFICATE", Bytes: cert2.Raw})
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	certs, err := LoadCertificateBundle(caFile)
	if err != nil {
		t.Fatal(err)
	}
	if len(certs) != 2 {
		t.Fatalf("Wrong number of certs received, expected: %d, received %d", 2, len(certs))
	}

	pool, err := LoadCertificatePool(caFile)
	if err != nil {
		t.Fatal(err)
	}

	if len(pool.Subjects()) != 2 {
		t.Fatalf("Invalid certificate pool")
	}
}
