package tlsconfig

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"io"
	"io/ioutil"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

var certTemplate = x509.Certificate{
	SerialNumber: big.NewInt(199999),
	Subject: pkix.Name{
		CommonName: "test",
	},
	NotBefore: time.Now().AddDate(-1, 1, 1),
	NotAfter:  time.Now().AddDate(1, 1, 1),

	KeyUsage:    x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
	ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageCodeSigning},

	BasicConstraintsValid: true,
}

func generateCertificate(t *testing.T, signer crypto.Signer, out io.Writer) {
	derBytes, err := x509.CreateCertificate(rand.Reader, &certTemplate, &certTemplate, signer.Public(), signer)
	if err != nil {
		t.Fatal("Unable to generate a certificate", err.Error())
	}

	if err = pem.Encode(out, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		t.Fatal("Unable to write cert to file", err.Error())
	}
}

// generates a multiple-certificate file with both RSA and ECDSA certs and
// returns the filename so that cleanup can be deferred.
func generateMultiCert(t *testing.T, tempDir string) string {
	certOut, err := os.Create(filepath.Join(tempDir, "multi"))
	if err != nil {
		t.Fatal("Unable to create file to write multi-cert to", err.Error())
	}
	defer certOut.Close()

	rsaKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal("Unable to generate RSA key for multi-cert", err.Error())
	}
	ecKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal("Unable to generate ECDSA key for multi-cert", err.Error())
	}

	for _, signer := range []crypto.Signer{rsaKey, ecKey} {
		generateCertificate(t, signer, certOut)
	}

	return certOut.Name()
}

func generateCertAndKey(t *testing.T, tempDir string) (string, string) {
	rsaKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal("Unable to generate RSA key", err.Error())

	}
	keyBytes := x509.MarshalPKCS1PrivateKey(rsaKey)

	keyOut, err := os.Create(filepath.Join(tempDir, "key"))
	if err != nil {
		t.Fatal("Unable to create file to write key to", err.Error())

	}
	defer keyOut.Close()

	if err = pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: keyBytes}); err != nil {
		t.Fatal("Unable to write key to file", err.Error())
	}

	certOut, err := os.Create(filepath.Join(tempDir, "cert"))
	if err != nil {
		t.Fatal("Unable to create file to write cert to", err.Error())
	}
	defer certOut.Close()

	generateCertificate(t, rsaKey, certOut)

	return keyOut.Name(), certOut.Name()
}

func makeTempDir(t *testing.T) string {
	tempDir, err := ioutil.TempDir("", "tlsconfig-test")
	if err != nil {
		t.Fatal("Could not make a temporary directory", err.Error())
	}
	return tempDir
}

// If the cert files and directory are provided but are invalid, an error is
// returned.
func TestConfigServerTLSFailsIfUnableToLoadCerts(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	key, cert := generateCertAndKey(t, tempDir)
	ca := generateMultiCert(t, tempDir)

	tempFile, err := ioutil.TempFile("", "cert-test")
	if err != nil {
		t.Fatal("Unable to create temporary empty file")
	}
	defer os.RemoveAll(tempFile.Name())
	tempFile.Close()

	for _, badFile := range []string{"not-a-file", tempFile.Name()} {
		for i := 0; i < 3; i++ {
			files := []string{cert, key, ca}
			files[i] = badFile

			result, err := Server(Options{
				CertFile:   files[0],
				KeyFile:    files[1],
				CAFile:     files[2],
				ClientAuth: tls.VerifyClientCertIfGiven,
			})
			if err == nil || result != nil {
				t.Fatal("Expected a non-real file to error and return a nil TLS config")
			}
		}
	}
}

// If server cert and key are provided and client auth and client CA are not
// set, a tls config with only the server certs will be returned.
func TestConfigServerTLSServerCertsOnly(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	key, cert := generateCertAndKey(t, tempDir)

	keypair, err := tls.LoadX509KeyPair(cert, key)
	if err != nil {
		t.Fatal("Unable to load the generated cert and key")
	}

	tlsConfig, err := Server(Options{
		CertFile: cert,
		KeyFile:  key,
	})
	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure server TLS", err)
	}

	if len(tlsConfig.Certificates) != 1 {
		t.Fatal("Unexpected server certificates")
	}
	if len(tlsConfig.Certificates[0].Certificate) != len(keypair.Certificate) {
		t.Fatal("Unexpected server certificates")
	}
	for i, cert := range tlsConfig.Certificates[0].Certificate {
		if !bytes.Equal(cert, keypair.Certificate[i]) {
			t.Fatal("Unexpected server certificates")
		}
	}

	if !reflect.DeepEqual(tlsConfig.CipherSuites, DefaultServerAcceptedCiphers) {
		t.Fatal("Unexpected server cipher suites")
	}
	if !tlsConfig.PreferServerCipherSuites {
		t.Fatal("Expected server to prefer cipher suites")
	}
	if tlsConfig.MinVersion != tls.VersionTLS10 {
		t.Fatal("Unexpected server TLS version")
	}
}

// If client CA is provided, it will only be used if the client auth is >=
// VerifyClientCertIfGiven
func TestConfigServerTLSClientCANotSetIfClientAuthTooLow(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	key, cert := generateCertAndKey(t, tempDir)
	ca := generateMultiCert(t, tempDir)

	tlsConfig, err := Server(Options{
		CertFile:   cert,
		KeyFile:    key,
		ClientAuth: tls.RequestClientCert,
		CAFile:     ca,
	})

	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure server TLS", err)
	}

	if len(tlsConfig.Certificates) != 1 {
		t.Fatal("Unexpected server certificates")
	}
	if tlsConfig.ClientAuth != tls.RequestClientCert {
		t.Fatal("ClientAuth was not set to what was in the options")
	}
	if tlsConfig.ClientCAs != nil {
		t.Fatalf("Client CAs should never have been set")
	}
}

// If client CA is provided, it will only be used if the client auth is >=
// VerifyClientCertIfGiven
func TestConfigServerTLSClientCASet(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	key, cert := generateCertAndKey(t, tempDir)
	ca := generateMultiCert(t, tempDir)

	tlsConfig, err := Server(Options{
		CertFile:   cert,
		KeyFile:    key,
		ClientAuth: tls.VerifyClientCertIfGiven,
		CAFile:     ca,
	})

	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure server TLS", err)
	}

	if len(tlsConfig.Certificates) != 1 {
		t.Fatal("Unexpected server certificates")
	}
	if tlsConfig.ClientAuth != tls.VerifyClientCertIfGiven {
		t.Fatal("ClientAuth was not set to what was in the options")
	}
	if tlsConfig.ClientCAs == nil || len(tlsConfig.ClientCAs.Subjects()) != 2 {
		t.Fatalf("Client CAs were never set correctly")
	}
}

// The root CA is never set if InsecureSkipBoolean is set to true, but the
// default client options are set
func TestConfigClientTLSNoVerify(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	ca := generateMultiCert(t, tempDir)

	tlsConfig, err := Client(Options{CAFile: ca, InsecureSkipVerify: true})

	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure client TLS", err)
	}

	if tlsConfig.RootCAs != nil {
		t.Fatal("Should not have set Root CAs", err)
	}

	if !reflect.DeepEqual(tlsConfig.CipherSuites, clientCipherSuites) {
		t.Fatal("Unexpected client cipher suites")
	}
	if tlsConfig.MinVersion != tls.VersionTLS12 {
		t.Fatal("Unexpected client TLS version")
	}

	if tlsConfig.Certificates != nil {
		t.Fatal("Somehow client certificates were set")
	}
}

// The root CA is never set if InsecureSkipBoolean is set to false and root CA
// is not provided.
func TestConfigClientTLSNoRoot(t *testing.T) {
	tlsConfig, err := Client(Options{})

	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure client TLS", err)
	}

	if tlsConfig.RootCAs != nil {
		t.Fatal("Should not have set Root CAs", err)
	}

	if !reflect.DeepEqual(tlsConfig.CipherSuites, clientCipherSuites) {
		t.Fatal("Unexpected client cipher suites")
	}
	if tlsConfig.MinVersion != tls.VersionTLS12 {
		t.Fatal("Unexpected client TLS version")
	}

	if tlsConfig.Certificates != nil {
		t.Fatal("Somehow client certificates were set")
	}
}

// The RootCA is set if the file is provided and InsecureSkipVerify is false
func TestConfigClientTLSRootCAFileWithOneCert(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	ca := generateMultiCert(t, tempDir)

	tlsConfig, err := Client(Options{CAFile: ca})

	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure client TLS", err)
	}

	if tlsConfig.RootCAs == nil || len(tlsConfig.RootCAs.Subjects()) != 2 {
		t.Fatal("Root CAs not set properly", err)
	}
	if tlsConfig.Certificates != nil {
		t.Fatal("Somehow client certificates were set")
	}
}

// An error is returned if a root CA is provided but the file doesn't exist.
func TestConfigClientTLSNonexistentRootCAFile(t *testing.T) {
	tlsConfig, err := Client(Options{CAFile: "nonexistent"})

	if err == nil || tlsConfig != nil {
		t.Fatal("Should not have been able to configure client TLS", err)
	}
}

// An error is returned if either the client cert or the key are provided
// but invalid or blank.
func TestConfigClientTLSClientCertOrKeyInvalid(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	key, cert := generateCertAndKey(t, tempDir)

	tempFile, err := ioutil.TempFile("", "cert-test")
	if err != nil {
		t.Fatal("Unable to create temporary empty file")
	}
	defer os.Remove(tempFile.Name())
	tempFile.Close()

	for i := 0; i < 2; i++ {
		for _, invalid := range []string{"not-a-file", "", tempFile.Name()} {
			files := []string{cert, key}
			files[i] = invalid

			tlsConfig, err := Client(Options{CertFile: files[0], KeyFile: files[1]})
			if err == nil || tlsConfig != nil {
				t.Fatal("Should not have been able to configure client TLS", err)
			}
		}
	}
}

// The certificate is set if the client cert and client key are provided and
// valid.
func TestConfigClientTLSValidClientCertAndKey(t *testing.T) {
	tempDir := makeTempDir(t)
	defer os.RemoveAll(tempDir)
	key, cert := generateCertAndKey(t, tempDir)

	keypair, err := tls.LoadX509KeyPair(cert, key)
	if err != nil {
		t.Fatal("Unable to load the generated cert and key")
	}

	tlsConfig, err := Client(Options{CertFile: cert, KeyFile: key})

	if err != nil || tlsConfig == nil {
		t.Fatal("Unable to configure client TLS", err)
	}

	if len(tlsConfig.Certificates) != 1 {
		t.Fatal("Unexpected client certificates")
	}
	if len(tlsConfig.Certificates[0].Certificate) != len(keypair.Certificate) {
		t.Fatal("Unexpected client certificates")
	}
	for i, cert := range tlsConfig.Certificates[0].Certificate {
		if !bytes.Equal(cert, keypair.Certificate[i]) {
			t.Fatal("Unexpected client certificates")
		}
	}

	if tlsConfig.RootCAs != nil {
		t.Fatal("Root CAs should not have been set", err)
	}
}
