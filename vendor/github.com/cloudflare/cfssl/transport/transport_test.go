package transport

import (
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/api/client"
	"github.com/cloudflare/cfssl/csr"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/log"
	"github.com/cloudflare/cfssl/transport/core"
)

var (
	testRemote   = envOrDefault("CFSSL_REMOTE", "127.0.0.1:8888")
	testLabel    = envOrDefault("CFSSL_LABEL", "")
	testProfile  = envOrDefault("CFSSL_PROFILE", "transport-test")
	disableTests bool
)

func cfsslIsAvailable() bool {
	defaultRemote := client.NewServer(testRemote)

	infoReq := info.Req{
		Profile: testProfile,
		Label:   testLabel,
	}

	out, err := json.Marshal(infoReq)
	if err != nil {
		return false
	}

	_, err = defaultRemote.Info(out)
	if err != nil {
		log.Debug("CFSSL remote is unavailable, skipping tests")
		return false
	}

	return true
}

func removeIfPresent(path string) error {
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		return os.Remove(path)
	}
	return nil
}

func TestMain(m *testing.M) {
	if fi, err := os.Stat("testdata"); os.IsNotExist(err) {
		err = os.Mkdir("testdata", 0755)
		if err != nil {
			log.Fatalf("unable to setup testdata directory: %v", err)
		}
	} else if fi != nil && !fi.Mode().IsDir() {
		log.Fatalf("testdata exists but isn't a directory")
	} else if err != nil {
		log.Fatalf("%v", err)
	}

	var exitCode int
	if cfsslIsAvailable() {
		exitCode = m.Run()
	}

	err := removeIfPresent(testKey)
	if err == nil {
		err = removeIfPresent(testCert)
	}
	if err != nil {
		os.Exit(1)
	}

	err = removeIfPresent(testLKey)
	if err == nil {
		err = removeIfPresent(testLCert)
	}
	if err != nil {
		os.Exit(1)
	}

	os.Exit(exitCode)
}

var (
	tr           *Transport
	testKey      = "testdata/test.key"
	testCert     = "testdata/test.pem"
	testIdentity = &core.Identity{
		Request: &csr.CertificateRequest{
			CN: "localhost test certificate",
		},
		Roots: []*core.Root{
			{
				Type: "system",
			},
			{
				Type: "cfssl",
				Metadata: map[string]string{
					"host":    testRemote,
					"label":   testLabel,
					"profile": testProfile,
				},
			},
		},
		Profiles: map[string]map[string]string{
			"paths": {
				"private_key": testKey,
				"certificate": testCert,
			},
			"cfssl": {
				"label":   testLabel,
				"profile": testProfile,
				"remote":  testRemote,
			},
		},
	}
)

func TestTransportSetup(t *testing.T) {
	var before = 55 * time.Second
	var err error

	tr, err = New(before, testIdentity)
	if err != nil {
		t.Fatalf("failed to set up transport: %v", err)
	}
}

func TestRefreshKeys(t *testing.T) {
	err := tr.RefreshKeys()
	if err != nil {
		t.Fatalf("%v", err)
	}
}

func TestAutoUpdate(t *testing.T) {
	// To force a refresh, make sure that the certificate is
	// updated 5 seconds from now.
	cert := tr.Provider.Certificate()
	if cert == nil {
		t.Fatal("no certificate from provider")
	}

	certUpdates := make(chan time.Time, 0)
	errUpdates := make(chan error, 0)
	oldBefore := tr.Before
	before := cert.NotAfter.Sub(time.Now())
	before -= 5 * time.Second
	tr.Before = before
	defer func() {
		tr.Before = oldBefore
		PollInterval = 30 * time.Second
	}()

	PollInterval = 2 * time.Second

	go tr.AutoUpdate(certUpdates, errUpdates)
	log.Debugf("waiting for certificate update or error from auto updater")
	select {
	case <-certUpdates:
		// Nothing needs to be done
	case err := <-errUpdates:
		t.Fatalf("%v", err)
	case <-time.After(15 * time.Second):
		t.Fatal("timeout waiting for update")
	}
}

var (
	l             *Listener
	testLKey      = "testdata/server.key"
	testLCert     = "testdata/server.pem"
	testLIdentity = &core.Identity{
		Request: &csr.CertificateRequest{
			CN:    "localhost test certificate",
			Hosts: []string{"127.0.0.1"},
		},
		Profiles: map[string]map[string]string{
			"paths": {
				"private_key": testLKey,
				"certificate": testLCert,
			},
			"cfssl": {
				"label":   testLabel,
				"profile": testProfile,
				"remote":  testRemote,
			},
		},
		Roots: []*core.Root{
			{
				Type: "system",
			},
			{
				Type: "cfssl",
				Metadata: map[string]string{
					"host":    testRemote,
					"label":   testLabel,
					"profile": testProfile,
				},
			},
		},
	}
)

func testListen(t *testing.T) {
	log.Debug("listener waiting for connection")
	conn, err := l.Accept()
	if err != nil {
		t.Fatalf("%v", err)
	}

	log.Debugf("client has connected")
	conn.Write([]byte("hello"))

	conn.Close()
}

func TestListener(t *testing.T) {
	var before = 55 * time.Second

	trl, err := New(before, testLIdentity)
	if err != nil {
		t.Fatalf("failed to set up transport: %v", err)
	}

	trl.Identity.Request.CN = "localhost test server"

	err = trl.RefreshKeys()
	if err != nil {
		t.Fatalf("%v", err)
	}

	l, err = Listen("127.0.0.1:8765", trl)
	if err != nil {
		t.Fatalf("%v", err)
	}

	errChan := make(chan error, 0)
	go func() {
		err := <-errChan
		if err != nil {
			t.Fatalf("listener auto update failed: %v", err)
		}
	}()

	cert := trl.Provider.Certificate()
	before = cert.NotAfter.Sub(time.Now())
	before -= 5 * time.Second

	trl.Before = before
	go l.AutoUpdate(nil, errChan)
	go testListen(t)

	<-time.After(1 * time.Second)
	log.Debug("dialer making connection")
	conn, err := Dial("127.0.0.1:8765", tr)
	if err != nil {
		log.Debugf("certificate time: %s-%s / %s",
			trl.Provider.Certificate().NotBefore,
			trl.Provider.Certificate().NotAfter,
			time.Now().UTC())
		log.Debugf("%#v", trl.Provider.Certificate())
		t.Fatalf("%v", err)
	}
	log.Debugf("client connected to server")

	conn.Close()
}
