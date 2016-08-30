package tlsutil

import (
	"crypto/tls"
	"crypto/x509"
	"io"
	"io/ioutil"
	"net"
	"testing"

	"github.com/hashicorp/yamux"
)

func TestConfig_AppendCA_None(t *testing.T) {
	conf := &Config{}
	pool := x509.NewCertPool()
	err := conf.AppendCA(pool)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(pool.Subjects()) != 0 {
		t.Fatalf("bad: %v", pool.Subjects())
	}
}

func TestConfig_CACertificate_Valid(t *testing.T) {
	conf := &Config{
		CAFile: "../test/ca/root.cer",
	}
	pool := x509.NewCertPool()
	err := conf.AppendCA(pool)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(pool.Subjects()) == 0 {
		t.Fatalf("expected cert")
	}
}

func TestConfig_KeyPair_None(t *testing.T) {
	conf := &Config{}
	cert, err := conf.KeyPair()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if cert != nil {
		t.Fatalf("bad: %v", cert)
	}
}

func TestConfig_KeyPair_Valid(t *testing.T) {
	conf := &Config{
		CertFile: "../test/key/ourdomain.cer",
		KeyFile:  "../test/key/ourdomain.key",
	}
	cert, err := conf.KeyPair()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if cert == nil {
		t.Fatalf("expected cert")
	}
}

func TestConfig_OutgoingTLS_MissingCA(t *testing.T) {
	conf := &Config{
		VerifyOutgoing: true,
	}
	tls, err := conf.OutgoingTLSConfig()
	if err == nil {
		t.Fatalf("expected err")
	}
	if tls != nil {
		t.Fatalf("bad: %v", tls)
	}
}

func TestConfig_OutgoingTLS_OnlyCA(t *testing.T) {
	conf := &Config{
		CAFile: "../test/ca/root.cer",
	}
	tls, err := conf.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tls != nil {
		t.Fatalf("expected no config")
	}
}

func TestConfig_OutgoingTLS_VerifyOutgoing(t *testing.T) {
	conf := &Config{
		VerifyOutgoing: true,
		CAFile:         "../test/ca/root.cer",
	}
	tls, err := conf.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tls == nil {
		t.Fatalf("expected config")
	}
	if len(tls.RootCAs.Subjects()) != 1 {
		t.Fatalf("expect root cert")
	}
	if tls.ServerName != "" {
		t.Fatalf("expect no server name verification")
	}
	if !tls.InsecureSkipVerify {
		t.Fatalf("should skip built-in verification")
	}
}

func TestConfig_OutgoingTLS_ServerName(t *testing.T) {
	conf := &Config{
		VerifyOutgoing: true,
		CAFile:         "../test/ca/root.cer",
		ServerName:     "consul.example.com",
	}
	tls, err := conf.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tls == nil {
		t.Fatalf("expected config")
	}
	if len(tls.RootCAs.Subjects()) != 1 {
		t.Fatalf("expect root cert")
	}
	if tls.ServerName != "consul.example.com" {
		t.Fatalf("expect server name")
	}
	if tls.InsecureSkipVerify {
		t.Fatalf("should not skip built-in verification")
	}
}

func TestConfig_OutgoingTLS_VerifyHostname(t *testing.T) {
	conf := &Config{
		VerifyServerHostname: true,
		CAFile:               "../test/ca/root.cer",
	}
	tls, err := conf.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tls == nil {
		t.Fatalf("expected config")
	}
	if len(tls.RootCAs.Subjects()) != 1 {
		t.Fatalf("expect root cert")
	}
	if tls.ServerName != "VerifyServerHostname" {
		t.Fatalf("expect server name")
	}
	if tls.InsecureSkipVerify {
		t.Fatalf("should not skip built-in verification")
	}
}

func TestConfig_OutgoingTLS_WithKeyPair(t *testing.T) {
	conf := &Config{
		VerifyOutgoing: true,
		CAFile:         "../test/ca/root.cer",
		CertFile:       "../test/key/ourdomain.cer",
		KeyFile:        "../test/key/ourdomain.key",
	}
	tls, err := conf.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tls == nil {
		t.Fatalf("expected config")
	}
	if len(tls.RootCAs.Subjects()) != 1 {
		t.Fatalf("expect root cert")
	}
	if !tls.InsecureSkipVerify {
		t.Fatalf("should skip verification")
	}
	if len(tls.Certificates) != 1 {
		t.Fatalf("expected client cert")
	}
}

func startTLSServer(config *Config) (net.Conn, chan error) {
	errc := make(chan error, 1)

	tlsConfigServer, err := config.IncomingTLSConfig()
	if err != nil {
		errc <- err
		return nil, errc
	}

	client, server := net.Pipe()

	// Use yamux to buffer the reads, otherwise it's easy to deadlock
	muxConf := yamux.DefaultConfig()
	serverSession, _ := yamux.Server(server, muxConf)
	clientSession, _ := yamux.Client(client, muxConf)
	clientConn, _ := clientSession.Open()
	serverConn, _ := serverSession.Accept()

	go func() {
		tlsServer := tls.Server(serverConn, tlsConfigServer)
		if err := tlsServer.Handshake(); err != nil {
			errc <- err
		}
		close(errc)
		// Because net.Pipe() is unbuffered, if both sides
		// Close() simultaneously, we will deadlock as they
		// both send an alert and then block. So we make the
		// server read any data from the client until error or
		// EOF, which will allow the client to Close(), and
		// *then* we Close() the server.
		io.Copy(ioutil.Discard, tlsServer)
		tlsServer.Close()
	}()
	return clientConn, errc
}

func TestConfig_outgoingWrapper_OK(t *testing.T) {
	config := &Config{
		CAFile:               "../test/hostname/CertAuth.crt",
		CertFile:             "../test/hostname/Alice.crt",
		KeyFile:              "../test/hostname/Alice.key",
		VerifyServerHostname: true,
		Domain:               "consul",
	}

	client, errc := startTLSServer(config)
	if client == nil {
		t.Fatalf("startTLSServer err: %v", <-errc)
	}

	wrap, err := config.OutgoingTLSWrapper()
	if err != nil {
		t.Fatalf("OutgoingTLSWrapper err: %v", err)
	}

	tlsClient, err := wrap("dc1", client)
	if err != nil {
		t.Fatalf("wrapTLS err: %v", err)
	}
	defer tlsClient.Close()
	if err := tlsClient.(*tls.Conn).Handshake(); err != nil {
		t.Fatalf("write err: %v", err)
	}

	err = <-errc
	if err != nil {
		t.Fatalf("server: %v", err)
	}
}

func TestConfig_outgoingWrapper_BadDC(t *testing.T) {
	config := &Config{
		CAFile:               "../test/hostname/CertAuth.crt",
		CertFile:             "../test/hostname/Alice.crt",
		KeyFile:              "../test/hostname/Alice.key",
		VerifyServerHostname: true,
		Domain:               "consul",
	}

	client, errc := startTLSServer(config)
	if client == nil {
		t.Fatalf("startTLSServer err: %v", <-errc)
	}

	wrap, err := config.OutgoingTLSWrapper()
	if err != nil {
		t.Fatalf("OutgoingTLSWrapper err: %v", err)
	}

	tlsClient, err := wrap("dc2", client)
	if err != nil {
		t.Fatalf("wrapTLS err: %v", err)
	}
	defer tlsClient.Close()
	err = tlsClient.(*tls.Conn).Handshake()

	if _, ok := err.(x509.HostnameError); !ok {
		t.Fatalf("should get hostname err: %v", err)
	}

	<-errc
}

func TestConfig_outgoingWrapper_BadCert(t *testing.T) {
	config := &Config{
		CAFile:               "../test/ca/root.cer",
		CertFile:             "../test/key/ourdomain.cer",
		KeyFile:              "../test/key/ourdomain.key",
		VerifyServerHostname: true,
		Domain:               "consul",
	}

	client, errc := startTLSServer(config)
	if client == nil {
		t.Fatalf("startTLSServer err: %v", <-errc)
	}

	wrap, err := config.OutgoingTLSWrapper()
	if err != nil {
		t.Fatalf("OutgoingTLSWrapper err: %v", err)
	}

	tlsClient, err := wrap("dc1", client)
	if err != nil {
		t.Fatalf("wrapTLS err: %v", err)
	}
	defer tlsClient.Close()
	err = tlsClient.(*tls.Conn).Handshake()

	if _, ok := err.(x509.HostnameError); !ok {
		t.Fatalf("should get hostname err: %v", err)
	}

	<-errc
}

func TestConfig_wrapTLS_OK(t *testing.T) {
	config := &Config{
		CAFile:         "../test/ca/root.cer",
		CertFile:       "../test/key/ourdomain.cer",
		KeyFile:        "../test/key/ourdomain.key",
		VerifyOutgoing: true,
	}

	client, errc := startTLSServer(config)
	if client == nil {
		t.Fatalf("startTLSServer err: %v", <-errc)
	}

	clientConfig, err := config.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("OutgoingTLSConfig err: %v", err)
	}

	tlsClient, err := WrapTLSClient(client, clientConfig)
	if err != nil {
		t.Fatalf("wrapTLS err: %v", err)
	} else {
		tlsClient.Close()
	}
	err = <-errc
	if err != nil {
		t.Fatalf("server: %v", err)
	}
}

func TestConfig_wrapTLS_BadCert(t *testing.T) {
	serverConfig := &Config{
		CertFile: "../test/key/ssl-cert-snakeoil.pem",
		KeyFile:  "../test/key/ssl-cert-snakeoil.key",
	}

	client, errc := startTLSServer(serverConfig)
	if client == nil {
		t.Fatalf("startTLSServer err: %v", <-errc)
	}

	clientConfig := &Config{
		CAFile:         "../test/ca/root.cer",
		VerifyOutgoing: true,
	}

	clientTLSConfig, err := clientConfig.OutgoingTLSConfig()
	if err != nil {
		t.Fatalf("OutgoingTLSConfig err: %v", err)
	}

	tlsClient, err := WrapTLSClient(client, clientTLSConfig)
	if err == nil {
		t.Fatalf("wrapTLS no err")
	}
	if tlsClient != nil {
		t.Fatalf("returned a client")
	}

	err = <-errc
	if err != nil {
		t.Fatalf("server: %v", err)
	}
}

func TestConfig_IncomingTLS(t *testing.T) {
	conf := &Config{
		VerifyIncoming: true,
		CAFile:         "../test/ca/root.cer",
		CertFile:       "../test/key/ourdomain.cer",
		KeyFile:        "../test/key/ourdomain.key",
	}
	tlsC, err := conf.IncomingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tlsC == nil {
		t.Fatalf("expected config")
	}
	if len(tlsC.ClientCAs.Subjects()) != 1 {
		t.Fatalf("expect client cert")
	}
	if tlsC.ClientAuth != tls.RequireAndVerifyClientCert {
		t.Fatalf("should not skip verification")
	}
	if len(tlsC.Certificates) != 1 {
		t.Fatalf("expected client cert")
	}
}

func TestConfig_IncomingTLS_MissingCA(t *testing.T) {
	conf := &Config{
		VerifyIncoming: true,
		CertFile:       "../test/key/ourdomain.cer",
		KeyFile:        "../test/key/ourdomain.key",
	}
	_, err := conf.IncomingTLSConfig()
	if err == nil {
		t.Fatalf("expected err")
	}
}

func TestConfig_IncomingTLS_MissingKey(t *testing.T) {
	conf := &Config{
		VerifyIncoming: true,
		CAFile:         "../test/ca/root.cer",
	}
	_, err := conf.IncomingTLSConfig()
	if err == nil {
		t.Fatalf("expected err")
	}
}

func TestConfig_IncomingTLS_NoVerify(t *testing.T) {
	conf := &Config{}
	tlsC, err := conf.IncomingTLSConfig()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if tlsC == nil {
		t.Fatalf("expected config")
	}
	if len(tlsC.ClientCAs.Subjects()) != 0 {
		t.Fatalf("do not expect client cert")
	}
	if tlsC.ClientAuth != tls.NoClientCert {
		t.Fatalf("should skip verification")
	}
	if len(tlsC.Certificates) != 0 {
		t.Fatalf("unexpected client cert")
	}
}
