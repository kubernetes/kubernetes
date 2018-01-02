package main

import (
	"crypto/tls"
	"fmt"
	"html"
	"log"
	"net"
	"net/http"

	"github.com/docker/libtrust"
)

var (
	serverAddress             = "localhost:8888"
	privateKeyFilename        = "server_data/private_key.pem"
	authorizedClientsFilename = "server_data/trusted_clients.pem"
)

func requestHandler(w http.ResponseWriter, r *http.Request) {
	clientCert := r.TLS.PeerCertificates[0]
	keyID := clientCert.Subject.CommonName
	log.Printf("Request from keyID: %s\n", keyID)
	fmt.Fprintf(w, "Hello, client! I'm a server! And you are %T: %s.\n", clientCert.PublicKey, html.EscapeString(keyID))
}

func main() {
	// Load server key.
	serverKey, err := libtrust.LoadKeyFile(privateKeyFilename)
	if err != nil {
		log.Fatal(err)
	}

	// Generate server certificate.
	selfSignedServerCert, err := libtrust.GenerateSelfSignedServerCert(
		serverKey, []string{"localhost"}, []net.IP{net.ParseIP("127.0.0.1")},
	)
	if err != nil {
		log.Fatal(err)
	}

	// Load authorized client keys.
	authorizedClients, err := libtrust.LoadKeySetFile(authorizedClientsFilename)
	if err != nil {
		log.Fatal(err)
	}

	// Create CA pool using trusted client keys.
	caPool, err := libtrust.GenerateCACertPool(serverKey, authorizedClients)
	if err != nil {
		log.Fatal(err)
	}

	// Create TLS config, requiring client certificates.
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{
			tls.Certificate{
				Certificate: [][]byte{selfSignedServerCert.Raw},
				PrivateKey:  serverKey.CryptoPrivateKey(),
				Leaf:        selfSignedServerCert,
			},
		},
		ClientAuth: tls.RequireAndVerifyClientCert,
		ClientCAs:  caPool,
	}

	// Create HTTP server with simple request handler.
	server := &http.Server{
		Addr:    serverAddress,
		Handler: http.HandlerFunc(requestHandler),
	}

	// Listen and server HTTPS using the libtrust TLS config.
	listener, err := net.Listen("tcp", server.Addr)
	if err != nil {
		log.Fatal(err)
	}
	tlsListener := tls.NewListener(listener, tlsConfig)
	server.Serve(tlsListener)
}
