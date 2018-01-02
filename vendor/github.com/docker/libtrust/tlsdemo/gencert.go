package main

import (
	"encoding/pem"
	"fmt"
	"log"
	"net"

	"github.com/docker/libtrust"
)

var (
	serverAddress            = "localhost:8888"
	clientPrivateKeyFilename = "client_data/private_key.pem"
	trustedHostsFilename     = "client_data/trusted_hosts.pem"
)

func main() {
	key, err := libtrust.LoadKeyFile(clientPrivateKeyFilename)
	if err != nil {
		log.Fatal(err)
	}

	keyPEMBlock, err := key.PEMBlock()
	if err != nil {
		log.Fatal(err)
	}

	encodedPrivKey := pem.EncodeToMemory(keyPEMBlock)
	fmt.Printf("Client Key:\n\n%s\n", string(encodedPrivKey))

	cert, err := libtrust.GenerateSelfSignedClientCert(key)
	if err != nil {
		log.Fatal(err)
	}

	encodedCert := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})
	fmt.Printf("Client Cert:\n\n%s\n", string(encodedCert))

	trustedServerKeys, err := libtrust.LoadKeySetFile(trustedHostsFilename)
	if err != nil {
		log.Fatal(err)
	}

	hostname, _, err := net.SplitHostPort(serverAddress)
	if err != nil {
		log.Fatal(err)
	}

	trustedServerKeys, err = libtrust.FilterByHosts(trustedServerKeys, hostname, false)
	if err != nil {
		log.Fatal(err)
	}

	caCert, err := libtrust.GenerateCACert(key, trustedServerKeys[0])
	if err != nil {
		log.Fatal(err)
	}

	encodedCert = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCert.Raw})
	fmt.Printf("CA Cert:\n\n%s\n", string(encodedCert))
}
