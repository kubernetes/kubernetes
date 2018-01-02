package main

import (
	"log"

	"github.com/docker/libtrust"
)

func main() {
	// Generate client key.
	clientKey, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		log.Fatal(err)
	}

	// Add a comment for the client key.
	clientKey.AddExtendedField("comment", "TLS Demo Client")

	// Save the client key, public and private versions.
	err = libtrust.SaveKey("client_data/private_key.pem", clientKey)
	if err != nil {
		log.Fatal(err)
	}

	err = libtrust.SavePublicKey("client_data/public_key.pem", clientKey.PublicKey())
	if err != nil {
		log.Fatal(err)
	}

	// Generate server key.
	serverKey, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		log.Fatal(err)
	}

	// Set the list of addresses to use for the server.
	serverKey.AddExtendedField("hosts", []string{"localhost", "docker.example.com"})

	// Save the server key, public and private versions.
	err = libtrust.SaveKey("server_data/private_key.pem", serverKey)
	if err != nil {
		log.Fatal(err)
	}

	err = libtrust.SavePublicKey("server_data/public_key.pem", serverKey.PublicKey())
	if err != nil {
		log.Fatal(err)
	}

	// Generate Authorized Keys file for server.
	err = libtrust.AddKeySetFile("server_data/trusted_clients.pem", clientKey.PublicKey())
	if err != nil {
		log.Fatal(err)
	}

	// Generate Known Host Keys file for client.
	err = libtrust.AddKeySetFile("client_data/trusted_hosts.pem", serverKey.PublicKey())
	if err != nil {
		log.Fatal(err)
	}
}
