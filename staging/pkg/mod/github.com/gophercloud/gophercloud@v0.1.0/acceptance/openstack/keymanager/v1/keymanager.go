package v1

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"math/big"
	"strings"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/containers"
	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/orders"
	"github.com/gophercloud/gophercloud/openstack/keymanager/v1/secrets"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateAsymmetric Order will create a random asymmetric order.
// An error will be returned if the order could not be created.
func CreateAsymmetricOrder(t *testing.T, client *gophercloud.ServiceClient) (*orders.Order, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create order %s", name)

	expiration := time.Date(2049, 1, 1, 1, 1, 1, 0, time.UTC)
	createOpts := orders.CreateOpts{
		Type: orders.AsymmetricOrder,
		Meta: orders.MetaOpts{
			Name:       name,
			Algorithm:  "rsa",
			BitLength:  2048,
			Mode:       "cbc",
			Expiration: &expiration,
		},
	}

	order, err := orders.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	orderID, err := ParseID(order.OrderRef)
	if err != nil {
		return nil, err
	}

	err = WaitForOrder(client, orderID)
	th.AssertNoErr(t, err)

	order, err = orders.Get(client, orderID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, order)
	tools.PrintResource(t, order.Meta.Expiration)

	th.AssertEquals(t, order.Meta.Name, name)
	th.AssertEquals(t, order.Type, "asymmetric")

	return order, nil
}

// CreateCertificateContainer will create a random certificate container.
// An error will be returned if the container could not be created.
func CreateCertificateContainer(t *testing.T, client *gophercloud.ServiceClient, passphrase, private, certificate *secrets.Secret) (*containers.Container, error) {
	containerName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create container %s", containerName)

	createOpts := containers.CreateOpts{
		Type: containers.CertificateContainer,
		Name: containerName,
		SecretRefs: []containers.SecretRef{
			{
				Name:      "certificate",
				SecretRef: certificate.SecretRef,
			},
			{
				Name:      "private_key",
				SecretRef: private.SecretRef,
			},
			{
				Name:      "private_key_passphrase",
				SecretRef: passphrase.SecretRef,
			},
		},
	}

	container, err := containers.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created container: %s", container.ContainerRef)

	containerID, err := ParseID(container.ContainerRef)
	if err != nil {
		return nil, err
	}

	container, err = containers.Get(client, containerID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, container)

	th.AssertEquals(t, container.Name, containerName)
	th.AssertEquals(t, container.Type, "certificate")

	return container, nil
}

// CreateKeyOrder will create a random key order.
// An error will be returned if the order could not be created.
func CreateKeyOrder(t *testing.T, client *gophercloud.ServiceClient) (*orders.Order, error) {
	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create order %s", name)

	expiration := time.Date(2049, 1, 1, 1, 1, 1, 0, time.UTC)
	createOpts := orders.CreateOpts{
		Type: orders.KeyOrder,
		Meta: orders.MetaOpts{
			Name:       name,
			Algorithm:  "aes",
			BitLength:  256,
			Mode:       "cbc",
			Expiration: &expiration,
		},
	}

	order, err := orders.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	orderID, err := ParseID(order.OrderRef)
	if err != nil {
		return nil, err
	}

	order, err = orders.Get(client, orderID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, order)
	tools.PrintResource(t, order.Meta.Expiration)

	th.AssertEquals(t, order.Meta.Name, name)
	th.AssertEquals(t, order.Type, "key")

	return order, nil
}

// CreateRSAContainer will create a random RSA container.
// An error will be returned if the container could not be created.
func CreateRSAContainer(t *testing.T, client *gophercloud.ServiceClient, passphrase, private, public *secrets.Secret) (*containers.Container, error) {
	containerName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create container %s", containerName)

	createOpts := containers.CreateOpts{
		Type: containers.RSAContainer,
		Name: containerName,
		SecretRefs: []containers.SecretRef{
			{
				Name:      "public_key",
				SecretRef: public.SecretRef,
			},
			{
				Name:      "private_key",
				SecretRef: private.SecretRef,
			},
			{
				Name:      "private_key_passphrase",
				SecretRef: passphrase.SecretRef,
			},
		},
	}

	container, err := containers.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created container: %s", container.ContainerRef)

	containerID, err := ParseID(container.ContainerRef)
	if err != nil {
		return nil, err
	}

	container, err = containers.Get(client, containerID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, container)

	th.AssertEquals(t, container.Name, containerName)
	th.AssertEquals(t, container.Type, "rsa")

	return container, nil
}

// CreateCertificateSecret will create a random certificate secret. An error
// will be returned if the secret could not be created.
func CreateCertificateSecret(t *testing.T, client *gophercloud.ServiceClient, cert []byte) (*secrets.Secret, error) {
	b64Cert := base64.StdEncoding.EncodeToString(cert)

	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create public key %s", name)

	createOpts := secrets.CreateOpts{
		Name:                   name,
		SecretType:             secrets.CertificateSecret,
		Payload:                b64Cert,
		PayloadContentType:     "application/octet-stream",
		PayloadContentEncoding: "base64",
		Algorithm:              "rsa",
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, name)
	th.AssertEquals(t, secret.Algorithm, "rsa")

	return secret, nil
}

// CreateEmptySecret will create a random secret with no payload. An error will
// be returned if the secret could not be created.
func CreateEmptySecret(t *testing.T, client *gophercloud.ServiceClient) (*secrets.Secret, error) {
	secretName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create secret %s", secretName)

	createOpts := secrets.CreateOpts{
		Algorithm:  "aes",
		BitLength:  256,
		Mode:       "cbc",
		Name:       secretName,
		SecretType: secrets.OpaqueSecret,
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, secretName)
	th.AssertEquals(t, secret.Algorithm, "aes")

	return secret, nil
}

// CreateGenericContainer will create a random generic container with a
// specified secret. An error will be returned if the container could not
// be created.
func CreateGenericContainer(t *testing.T, client *gophercloud.ServiceClient, secret *secrets.Secret) (*containers.Container, error) {
	containerName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create container %s", containerName)

	createOpts := containers.CreateOpts{
		Type: containers.GenericContainer,
		Name: containerName,
		SecretRefs: []containers.SecretRef{
			{
				Name:      secret.Name,
				SecretRef: secret.SecretRef,
			},
		},
	}

	container, err := containers.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created container: %s", container.ContainerRef)

	containerID, err := ParseID(container.ContainerRef)
	if err != nil {
		return nil, err
	}

	container, err = containers.Get(client, containerID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, container)

	th.AssertEquals(t, container.Name, containerName)
	th.AssertEquals(t, container.Type, "generic")

	return container, nil
}

// CreatePassphraseSecret will create a random passphrase secret.
// An error will be returned if the secret could not be created.
func CreatePassphraseSecret(t *testing.T, client *gophercloud.ServiceClient, passphrase string) (*secrets.Secret, error) {
	secretName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create secret %s", secretName)

	createOpts := secrets.CreateOpts{
		Algorithm:          "aes",
		BitLength:          256,
		Mode:               "cbc",
		Name:               secretName,
		Payload:            passphrase,
		PayloadContentType: "text/plain",
		SecretType:         secrets.PassphraseSecret,
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, secretName)
	th.AssertEquals(t, secret.Algorithm, "aes")

	return secret, nil
}

// CreatePublicSecret will create a random public secret. An error
// will be returned if the secret could not be created.
func CreatePublicSecret(t *testing.T, client *gophercloud.ServiceClient, pub []byte) (*secrets.Secret, error) {
	b64Cert := base64.StdEncoding.EncodeToString(pub)

	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create public key %s", name)

	createOpts := secrets.CreateOpts{
		Name:                   name,
		SecretType:             secrets.PublicSecret,
		Payload:                b64Cert,
		PayloadContentType:     "application/octet-stream",
		PayloadContentEncoding: "base64",
		Algorithm:              "rsa",
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, name)
	th.AssertEquals(t, secret.Algorithm, "rsa")

	return secret, nil
}

// CreatePrivateSecret will create a random private secret. An error
// will be returned if the secret could not be created.
func CreatePrivateSecret(t *testing.T, client *gophercloud.ServiceClient, priv []byte) (*secrets.Secret, error) {
	b64Cert := base64.StdEncoding.EncodeToString(priv)

	name := tools.RandomString("TESTACC-", 8)
	t.Logf("Attempting to create public key %s", name)

	createOpts := secrets.CreateOpts{
		Name:                   name,
		SecretType:             secrets.PrivateSecret,
		Payload:                b64Cert,
		PayloadContentType:     "application/octet-stream",
		PayloadContentEncoding: "base64",
		Algorithm:              "rsa",
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, name)
	th.AssertEquals(t, secret.Algorithm, "rsa")

	return secret, nil
}

// CreateSecretWithPayload will create a random secret with a given payload.
// An error will be returned if the secret could not be created.
func CreateSecretWithPayload(t *testing.T, client *gophercloud.ServiceClient, payload string) (*secrets.Secret, error) {
	secretName := tools.RandomString("TESTACC-", 8)

	t.Logf("Attempting to create secret %s", secretName)

	expiration := time.Date(2049, 1, 1, 1, 1, 1, 0, time.UTC)
	createOpts := secrets.CreateOpts{
		Algorithm:          "aes",
		BitLength:          256,
		Mode:               "cbc",
		Name:               secretName,
		Payload:            payload,
		PayloadContentType: "text/plain",
		SecretType:         secrets.OpaqueSecret,
		Expiration:         &expiration,
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, secretName)
	th.AssertEquals(t, secret.Algorithm, "aes")
	th.AssertEquals(t, secret.Expiration, expiration)

	return secret, nil
}

// CreateSymmetricSecret will create a random symmetric secret. An error
// will be returned if the secret could not be created.
func CreateSymmetricSecret(t *testing.T, client *gophercloud.ServiceClient) (*secrets.Secret, error) {
	name := tools.RandomString("TESTACC-", 8)
	key := tools.RandomString("", 256)
	b64Key := base64.StdEncoding.EncodeToString([]byte(key))

	t.Logf("Attempting to create symmetric key %s", name)

	createOpts := secrets.CreateOpts{
		Name:                   name,
		SecretType:             secrets.SymmetricSecret,
		Payload:                b64Key,
		PayloadContentType:     "application/octet-stream",
		PayloadContentEncoding: "base64",
		Algorithm:              "aes",
		BitLength:              256,
		Mode:                   "cbc",
	}

	secret, err := secrets.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created secret: %s", secret.SecretRef)

	secretID, err := ParseID(secret.SecretRef)
	if err != nil {
		return nil, err
	}

	secret, err = secrets.Get(client, secretID).Extract()
	if err != nil {
		return nil, err
	}

	tools.PrintResource(t, secret)

	th.AssertEquals(t, secret.Name, name)
	th.AssertEquals(t, secret.Algorithm, "aes")

	return secret, nil
}

// DeleteContainer will delete a container. A fatal error will occur if the
// container could not be deleted. This works best when used as a deferred
// function.
func DeleteContainer(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete container %s", id)

	err := containers.Delete(client, id).ExtractErr()
	if err != nil {
		t.Fatalf("Could not delete container: %s", err)
	}

	t.Logf("Successfully deleted container %s", id)
}

// DeleteOrder will delete an order. A fatal error will occur if the
// order could not be deleted. This works best when used as a deferred
// function.
func DeleteOrder(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete order %s", id)

	err := orders.Delete(client, id).ExtractErr()
	if err != nil {
		t.Fatalf("Could not delete order: %s", err)
	}

	t.Logf("Successfully deleted order %s", id)
}

// DeleteSecret will delete a secret. A fatal error will occur if the secret
// could not be deleted. This works best when used as a deferred function.
func DeleteSecret(t *testing.T, client *gophercloud.ServiceClient, id string) {
	t.Logf("Attempting to delete secret %s", id)

	err := secrets.Delete(client, id).ExtractErr()
	if err != nil {
		t.Fatalf("Could not delete secret: %s", err)
	}

	t.Logf("Successfully deleted secret %s", id)
}

func ParseID(ref string) (string, error) {
	parts := strings.Split(ref, "/")
	if len(parts) < 2 {
		return "", fmt.Errorf("Could not parse %s", ref)
	}

	return parts[len(parts)-1], nil
}

// CreateCertificate will create a random certificate. A fatal error will
// be returned if creation failed.
// https://golang.org/src/crypto/tls/generate_cert.go
func CreateCertificate(t *testing.T, passphrase string) ([]byte, []byte, error) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, err
	}

	block := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	}

	if passphrase != "" {
		block, err = x509.EncryptPEMBlock(rand.Reader, block.Type, block.Bytes, []byte(passphrase), x509.PEMCipherAES256)
		if err != nil {
			return nil, nil, err
		}
	}

	keyPem := pem.EncodeToMemory(block)

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, nil, err
	}

	tpl := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"Some Org"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().AddDate(5, 0, 0),
		BasicConstraintsValid: true,
	}

	cert, err := x509.CreateCertificate(rand.Reader, &tpl, &tpl, &key.PublicKey, key)
	if err != nil {
		return nil, nil, err
	}

	certPem := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cert,
	})

	return keyPem, certPem, nil
}

// CreateRSAKeyPair will create a random RSA key pair. An error will be
// returned if the pair could not be created.
func CreateRSAKeyPair(t *testing.T, passphrase string) ([]byte, []byte, error) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, err
	}

	block := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	}

	if passphrase != "" {
		block, err = x509.EncryptPEMBlock(rand.Reader, block.Type, block.Bytes, []byte(passphrase), x509.PEMCipherAES256)
		if err != nil {
			return nil, nil, err
		}
	}

	keyPem := pem.EncodeToMemory(block)

	asn1Bytes, err := asn1.Marshal(key.PublicKey)
	if err != nil {
		return nil, nil, err
	}

	block = &pem.Block{
		Type:  "RSA PUBLIC KEY",
		Bytes: asn1Bytes,
	}

	pubPem := pem.EncodeToMemory(block)

	return keyPem, pubPem, nil
}

func WaitForOrder(client *gophercloud.ServiceClient, orderID string) error {
	return tools.WaitFor(func() (bool, error) {
		order, err := orders.Get(client, orderID).Extract()
		if err != nil {
			return false, err
		}

		if order.SecretRef != "" {
			return true, nil
		}

		if order.ContainerRef != "" {
			return true, nil
		}

		if order.Status == "ERROR" {
			return false, fmt.Errorf("Order %s in ERROR state", orderID)
		}

		return false, nil
	})
}
