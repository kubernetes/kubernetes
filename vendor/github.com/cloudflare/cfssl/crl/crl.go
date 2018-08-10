// Package crl exposes Certificate Revocation List generation functionality
package crl

import (
	"crypto"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"math/big"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
)

// NewCRLFromFile takes in a list of serial numbers, one per line, as well as the issuing certificate
// of the CRL, and the private key. This function is then used to parse the list and generate a CRL
func NewCRLFromFile(serialList, issuerFile, keyFile []byte, expiryTime string) ([]byte, error) {

	var revokedCerts []pkix.RevokedCertificate
	var oneWeek = time.Duration(604800) * time.Second

	expiryInt, err := strconv.ParseInt(expiryTime, 0, 32)
	if err != nil {
		return nil, err
	}
	newDurationFromInt := time.Duration(expiryInt) * time.Second
	newExpiryTime := time.Now().Add(newDurationFromInt)
	if expiryInt == 0 {
		newExpiryTime = time.Now().Add(oneWeek)
	}

	// Parse the PEM encoded certificate
	issuerCert, err := helpers.ParseCertificatePEM(issuerFile)
	if err != nil {
		return nil, err
	}

	// Split input file by new lines
	individualCerts := strings.Split(string(serialList), "\n")

	// For every new line, create a new revokedCertificate and add it to slice
	for _, value := range individualCerts {
		if len(strings.TrimSpace(value)) == 0 {
			continue
		}

		tempBigInt := new(big.Int)
		tempBigInt.SetString(value, 10)
		tempCert := pkix.RevokedCertificate{
			SerialNumber:   tempBigInt,
			RevocationTime: time.Now(),
		}
		revokedCerts = append(revokedCerts, tempCert)
	}

	strPassword := os.Getenv("CFSSL_CA_PK_PASSWORD")
	password := []byte(strPassword)
	if strPassword == "" {
		password = nil
	}

	// Parse the key given
	key, err := helpers.ParsePrivateKeyPEMWithPassword(keyFile, password)
	if err != nil {
		log.Debug("Malformed private key %v", err)
		return nil, err
	}

	return CreateGenericCRL(revokedCerts, key, issuerCert, newExpiryTime)
}

// NewCRLFromDB takes in a list of CertificateRecords, as well as the issuing certificate
// of the CRL, and the private key. This function is then used to parse the records and generate a CRL
func NewCRLFromDB(certs []certdb.CertificateRecord, issuerCert *x509.Certificate, key crypto.Signer, expiryTime time.Duration) ([]byte, error) {
	var revokedCerts []pkix.RevokedCertificate

	newExpiryTime := time.Now().Add(expiryTime)

	// For every record, create a new revokedCertificate and add it to slice
	for _, certRecord := range certs {
		serialInt := new(big.Int)
		serialInt.SetString(certRecord.Serial, 10)
		tempCert := pkix.RevokedCertificate{
			SerialNumber:   serialInt,
			RevocationTime: certRecord.RevokedAt,
		}
		revokedCerts = append(revokedCerts, tempCert)
	}

	return CreateGenericCRL(revokedCerts, key, issuerCert, newExpiryTime)
}

// CreateGenericCRL is a helper function that takes in all of the information above, and then calls the createCRL
// function. This outputs the bytes of the created CRL.
func CreateGenericCRL(certList []pkix.RevokedCertificate, key crypto.Signer, issuingCert *x509.Certificate, expiryTime time.Time) ([]byte, error) {
	crlBytes, err := issuingCert.CreateCRL(rand.Reader, key, certList, time.Now(), expiryTime)
	if err != nil {
		log.Debug("error creating CRL: %s", err)
	}

	return crlBytes, err

}
