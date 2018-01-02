// Package crl implements the HTTP handler for the crl commands.
package crl

import (
	"crypto/rand"
	"crypto/x509/pkix"
	"encoding/json"
	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/log"
	"io/ioutil"
	"math/big"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// This type is meant to be unmarshalled from JSON
type jsonCRLRequest struct {
	Certificate  string   `json:"certificate"`
	SerialNumber []string `json:"serialNumber"`
	PrivateKey   string   `json:"issuingKey"`
	ExpiryTime   string   `json:"expireTime"`
}

// Handle responds to requests for crl generation. It creates this crl
// based off of the given certificate, serial numbers, and private key
func crlHandler(w http.ResponseWriter, r *http.Request) error {

	var revokedCerts []pkix.RevokedCertificate
	var oneWeek = time.Duration(604800) * time.Second
	var newExpiryTime = time.Now()

	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	r.Body.Close()

	req := &jsonCRLRequest{}

	err = json.Unmarshal(body, req)
	if err != nil {
		log.Error(err)
	}

	if req.ExpiryTime != "" {
		expiryTime := strings.TrimSpace(req.ExpiryTime)
		expiryInt, err := strconv.ParseInt(expiryTime, 0, 32)
		if err != nil {
			return err
		}

		newExpiryTime = time.Now().Add((time.Duration(expiryInt) * time.Second))
	}

	if req.ExpiryTime == "" {
		newExpiryTime = time.Now().Add(oneWeek)
	}

	if err != nil {
		return err
	}

	cert, err := helpers.ParseCertificatePEM([]byte(req.Certificate))
	if err != nil {
		log.Error("Error from ParseCertificatePEM", err)
		return errors.NewBadRequestString("Malformed certificate")
	}

	for _, value := range req.SerialNumber {
		tempBigInt := new(big.Int)
		tempBigInt.SetString(value, 10)
		tempCert := pkix.RevokedCertificate{
			SerialNumber:   tempBigInt,
			RevocationTime: time.Now(),
		}
		revokedCerts = append(revokedCerts, tempCert)
	}

	key, err := helpers.ParsePrivateKeyPEM([]byte(req.PrivateKey))
	if err != nil {
		log.Debug("Malformed private key %v", err)
		return errors.NewBadRequestString("Malformed Private Key")
	}

	result, err := cert.CreateCRL(rand.Reader, key, revokedCerts, time.Now(), newExpiryTime)

	return api.SendResponse(w, result)
}

// NewHandler returns a new http.Handler that handles a crl generation request.
func NewHandler() http.Handler {
	return api.HTTPHandler{
		Handler: api.HandlerFunc(crlHandler),
		Methods: []string{"POST"},
	}
}
