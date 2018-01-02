package ocsprefresh

import (
	"encoding/hex"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/certdb/sql"
	"github.com/cloudflare/cfssl/certdb/testdb"
	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/helpers"
	"golang.org/x/crypto/ocsp"
	"io/ioutil"
)

var dbAccessor certdb.Accessor

func TestOCSPRefreshMain(t *testing.T) {
	db := testdb.SQLiteDB("../../certdb/testdb/certstore_development.db")

	certPEM, err := ioutil.ReadFile("../../ocsp/testdata/cert.pem")
	if err != nil {
		t.Fatal(err)
	}
	cert, err := helpers.ParseCertificatePEM(certPEM)
	if err != nil {
		t.Fatal(err)
	}

	expirationTime := time.Now().AddDate(1, 0, 0)
	certRecord := certdb.CertificateRecord{
		Serial: cert.SerialNumber.String(),
		AKI:    hex.EncodeToString(cert.AuthorityKeyId),
		Expiry: expirationTime,
		PEM:    string(certPEM),
		Status: "good",
	}

	dbAccessor = sql.NewAccessor(db)
	err = dbAccessor.InsertCertificate(certRecord)
	if err != nil {
		t.Fatal(err)
	}

	err = ocsprefreshMain([]string{}, cli.Config{
		CAFile:           "../../ocsp/testdata/ca.pem",
		ResponderFile:    "../../ocsp/testdata/server.crt",
		ResponderKeyFile: "../../ocsp/testdata/server.key",
		DBConfigFile:     "../testdata/db-config.json",
		Interval:         helpers.OneDay,
	})

	if err != nil {
		t.Fatal(err)
	}

	records, err := dbAccessor.GetUnexpiredOCSPs()
	if err != nil {
		t.Fatal("Failed to get OCSP responses")
	}

	if len(records) != 1 {
		t.Fatal("Expected one OCSP response")
	}

	var resp *ocsp.Response
	resp, err = ocsp.ParseResponse([]byte(records[0].Body), nil)
	if err != nil {
		t.Fatal("Failed to parse OCSP response")
	}
	if resp.Status != ocsp.Good {
		t.Fatal("Expected cert status 'good'")
	}

	err = dbAccessor.RevokeCertificate(certRecord.Serial, certRecord.AKI, ocsp.KeyCompromise)
	if err != nil {
		t.Fatal("Failed to revoke certificate")
	}

	err = ocsprefreshMain([]string{}, cli.Config{
		CAFile:           "../../ocsp/testdata/ca.pem",
		ResponderFile:    "../../ocsp/testdata/server.crt",
		ResponderKeyFile: "../../ocsp/testdata/server.key",
		DBConfigFile:     "../testdata/db-config.json",
		Interval:         helpers.OneDay,
	})

	if err != nil {
		t.Fatal(err)
	}

	records, err = dbAccessor.GetUnexpiredOCSPs()
	if err != nil {
		t.Fatal("Failed to get OCSP responses")
	}

	if len(records) != 1 {
		t.Fatal("Expected one OCSP response")
	}

	resp, err = ocsp.ParseResponse([]byte(records[0].Body), nil)
	if err != nil {
		t.Fatal("Failed to parse OCSP response")
	}
	if resp.Status != ocsp.Revoked {
		t.Fatal("Expected cert status 'revoked'")
	}
}
