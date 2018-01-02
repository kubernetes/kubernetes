package revoke

import (
	"testing"
	"time"

	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/certdb/sql"
	"github.com/cloudflare/cfssl/certdb/testdb"
	"github.com/cloudflare/cfssl/cli"
	"golang.org/x/crypto/ocsp"
)

var dbAccessor certdb.Accessor

const (
	fakeAKI = "fake aki"
)

func prepDB() (err error) {
	db := testdb.SQLiteDB("../../certdb/testdb/certstore_development.db")
	expirationTime := time.Now().AddDate(1, 0, 0)
	var cert = certdb.CertificateRecord{
		Serial: "1",
		AKI:    fakeAKI,
		Expiry: expirationTime,
		PEM:    "unexpired cert",
	}

	dbAccessor = sql.NewAccessor(db)
	err = dbAccessor.InsertCertificate(cert)
	if err != nil {
		return err
	}

	return
}

func TestRevokeMain(t *testing.T) {
	err := prepDB()
	if err != nil {
		t.Fatal(err)
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", AKI: fakeAKI, DBConfigFile: "../testdata/db-config.json"})
	if err != nil {
		t.Fatal(err)
	}

	crs, err := dbAccessor.GetCertificate("1", fakeAKI)
	if err != nil {
		t.Fatal("Failed to get certificate")
	}

	if len(crs) != 1 {
		t.Fatal("Failed to get exactly one certificate")
	}

	cr := crs[0]
	if cr.Status != "revoked" {
		t.Fatal("Certificate not marked revoked after we revoked it")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", AKI: fakeAKI, Reason: "2", DBConfigFile: "../testdata/db-config.json"})
	if err != nil {
		t.Fatal(err)
	}

	crs, err = dbAccessor.GetCertificate("1", fakeAKI)
	if err != nil {
		t.Fatal("Failed to get certificate")
	}
	if len(crs) != 1 {
		t.Fatal("Failed to get exactly one certificate")
	}

	cr = crs[0]
	if cr.Reason != 2 {
		t.Fatal("Certificate revocation reason incorrect")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", AKI: fakeAKI, Reason: "Superseded", DBConfigFile: "../testdata/db-config.json"})
	if err != nil {
		t.Fatal(err)
	}

	crs, err = dbAccessor.GetCertificate("1", fakeAKI)
	if err != nil {
		t.Fatal("Failed to get certificate")
	}
	if len(crs) != 1 {
		t.Fatal("Failed to get exactly one certificate")
	}

	cr = crs[0]
	if cr.Reason != ocsp.Superseded {
		t.Fatal("Certificate revocation reason incorrect")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", AKI: fakeAKI, Reason: "invalid_reason", DBConfigFile: "../testdata/db-config.json"})
	if err == nil {
		t.Fatal("Expected error from invalid reason")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", AKI: fakeAKI, Reason: "999", DBConfigFile: "../testdata/db-config.json"})
	if err == nil {
		t.Fatal("Expected error from invalid reason")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "2", AKI: fakeAKI, DBConfigFile: "../testdata/db-config.json"})
	if err == nil {
		t.Fatal("Expected error from unrecognized serial number")
	}

	err = revokeMain([]string{}, cli.Config{AKI: fakeAKI, DBConfigFile: "../testdata/db-config.json"})
	if err == nil {
		t.Fatal("Expected error from missing serial number")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", AKI: fakeAKI})
	if err == nil {
		t.Fatal("Expected error from missing db config")
	}

	err = revokeMain([]string{}, cli.Config{Serial: "1", DBConfigFile: "../testdata/db-config.json"})
	if err == nil {
		t.Fatal("Expected error from missing aki")
	}
}
