package signhandler

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/certdb/sql"
	"github.com/cloudflare/cfssl/certdb/testdb"
	"github.com/cloudflare/cfssl/config"
	"github.com/cloudflare/cfssl/signer"
	"github.com/cloudflare/cfssl/signer/local"
)

const (
	testCaFile    = "../testdata/ca.pem"
	testCaKeyFile = "../testdata/ca_key.pem"
	testCSRFile   = "../testdata/csr.pem"
)

// GetUnexpiredCertificates sometimes doesn't return a certificate with an
// expiry of 1m as above
var validLocalConfigLongerExpiry = `
{
	"signing": {
		"default": {
		    "usages": ["digital signature", "email protection"],
			"expiry": "10m"
		}
	}
}`

var dbAccessor certdb.Accessor

func TestSignerDBPersistence(t *testing.T) {
	conf, err := config.LoadConfig([]byte(validLocalConfigLongerExpiry))
	if err != nil {
		t.Fatal(err)
	}

	var s *local.Signer
	s, err = local.NewSignerFromFile(testCaFile, testCaKeyFile, conf.Signing)
	if err != nil {
		t.Fatal(err)
	}

	db := testdb.SQLiteDB("../../certdb/testdb/certstore_development.db")
	if err != nil {
		t.Fatal(err)
	}

	dbAccessor = sql.NewAccessor(db)
	s.SetDBAccessor(dbAccessor)

	var handler *api.HTTPHandler
	handler, err = NewHandlerFromSigner(signer.Signer(s))
	if err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(handler)
	defer ts.Close()

	var csrPEM, body []byte
	csrPEM, err = ioutil.ReadFile(testCSRFile)
	if err != nil {
		t.Fatal(err)
	}

	blob, err := json.Marshal(&map[string]string{"certificate_request": string(csrPEM)})
	if err != nil {
		t.Fatal(err)
	}

	var resp *http.Response
	resp, err = http.Post(ts.URL, "application/json", bytes.NewReader(blob))
	if err != nil {
		t.Fatal(err)
	}

	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	if resp.StatusCode != http.StatusOK {
		t.Fatal(resp.Status, string(body))
	}

	message := new(api.Response)
	err = json.Unmarshal(body, message)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}

	if !message.Success {
		t.Fatal("API operation failed")
	}

	crs, err := dbAccessor.GetUnexpiredCertificates()
	if err != nil {
		t.Fatal("Failed to get unexpired certificates")
	}

	if len(crs) != 1 {
		t.Fatal("Expected 1 unexpired certificate in the database after signing 1: len(crs)=", len(crs))
	}
}
