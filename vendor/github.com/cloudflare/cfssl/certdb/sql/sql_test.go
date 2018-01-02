package sql

import (
	"math"
	"testing"
	"time"

	"github.com/cloudflare/cfssl/certdb"
	"github.com/cloudflare/cfssl/certdb/testdb"

	"github.com/jmoiron/sqlx"
)

const (
	sqliteDBFile = "../testdb/certstore_development.db"
	fakeAKI      = "fake_aki"
)

func TestNoDB(t *testing.T) {
	dba := &Accessor{}
	_, err := dba.GetCertificate("foobar serial", "random aki")
	if err == nil {
		t.Fatal("should return error")
	}
}

type TestAccessor struct {
	Accessor certdb.Accessor
	DB       *sqlx.DB
}

func (ta *TestAccessor) Truncate() {
	testdb.Truncate(ta.DB)
}

func TestSQLite(t *testing.T) {
	db := testdb.SQLiteDB(sqliteDBFile)
	ta := TestAccessor{
		Accessor: NewAccessor(db),
		DB:       db,
	}
	testEverything(ta, t)
}

// roughlySameTime decides if t1 and t2 are close enough.
func roughlySameTime(t1, t2 time.Time) bool {
	// return true if the difference is smaller than 1 sec.
	return math.Abs(float64(t1.Sub(t2))) < float64(time.Second)
}

func testEverything(ta TestAccessor, t *testing.T) {
	testInsertCertificateAndGetCertificate(ta, t)
	testInsertCertificateAndGetUnexpiredCertificate(ta, t)
	testUpdateCertificateAndGetCertificate(ta, t)
	testInsertOCSPAndGetOCSP(ta, t)
	testInsertOCSPAndGetUnexpiredOCSP(ta, t)
	testUpdateOCSPAndGetOCSP(ta, t)
	testUpsertOCSPAndGetOCSP(ta, t)
}

func testInsertCertificateAndGetCertificate(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	expiry := time.Date(2010, time.December, 25, 23, 0, 0, 0, time.UTC)
	want := certdb.CertificateRecord{
		PEM:    "fake cert data",
		Serial: "fake serial",
		AKI:    fakeAKI,
		Status: "good",
		Reason: 0,
		Expiry: expiry,
	}

	if err := ta.Accessor.InsertCertificate(want); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetCertificate(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}

	if len(rets) != 1 {
		t.Fatal("should only return one record.")
	}

	got := rets[0]

	// relfection comparison with zero time objects are not stable as it seems
	if want.Serial != got.Serial || want.Status != got.Status ||
		want.AKI != got.AKI || !got.RevokedAt.IsZero() ||
		want.PEM != got.PEM || !roughlySameTime(got.Expiry, expiry) {
		t.Errorf("want Certificate %+v, got %+v", want, got)
	}

	unexpired, err := ta.Accessor.GetUnexpiredCertificates()

	if err != nil {
		t.Fatal(err)
	}

	if len(unexpired) != 0 {
		t.Error("should not have unexpired certificate record")
	}
}

func testInsertCertificateAndGetUnexpiredCertificate(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	expiry := time.Now().Add(time.Minute)
	want := certdb.CertificateRecord{
		PEM:    "fake cert data",
		Serial: "fake serial 2",
		AKI:    fakeAKI,
		Status: "good",
		Reason: 0,
		Expiry: expiry,
	}

	if err := ta.Accessor.InsertCertificate(want); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetCertificate(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}

	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got := rets[0]

	// relfection comparison with zero time objects are not stable as it seems
	if want.Serial != got.Serial || want.Status != got.Status ||
		want.AKI != got.AKI || !got.RevokedAt.IsZero() ||
		want.PEM != got.PEM || !roughlySameTime(got.Expiry, expiry) {
		t.Errorf("want Certificate %+v, got %+v", want, got)
	}

	unexpired, err := ta.Accessor.GetUnexpiredCertificates()

	if err != nil {
		t.Fatal(err)
	}

	if len(unexpired) != 1 {
		t.Error("Should have 1 unexpired certificate record:", len(unexpired))
	}
}

func testUpdateCertificateAndGetCertificate(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	expiry := time.Date(2010, time.December, 25, 23, 0, 0, 0, time.UTC)
	want := certdb.CertificateRecord{
		PEM:    "fake cert data",
		Serial: "fake serial 3",
		AKI:    fakeAKI,
		Status: "good",
		Reason: 0,
		Expiry: expiry,
	}

	// Make sure the revoke on a non-existent cert fails
	if err := ta.Accessor.RevokeCertificate(want.Serial, want.AKI, 2); err == nil {
		t.Fatal("Expected error")
	}

	if err := ta.Accessor.InsertCertificate(want); err != nil {
		t.Fatal(err)
	}

	// reason 2 is CACompromise
	if err := ta.Accessor.RevokeCertificate(want.Serial, want.AKI, 2); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetCertificate(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}

	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got := rets[0]

	// relfection comparison with zero time objects are not stable as it seems
	if want.Serial != got.Serial || got.Status != "revoked" ||
		want.AKI != got.AKI || got.RevokedAt.IsZero() ||
		want.PEM != got.PEM {
		t.Errorf("want Certificate %+v, got %+v", want, got)
	}
}

func testInsertOCSPAndGetOCSP(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	expiry := time.Date(2010, time.December, 25, 23, 0, 0, 0, time.UTC)
	want := certdb.OCSPRecord{
		Serial: "fake serial",
		AKI:    fakeAKI,
		Body:   "fake body",
		Expiry: expiry,
	}
	setupGoodCert(ta, t, want)

	if err := ta.Accessor.InsertOCSP(want); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetOCSP(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}
	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got := rets[0]

	if want.Serial != got.Serial || want.Body != got.Body ||
		!roughlySameTime(want.Expiry, got.Expiry) {
		t.Errorf("want OCSP %+v, got %+v", want, got)
	}

	unexpired, err := ta.Accessor.GetUnexpiredOCSPs()

	if err != nil {
		t.Fatal(err)
	}

	if len(unexpired) != 0 {
		t.Error("should not have unexpired certificate record")
	}
}

func testInsertOCSPAndGetUnexpiredOCSP(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	want := certdb.OCSPRecord{
		Serial: "fake serial 2",
		AKI:    fakeAKI,
		Body:   "fake body",
		Expiry: time.Now().Add(time.Minute),
	}
	setupGoodCert(ta, t, want)

	if err := ta.Accessor.InsertOCSP(want); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetOCSP(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}
	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got := rets[0]

	if want.Serial != got.Serial || want.Body != got.Body ||
		!roughlySameTime(want.Expiry, got.Expiry) {
		t.Errorf("want OCSP %+v, got %+v", want, got)
	}

	unexpired, err := ta.Accessor.GetUnexpiredOCSPs()

	if err != nil {
		t.Fatal(err)
	}

	if len(unexpired) != 1 {
		t.Error("should not have other than 1 unexpired certificate record:", len(unexpired))
	}
}

func testUpdateOCSPAndGetOCSP(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	want := certdb.OCSPRecord{
		Serial: "fake serial 3",
		AKI:    fakeAKI,
		Body:   "fake body",
		Expiry: time.Date(2010, time.December, 25, 23, 0, 0, 0, time.UTC),
	}
	setupGoodCert(ta, t, want)

	// Make sure the update fails
	if err := ta.Accessor.UpdateOCSP(want.Serial, want.AKI, want.Body, want.Expiry); err == nil {
		t.Fatal("Expected error")
	}

	if err := ta.Accessor.InsertOCSP(want); err != nil {
		t.Fatal(err)
	}

	want.Body = "fake body revoked"
	newExpiry := time.Now().Add(time.Hour)
	if err := ta.Accessor.UpdateOCSP(want.Serial, want.AKI, want.Body, newExpiry); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetOCSP(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}
	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got := rets[0]

	want.Expiry = newExpiry
	if want.Serial != got.Serial || got.Body != "fake body revoked" ||
		!roughlySameTime(newExpiry, got.Expiry) {
		t.Errorf("want OCSP %+v, got %+v", want, got)
	}
}

func testUpsertOCSPAndGetOCSP(ta TestAccessor, t *testing.T) {
	ta.Truncate()

	want := certdb.OCSPRecord{
		Serial: "fake serial 3",
		AKI:    fakeAKI,
		Body:   "fake body",
		Expiry: time.Date(2010, time.December, 25, 23, 0, 0, 0, time.UTC),
	}
	setupGoodCert(ta, t, want)

	if err := ta.Accessor.UpsertOCSP(want.Serial, want.AKI, want.Body, want.Expiry); err != nil {
		t.Fatal(err)
	}

	rets, err := ta.Accessor.GetOCSP(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}
	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got := rets[0]

	if want.Serial != got.Serial || want.Body != got.Body ||
		!roughlySameTime(want.Expiry, got.Expiry) {
		t.Errorf("want OCSP %+v, got %+v", want, got)
	}

	newExpiry := time.Now().Add(time.Hour)
	if err := ta.Accessor.UpsertOCSP(want.Serial, want.AKI, "fake body revoked", newExpiry); err != nil {
		t.Fatal(err)
	}

	rets, err = ta.Accessor.GetOCSP(want.Serial, want.AKI)
	if err != nil {
		t.Fatal(err)
	}
	if len(rets) != 1 {
		t.Fatal("should return exactly one record")
	}

	got = rets[0]

	want.Expiry = newExpiry
	if want.Serial != got.Serial || got.Body != "fake body revoked" ||
		!roughlySameTime(newExpiry, got.Expiry) {
		t.Errorf("want OCSP %+v, got %+v", want, got)
	}
}

func setupGoodCert(ta TestAccessor, t *testing.T, r certdb.OCSPRecord) {
	certWant := certdb.CertificateRecord{
		AKI:     r.AKI,
		CALabel: "default",
		Expiry:  time.Now().Add(time.Minute),
		PEM:     "fake cert data",
		Serial:  r.Serial,
		Status:  "good",
		Reason:  0,
	}

	if err := ta.Accessor.InsertCertificate(certWant); err != nil {
		t.Fatal(err)
	}
}
