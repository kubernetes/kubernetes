package certdb

import (
	"time"
)

// CertificateRecord encodes a certificate and its metadata
// that will be recorded in a database.
type CertificateRecord struct {
	Serial    string    `db:"serial_number"`
	AKI       string    `db:"authority_key_identifier"`
	CALabel   string    `db:"ca_label"`
	Status    string    `db:"status"`
	Reason    int       `db:"reason"`
	Expiry    time.Time `db:"expiry"`
	RevokedAt time.Time `db:"revoked_at"`
	PEM       string    `db:"pem"`
}

// OCSPRecord encodes a OCSP response body and its metadata
// that will be recorded in a database.
type OCSPRecord struct {
	Serial string    `db:"serial_number"`
	AKI    string    `db:"authority_key_identifier"`
	Body   string    `db:"body"`
	Expiry time.Time `db:"expiry"`
}

// Accessor abstracts the CRUD of certdb objects from a DB.
type Accessor interface {
	InsertCertificate(cr CertificateRecord) error
	GetCertificate(serial, aki string) ([]CertificateRecord, error)
	GetUnexpiredCertificates() ([]CertificateRecord, error)
	GetRevokedAndUnexpiredCertificates() ([]CertificateRecord, error)
	GetRevokedAndUnexpiredCertificatesByLabel(label string) ([]CertificateRecord, error)
	RevokeCertificate(serial, aki string, reasonCode int) error
	InsertOCSP(rr OCSPRecord) error
	GetOCSP(serial, aki string) ([]OCSPRecord, error)
	GetUnexpiredOCSPs() ([]OCSPRecord, error)
	UpdateOCSP(serial, aki, body string, expiry time.Time) error
	UpsertOCSP(serial, aki, body string, expiry time.Time) error
}
