package sql

import (
	"errors"
	"fmt"
	"time"

	"github.com/cloudflare/cfssl/certdb"
	cferr "github.com/cloudflare/cfssl/errors"

	"github.com/jmoiron/sqlx"
	"github.com/kisielk/sqlstruct"
)

// Match to sqlx
func init() {
	sqlstruct.TagName = "db"
}

const (
	insertSQL = `
INSERT INTO certificates (serial_number, authority_key_identifier, ca_label, status, reason, expiry, revoked_at, pem)
	VALUES (:serial_number, :authority_key_identifier, :ca_label, :status, :reason, :expiry, :revoked_at, :pem);`

	selectSQL = `
SELECT %s FROM certificates
	WHERE (serial_number = ? AND authority_key_identifier = ?);`

	selectAllUnexpiredSQL = `
SELECT %s FROM certificates
	WHERE CURRENT_TIMESTAMP < expiry;`

	selectAllRevokedAndUnexpiredWithLabelSQL = `
SELECT %s FROM certificates
	WHERE CURRENT_TIMESTAMP < expiry AND status='revoked' AND ca_label= ?;`

	selectAllRevokedAndUnexpiredSQL = `
SELECT %s FROM certificates
	WHERE CURRENT_TIMESTAMP < expiry AND status='revoked';`

	updateRevokeSQL = `
UPDATE certificates
	SET status='revoked', revoked_at=CURRENT_TIMESTAMP, reason=:reason
	WHERE (serial_number = :serial_number AND authority_key_identifier = :authority_key_identifier);`

	insertOCSPSQL = `
INSERT INTO ocsp_responses (serial_number, authority_key_identifier, body, expiry)
  VALUES (:serial_number, :authority_key_identifier, :body, :expiry);`

	updateOCSPSQL = `
UPDATE ocsp_responses
  SET body = :body, expiry = :expiry
	WHERE (serial_number = :serial_number AND authority_key_identifier = :authority_key_identifier);`

	selectAllUnexpiredOCSPSQL = `
SELECT %s FROM ocsp_responses
	WHERE CURRENT_TIMESTAMP < expiry;`

	selectOCSPSQL = `
SELECT %s FROM ocsp_responses
  WHERE (serial_number = ? AND authority_key_identifier = ?);`
)

// Accessor implements certdb.Accessor interface.
type Accessor struct {
	db *sqlx.DB
}

func wrapSQLError(err error) error {
	if err != nil {
		return cferr.Wrap(cferr.CertStoreError, cferr.Unknown, err)
	}
	return nil
}

func (d *Accessor) checkDB() error {
	if d.db == nil {
		return cferr.Wrap(cferr.CertStoreError, cferr.Unknown,
			errors.New("unknown db object, please check SetDB method"))
	}
	return nil
}

// NewAccessor returns a new Accessor.
func NewAccessor(db *sqlx.DB) *Accessor {
	return &Accessor{db: db}
}

// SetDB changes the underlying sql.DB object Accessor is manipulating.
func (d *Accessor) SetDB(db *sqlx.DB) {
	d.db = db
	return
}

// InsertCertificate puts a certdb.CertificateRecord into db.
func (d *Accessor) InsertCertificate(cr certdb.CertificateRecord) error {
	err := d.checkDB()
	if err != nil {
		return err
	}

	res, err := d.db.NamedExec(insertSQL, &certdb.CertificateRecord{
		Serial:    cr.Serial,
		AKI:       cr.AKI,
		CALabel:   cr.CALabel,
		Status:    cr.Status,
		Reason:    cr.Reason,
		Expiry:    cr.Expiry.UTC(),
		RevokedAt: cr.RevokedAt.UTC(),
		PEM:       cr.PEM,
	})
	if err != nil {
		return wrapSQLError(err)
	}

	numRowsAffected, err := res.RowsAffected()

	if numRowsAffected == 0 {
		return cferr.Wrap(cferr.CertStoreError, cferr.InsertionFailed, fmt.Errorf("failed to insert the certificate record"))
	}

	if numRowsAffected != 1 {
		return wrapSQLError(fmt.Errorf("%d rows are affected, should be 1 row", numRowsAffected))
	}

	return err
}

// GetCertificate gets a certdb.CertificateRecord indexed by serial.
func (d *Accessor) GetCertificate(serial, aki string) (crs []certdb.CertificateRecord, err error) {
	err = d.checkDB()
	if err != nil {
		return nil, err
	}

	err = d.db.Select(&crs, fmt.Sprintf(d.db.Rebind(selectSQL), sqlstruct.Columns(certdb.CertificateRecord{})), serial, aki)
	if err != nil {
		return nil, wrapSQLError(err)
	}

	return crs, nil
}

// GetUnexpiredCertificates gets all unexpired certificate from db.
func (d *Accessor) GetUnexpiredCertificates() (crs []certdb.CertificateRecord, err error) {
	err = d.checkDB()
	if err != nil {
		return nil, err
	}

	err = d.db.Select(&crs, fmt.Sprintf(d.db.Rebind(selectAllUnexpiredSQL), sqlstruct.Columns(certdb.CertificateRecord{})))
	if err != nil {
		return nil, wrapSQLError(err)
	}

	return crs, nil
}

// GetRevokedAndUnexpiredCertificates gets all revoked and unexpired certificate from db (for CRLs).
func (d *Accessor) GetRevokedAndUnexpiredCertificates() (crs []certdb.CertificateRecord, err error) {
	err = d.checkDB()
	if err != nil {
		return nil, err
	}

	err = d.db.Select(&crs, fmt.Sprintf(d.db.Rebind(selectAllRevokedAndUnexpiredSQL), sqlstruct.Columns(certdb.CertificateRecord{})))
	if err != nil {
		return nil, wrapSQLError(err)
	}

	return crs, nil
}

// GetRevokedAndUnexpiredCertificatesByLabel gets all revoked and unexpired certificate from db (for CRLs) with specified ca_label.
func (d *Accessor) GetRevokedAndUnexpiredCertificatesByLabel(label string) (crs []certdb.CertificateRecord, err error) {
	err = d.checkDB()
	if err != nil {
		return nil, err
	}

	err = d.db.Select(&crs, fmt.Sprintf(d.db.Rebind(selectAllRevokedAndUnexpiredWithLabelSQL), sqlstruct.Columns(certdb.CertificateRecord{})), label)
	if err != nil {
		return nil, wrapSQLError(err)
	}

	return crs, nil
}

// RevokeCertificate updates a certificate with a given serial number and marks it revoked.
func (d *Accessor) RevokeCertificate(serial, aki string, reasonCode int) error {
	err := d.checkDB()
	if err != nil {
		return err
	}

	result, err := d.db.NamedExec(updateRevokeSQL, &certdb.CertificateRecord{
		AKI:    aki,
		Reason: reasonCode,
		Serial: serial,
	})
	if err != nil {
		return wrapSQLError(err)
	}

	numRowsAffected, err := result.RowsAffected()

	if numRowsAffected == 0 {
		return cferr.Wrap(cferr.CertStoreError, cferr.RecordNotFound, fmt.Errorf("failed to revoke the certificate: certificate not found"))
	}

	if numRowsAffected != 1 {
		return wrapSQLError(fmt.Errorf("%d rows are affected, should be 1 row", numRowsAffected))
	}

	return err
}

// InsertOCSP puts a new certdb.OCSPRecord into the db.
func (d *Accessor) InsertOCSP(rr certdb.OCSPRecord) error {
	err := d.checkDB()
	if err != nil {
		return err
	}

	result, err := d.db.NamedExec(insertOCSPSQL, &certdb.OCSPRecord{
		AKI:    rr.AKI,
		Body:   rr.Body,
		Expiry: rr.Expiry.UTC(),
		Serial: rr.Serial,
	})
	if err != nil {
		return wrapSQLError(err)
	}

	numRowsAffected, err := result.RowsAffected()

	if numRowsAffected == 0 {
		return cferr.Wrap(cferr.CertStoreError, cferr.InsertionFailed, fmt.Errorf("failed to insert the OCSP record"))
	}

	if numRowsAffected != 1 {
		return wrapSQLError(fmt.Errorf("%d rows are affected, should be 1 row", numRowsAffected))
	}

	return err
}

// GetOCSP retrieves a certdb.OCSPRecord from db by serial.
func (d *Accessor) GetOCSP(serial, aki string) (ors []certdb.OCSPRecord, err error) {
	err = d.checkDB()
	if err != nil {
		return nil, err
	}

	err = d.db.Select(&ors, fmt.Sprintf(d.db.Rebind(selectOCSPSQL), sqlstruct.Columns(certdb.OCSPRecord{})), serial, aki)
	if err != nil {
		return nil, wrapSQLError(err)
	}

	return ors, nil
}

// GetUnexpiredOCSPs retrieves all unexpired certdb.OCSPRecord from db.
func (d *Accessor) GetUnexpiredOCSPs() (ors []certdb.OCSPRecord, err error) {
	err = d.checkDB()
	if err != nil {
		return nil, err
	}

	err = d.db.Select(&ors, fmt.Sprintf(d.db.Rebind(selectAllUnexpiredOCSPSQL), sqlstruct.Columns(certdb.OCSPRecord{})))
	if err != nil {
		return nil, wrapSQLError(err)
	}

	return ors, nil
}

// UpdateOCSP updates a ocsp response record with a given serial number.
func (d *Accessor) UpdateOCSP(serial, aki, body string, expiry time.Time) error {
	err := d.checkDB()
	if err != nil {
		return err
	}

	result, err := d.db.NamedExec(updateOCSPSQL, &certdb.OCSPRecord{
		AKI:    aki,
		Body:   body,
		Expiry: expiry.UTC(),
		Serial: serial,
	})
	if err != nil {
		return wrapSQLError(err)
	}

	numRowsAffected, err := result.RowsAffected()

	if numRowsAffected == 0 {
		return cferr.Wrap(cferr.CertStoreError, cferr.RecordNotFound, fmt.Errorf("failed to update the OCSP record"))
	}

	if numRowsAffected != 1 {
		return wrapSQLError(fmt.Errorf("%d rows are affected, should be 1 row", numRowsAffected))
	}

	return err
}

// UpsertOCSP update a ocsp response record with a given serial number,
// or insert the record if it doesn't yet exist in the db
// Implementation note:
// We didn't implement 'upsert' with SQL statement and we lost race condition
// prevention provided by underlying DBMS.
// Reasoning:
// 1. it's diffcult to support multiple DBMS backends in the same time, the
// SQL syntax differs from one to another.
// 2. we don't need a strict simultaneous consistency between OCSP and certificate
// status. It's OK that a OCSP response still shows 'good' while the
// corresponding certificate is being revoked seconds ago, as long as the OCSP
// response catches up to be eventually consistent (within hours to days).
// Write race condition between OCSP writers on OCSP table is not a problem,
// since we don't have write race condition on Certificate table and OCSP
// writers should periodically use Certificate table to update OCSP table
// to catch up.
func (d *Accessor) UpsertOCSP(serial, aki, body string, expiry time.Time) error {
	err := d.checkDB()
	if err != nil {
		return err
	}

	result, err := d.db.NamedExec(updateOCSPSQL, &certdb.OCSPRecord{
		AKI:    aki,
		Body:   body,
		Expiry: expiry.UTC(),
		Serial: serial,
	})

	if err != nil {
		return wrapSQLError(err)
	}

	numRowsAffected, err := result.RowsAffected()

	if numRowsAffected == 0 {
		return d.InsertOCSP(certdb.OCSPRecord{Serial: serial, AKI: aki, Body: body, Expiry: expiry})
	}

	if numRowsAffected != 1 {
		return wrapSQLError(fmt.Errorf("%d rows are affected, should be 1 row", numRowsAffected))
	}

	return err
}
