// Copyright 2019 Thales e-Security, Inc
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package crypto11

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/asn1"
	"math/big"

	"github.com/miekg/pkcs11"
	"github.com/pkg/errors"
)

// FindCertificate retrieves a previously imported certificate. Any combination of id, label
// and serial can be provided. An error is return if all are nil.
func findCertificate(session *pkcs11Session, id []byte, label []byte, serial *big.Int) (cert *x509.Certificate, err error) {

	rawCertificate, err := findRawCertificate(session, id, label, serial)
	if err != nil {
		return nil, err
	}

	if rawCertificate != nil {
		cert, err = x509.ParseCertificate(rawCertificate)
		if err != nil {
			return nil, err
		}
	}

	return cert, err
}

func findRawCertificate(session *pkcs11Session, id []byte, label []byte, serial *big.Int) (rawCertificate []byte, err error) {
	if id == nil && label == nil && serial == nil {
		return nil, errors.New("id, label and serial cannot all be nil")
	}

	var template []*pkcs11.Attribute

	if id != nil {
		template = append(template, pkcs11.NewAttribute(pkcs11.CKA_ID, id))
	}
	if label != nil {
		template = append(template, pkcs11.NewAttribute(pkcs11.CKA_LABEL, label))
	}
	if serial != nil {
		derSerial, err := asn1.Marshal(serial)
		if err != nil {
			return nil, errors.WithMessage(err, "failed to encode serial")
		}

		template = append(template, pkcs11.NewAttribute(pkcs11.CKA_SERIAL_NUMBER, derSerial))
	}

	template = append(template, pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_CERTIFICATE))

	if err = session.ctx.FindObjectsInit(session.handle, template); err != nil {
		return nil, err
	}
	defer func() {
		finalErr := session.ctx.FindObjectsFinal(session.handle)
		if err == nil {
			err = finalErr
		}
	}()

	handles, _, err := session.ctx.FindObjects(session.handle, 1)
	if err != nil {
		return nil, err
	}
	if len(handles) == 0 {
		return nil, nil
	}

	attributes := []*pkcs11.Attribute{
		pkcs11.NewAttribute(pkcs11.CKA_VALUE, 0),
	}

	if attributes, err = session.ctx.GetAttributeValue(session.handle, handles[0], attributes); err != nil {
		return nil, err
	}

	rawCertificate = attributes[0].Value

	return
}

// FindCertificate retrieves a previously imported certificate. Any combination of id, label
// and serial can be provided. An error is return if all are nil.
func (c *Context) FindCertificate(id []byte, label []byte, serial *big.Int) (*x509.Certificate, error) {

	if c.closed.Get() {
		return nil, errClosed
	}

	var cert *x509.Certificate
	err := c.withSession(func(session *pkcs11Session) (err error) {
		cert, err = findCertificate(session, id, label, serial)
		return err
	})

	return cert, err
}

func (c *Context) FindAllPairedCertificates() (certificates []tls.Certificate, err error) {
	if c.closed.Get() {
		return nil, errClosed
	}

	err = c.withSession(func(session *pkcs11Session) error {
		// Add the private key class to the template to find the private half
		privAttributes := AttributeSet{}
		err = privAttributes.Set(CkaClass, pkcs11.CKO_PRIVATE_KEY)
		if err != nil {
			return err
		}

		privHandles, err := findKeysWithAttributes(session, privAttributes.ToSlice())
		if err != nil {
			return err
		}

		for _, privHandle := range privHandles {

			privateKey, certificate, err := c.makeKeyPair(session, &privHandle)

			if err == errNoCkaId || err == errNoPublicHalf {
				continue
			}

			if err != nil {
				return err
			}

			if certificate == nil {
				continue
			}

			tlsCert := tls.Certificate{
				Leaf:       certificate,
				PrivateKey: privateKey,
			}

			tlsCert.Certificate = append(tlsCert.Certificate, certificate.Raw)
			certificates = append(certificates, tlsCert)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return
}

// ImportCertificate imports a certificate onto the token. The id parameter is used to
// set CKA_ID and must be non-nil.
func (c *Context) ImportCertificate(id []byte, certificate *x509.Certificate) error {
	if c.closed.Get() {
		return errClosed
	}

	if err := notNilBytes(id, "id"); err != nil {
		return err
	}

	template, err := NewAttributeSetWithID(id)
	if err != nil {
		return err
	}
	return c.ImportCertificateWithAttributes(template, certificate)
}

// ImportCertificateWithLabel imports a certificate onto the token.  The id and label parameters are used to
// set CKA_ID and CKA_LABEL respectively and must be non-nil.
func (c *Context) ImportCertificateWithLabel(id []byte, label []byte, certificate *x509.Certificate) error {
	if c.closed.Get() {
		return errClosed
	}

	if err := notNilBytes(id, "id"); err != nil {
		return err
	}
	if err := notNilBytes(label, "label"); err != nil {
		return err
	}

	template, err := NewAttributeSetWithIDAndLabel(id, label)
	if err != nil {
		return err
	}
	return c.ImportCertificateWithAttributes(template, certificate)
}

// ImportCertificateWithAttributes imports a certificate onto the token. After this function returns, template
// will contain the attributes applied to the certificate. If required attributes are missing, they will be set to a
// default value.
func (c *Context) ImportCertificateWithAttributes(template AttributeSet, certificate *x509.Certificate) error {
	if c.closed.Get() {
		return errClosed
	}

	if certificate == nil {
		return errors.New("certificate cannot be nil")
	}

	serial, err := asn1.Marshal(certificate.SerialNumber)
	if err != nil {
		return err
	}

	template.AddIfNotPresent([]*pkcs11.Attribute{
		pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_CERTIFICATE),
		pkcs11.NewAttribute(pkcs11.CKA_CERTIFICATE_TYPE, pkcs11.CKC_X_509),
		pkcs11.NewAttribute(pkcs11.CKA_TOKEN, true),
		pkcs11.NewAttribute(pkcs11.CKA_PRIVATE, false),
		pkcs11.NewAttribute(pkcs11.CKA_SUBJECT, certificate.RawSubject),
		pkcs11.NewAttribute(pkcs11.CKA_ISSUER, certificate.RawIssuer),
		pkcs11.NewAttribute(pkcs11.CKA_SERIAL_NUMBER, serial),
		pkcs11.NewAttribute(pkcs11.CKA_VALUE, certificate.Raw),
	})

	err = c.withSession(func(session *pkcs11Session) error {
		_, err = session.ctx.CreateObject(session.handle, template.ToSlice())
		return err
	})

	return err
}

// DeleteCertificate destroys a previously imported certificate. it will return
// nil if succeeds or if the certificate does not exist. Any combination of id,
// label and serial can be provided. An error is return if all are nil.
func (c *Context) DeleteCertificate(id []byte, label []byte, serial *big.Int) error {
	if id == nil && label == nil && serial == nil {
		return errors.New("id, label and serial cannot all be nil")
	}

	template := []*pkcs11.Attribute{
		pkcs11.NewAttribute(pkcs11.CKA_CLASS, pkcs11.CKO_CERTIFICATE),
	}

	if id != nil {
		template = append(template, pkcs11.NewAttribute(pkcs11.CKA_ID, id))
	}
	if label != nil {
		template = append(template, pkcs11.NewAttribute(pkcs11.CKA_LABEL, label))
	}
	if serial != nil {
		asn1Serial, err := asn1.Marshal(serial)
		if err != nil {
			return err
		}
		template = append(template, pkcs11.NewAttribute(pkcs11.CKA_SERIAL_NUMBER, asn1Serial))
	}

	err := c.withSession(func(session *pkcs11Session) error {
		err := session.ctx.FindObjectsInit(session.handle, template)
		if err != nil {
			return err
		}
		handles, _, err := session.ctx.FindObjects(session.handle, 1)
		finalErr := session.ctx.FindObjectsFinal(session.handle)
		if err != nil {
			return err
		}
		if finalErr != nil {
			return finalErr
		}
		if len(handles) == 0 {
			return nil
		}
		return session.ctx.DestroyObject(session.handle, handles[0])
	})

	return err
}
