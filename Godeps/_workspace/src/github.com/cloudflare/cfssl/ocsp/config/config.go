// Package config in the ocsp directory provides configuration data for an OCSP
// signer.
package config

import (
	"time"
)

// Config contains configuration information required to set up an OCSP
// signer. If PKCS11.Module is non-empty, PKCS11 signing will be used.
// Otherwise signing from a key file will be used.
type Config struct {
	CACertFile string
	ResponderCertFile string
	KeyFile string
	Interval time.Duration
	PKCS11 PKCS11Config
}

// PKCS11Config contains information specific to setting up a PKCS11 OCSP
// signer.
type PKCS11Config struct {
	Module string
	Token  string
	PIN    string
	Label  string
}

