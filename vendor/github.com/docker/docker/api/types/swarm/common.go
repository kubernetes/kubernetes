package swarm

import "time"

// Version represents the internal object version.
type Version struct {
	Index uint64 `json:",omitempty"`
}

// Meta is a base object inherited by most of the other once.
type Meta struct {
	Version   Version   `json:",omitempty"`
	CreatedAt time.Time `json:",omitempty"`
	UpdatedAt time.Time `json:",omitempty"`
}

// Annotations represents how to describe an object.
type Annotations struct {
	Name   string            `json:",omitempty"`
	Labels map[string]string `json:"Labels"`
}

// Driver represents a driver (network, logging, secrets backend).
type Driver struct {
	Name    string            `json:",omitempty"`
	Options map[string]string `json:",omitempty"`
}

// TLSInfo represents the TLS information about what CA certificate is trusted,
// and who the issuer for a TLS certificate is
type TLSInfo struct {
	// TrustRoot is the trusted CA root certificate in PEM format
	TrustRoot string `json:",omitempty"`

	// CertIssuer is the raw subject bytes of the issuer
	CertIssuerSubject []byte `json:",omitempty"`

	// CertIssuerPublicKey is the raw public key bytes of the issuer
	CertIssuerPublicKey []byte `json:",omitempty"`
}
