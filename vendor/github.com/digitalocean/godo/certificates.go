package godo

import (
	"path"

	"github.com/digitalocean/godo/context"
)

const certificatesBasePath = "/v2/certificates"

// CertificatesService is an interface for managing certificates with the DigitalOcean API.
// See: https://developers.digitalocean.com/documentation/v2/#certificates
type CertificatesService interface {
	Get(context.Context, string) (*Certificate, *Response, error)
	List(context.Context, *ListOptions) ([]Certificate, *Response, error)
	Create(context.Context, *CertificateRequest) (*Certificate, *Response, error)
	Delete(context.Context, string) (*Response, error)
}

// Certificate represents a DigitalOcean certificate configuration.
type Certificate struct {
	ID              string `json:"id,omitempty"`
	Name            string `json:"name,omitempty"`
	NotAfter        string `json:"not_after,omitempty"`
	SHA1Fingerprint string `json:"sha1_fingerprint,omitempty"`
	Created         string `json:"created_at,omitempty"`
}

// CertificateRequest represents configuration for a new certificate.
type CertificateRequest struct {
	Name             string `json:"name,omitempty"`
	PrivateKey       string `json:"private_key,omitempty"`
	LeafCertificate  string `json:"leaf_certificate,omitempty"`
	CertificateChain string `json:"certificate_chain,omitempty"`
}

type certificateRoot struct {
	Certificate *Certificate `json:"certificate"`
}

type certificatesRoot struct {
	Certificates []Certificate `json:"certificates"`
	Links        *Links        `json:"links"`
}

// CertificatesServiceOp handles communication with certificates methods of the DigitalOcean API.
type CertificatesServiceOp struct {
	client *Client
}

var _ CertificatesService = &CertificatesServiceOp{}

// Get an existing certificate by its identifier.
func (c *CertificatesServiceOp) Get(ctx context.Context, cID string) (*Certificate, *Response, error) {
	urlStr := path.Join(certificatesBasePath, cID)

	req, err := c.client.NewRequest(ctx, "GET", urlStr, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(certificateRoot)
	resp, err := c.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Certificate, resp, nil
}

// List all certificates.
func (c *CertificatesServiceOp) List(ctx context.Context, opt *ListOptions) ([]Certificate, *Response, error) {
	urlStr, err := addOptions(certificatesBasePath, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := c.client.NewRequest(ctx, "GET", urlStr, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(certificatesRoot)
	resp, err := c.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Certificates, resp, nil
}

// Create a new certificate with provided configuration.
func (c *CertificatesServiceOp) Create(ctx context.Context, cr *CertificateRequest) (*Certificate, *Response, error) {
	req, err := c.client.NewRequest(ctx, "POST", certificatesBasePath, cr)
	if err != nil {
		return nil, nil, err
	}

	root := new(certificateRoot)
	resp, err := c.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Certificate, resp, nil
}

// Delete a certificate by its identifier.
func (c *CertificatesServiceOp) Delete(ctx context.Context, cID string) (*Response, error) {
	urlStr := path.Join(certificatesBasePath, cID)

	req, err := c.client.NewRequest(ctx, "DELETE", urlStr, nil)
	if err != nil {
		return nil, err
	}

	return c.client.Do(ctx, req, nil)
}
