package ssl

import (
	"errors"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

var (
	errPrivateKey     = errors.New("PrivateKey is a required field")
	errCertificate    = errors.New("Certificate is a required field")
	errIntCertificate = errors.New("IntCertificate is a required field")
)

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Update operation in this package.
type UpdateOptsBuilder interface {
	ToSSLUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is the common options struct used in this package's Update
// operation.
type UpdateOpts struct {
	// Required - consult the SSLTermConfig struct for more info.
	SecurePort int

	// Required - consult the SSLTermConfig struct for more info.
	PrivateKey string

	// Required - consult the SSLTermConfig struct for more info.
	Certificate string

	// Required - consult the SSLTermConfig struct for more info.
	IntCertificate string

	// Optional - consult the SSLTermConfig struct for more info.
	Enabled *bool

	// Optional - consult the SSLTermConfig struct for more info.
	SecureTrafficOnly *bool
}

// ToSSLUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToSSLUpdateMap() (map[string]interface{}, error) {
	ssl := make(map[string]interface{})

	if opts.SecurePort == 0 {
		return ssl, errors.New("SecurePort needs to be an integer greater than 0")
	}
	if opts.PrivateKey == "" {
		return ssl, errPrivateKey
	}
	if opts.Certificate == "" {
		return ssl, errCertificate
	}
	if opts.IntCertificate == "" {
		return ssl, errIntCertificate
	}

	ssl["securePort"] = opts.SecurePort
	ssl["privateKey"] = opts.PrivateKey
	ssl["certificate"] = opts.Certificate
	ssl["intermediateCertificate"] = opts.IntCertificate

	if opts.Enabled != nil {
		ssl["enabled"] = &opts.Enabled
	}

	if opts.SecureTrafficOnly != nil {
		ssl["secureTrafficOnly"] = &opts.SecureTrafficOnly
	}

	return map[string]interface{}{"sslTermination": ssl}, nil
}

// Update is the operation responsible for updating the SSL Termination
// configuration for a load balancer.
func Update(c *gophercloud.ServiceClient, lbID int, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToSSLUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(rootURL(c, lbID), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Get is the operation responsible for showing the details of the SSL
// Termination configuration for a load balancer.
func Get(c *gophercloud.ServiceClient, lbID int) GetResult {
	var res GetResult
	_, res.Err = c.Get(rootURL(c, lbID), &res.Body, nil)
	return res
}

// Delete is the operation responsible for deleting the SSL Termination
// configuration for a load balancer.
func Delete(c *gophercloud.ServiceClient, lbID int) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(rootURL(c, lbID), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// ListCerts will list all of the certificate mappings associated with a
// SSL-terminated HTTP load balancer.
func ListCerts(c *gophercloud.ServiceClient, lbID int) pagination.Pager {
	url := certURL(c, lbID)
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return CertPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateCertOptsBuilder is the interface options structs have to satisfy in
// order to be used in the AddCert operation in this package.
type CreateCertOptsBuilder interface {
	ToCertCreateMap() (map[string]interface{}, error)
}

// CreateCertOpts represents the options used when adding a new certificate mapping.
type CreateCertOpts struct {
	HostName       string
	PrivateKey     string
	Certificate    string
	IntCertificate string
}

// ToCertCreateMap will cast an CreateCertOpts struct to a map for JSON serialization.
func (opts CreateCertOpts) ToCertCreateMap() (map[string]interface{}, error) {
	cm := make(map[string]interface{})

	if opts.HostName == "" {
		return cm, errors.New("HostName is a required option")
	}
	if opts.PrivateKey == "" {
		return cm, errPrivateKey
	}
	if opts.Certificate == "" {
		return cm, errCertificate
	}

	cm["hostName"] = opts.HostName
	cm["privateKey"] = opts.PrivateKey
	cm["certificate"] = opts.Certificate

	if opts.IntCertificate != "" {
		cm["intermediateCertificate"] = opts.IntCertificate
	}

	return map[string]interface{}{"certificateMapping": cm}, nil
}

// CreateCert will add a new SSL certificate and allow an SSL-terminated HTTP
// load balancer to use it. This feature is useful because it allows multiple
// certificates to be used. The maximum number of certificates that can be
// stored per LB is 20.
func CreateCert(c *gophercloud.ServiceClient, lbID int, opts CreateCertOptsBuilder) CreateCertResult {
	var res CreateCertResult

	reqBody, err := opts.ToCertCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(certURL(c, lbID), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return res
}

// GetCert will show the details of an existing SSL certificate.
func GetCert(c *gophercloud.ServiceClient, lbID, certID int) GetCertResult {
	var res GetCertResult
	_, res.Err = c.Get(certResourceURL(c, lbID, certID), &res.Body, nil)
	return res
}

// UpdateCertOptsBuilder is the interface options structs have to satisfy in
// order to be used in the UpdateCert operation in this package.
type UpdateCertOptsBuilder interface {
	ToCertUpdateMap() (map[string]interface{}, error)
}

// UpdateCertOpts represents the options needed to update a SSL certificate.
type UpdateCertOpts struct {
	HostName       string
	PrivateKey     string
	Certificate    string
	IntCertificate string
}

// ToCertUpdateMap will cast an UpdateCertOpts struct into a map for JSON
// seralization.
func (opts UpdateCertOpts) ToCertUpdateMap() (map[string]interface{}, error) {
	cm := make(map[string]interface{})

	if opts.HostName != "" {
		cm["hostName"] = opts.HostName
	}
	if opts.PrivateKey != "" {
		cm["privateKey"] = opts.PrivateKey
	}
	if opts.Certificate != "" {
		cm["certificate"] = opts.Certificate
	}
	if opts.IntCertificate != "" {
		cm["intermediateCertificate"] = opts.IntCertificate
	}

	return map[string]interface{}{"certificateMapping": cm}, nil
}

// UpdateCert is the operation responsible for updating the details of an
// existing SSL certificate.
func UpdateCert(c *gophercloud.ServiceClient, lbID, certID int, opts UpdateCertOptsBuilder) UpdateCertResult {
	var res UpdateCertResult

	reqBody, err := opts.ToCertUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(certResourceURL(c, lbID, certID), reqBody, &res.Body, nil)
	return res
}

// DeleteCert is the operation responsible for permanently removing a SSL
// certificate.
func DeleteCert(c *gophercloud.ServiceClient, lbID, certID int) DeleteResult {
	var res DeleteResult

	_, res.Err = c.Delete(certResourceURL(c, lbID, certID), &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return res
}
