package rootcerts

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
)

// Config determines where LoadCACerts will load certificates from. When both
// CAFile and CAPath are blank, this library's functions will either load
// system roots explicitly and return them, or set the CertPool to nil to allow
// Go's standard library to load system certs.
type Config struct {
	// CAFile is a path to a PEM-encoded certificate file or bundle. Takes
	// precedence over CAPath.
	CAFile string

	// CAPath is a path to a directory populated with PEM-encoded certificates.
	CAPath string
}

// ConfigureTLS sets up the RootCAs on the provided tls.Config based on the
// Config specified.
func ConfigureTLS(t *tls.Config, c *Config) error {
	if t == nil {
		return nil
	}
	pool, err := LoadCACerts(c)
	if err != nil {
		return err
	}
	t.RootCAs = pool
	return nil
}

// LoadCACerts loads a CertPool based on the Config specified.
func LoadCACerts(c *Config) (*x509.CertPool, error) {
	if c == nil {
		c = &Config{}
	}
	if c.CAFile != "" {
		return LoadCAFile(c.CAFile)
	}
	if c.CAPath != "" {
		return LoadCAPath(c.CAPath)
	}

	return LoadSystemCAs()
}

// LoadCAFile loads a single PEM-encoded file from the path specified.
func LoadCAFile(caFile string) (*x509.CertPool, error) {
	pool := x509.NewCertPool()

	pem, err := ioutil.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("Error loading CA File: %s", err)
	}

	ok := pool.AppendCertsFromPEM(pem)
	if !ok {
		return nil, fmt.Errorf("Error loading CA File: Couldn't parse PEM in: %s", caFile)
	}

	return pool, nil
}

// LoadCAPath walks the provided path and loads all certificates encounted into
// a pool.
func LoadCAPath(caPath string) (*x509.CertPool, error) {
	pool := x509.NewCertPool()
	walkFn := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		pem, err := ioutil.ReadFile(path)
		if err != nil {
			return fmt.Errorf("Error loading file from CAPath: %s", err)
		}

		ok := pool.AppendCertsFromPEM(pem)
		if !ok {
			return fmt.Errorf("Error loading CA Path: Couldn't parse PEM in: %s", path)
		}

		return nil
	}

	err := filepath.Walk(caPath, walkFn)
	if err != nil {
		return nil, err
	}

	return pool, nil
}
