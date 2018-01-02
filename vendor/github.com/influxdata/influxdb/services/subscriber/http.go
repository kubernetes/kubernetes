package subscriber

import (
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"time"

	"github.com/influxdata/influxdb/client/v2"
	"github.com/influxdata/influxdb/coordinator"
)

// HTTP supports writing points over HTTP using the line protocol.
type HTTP struct {
	c client.Client
}

// NewHTTP returns a new HTTP points writer with default options.
func NewHTTP(addr string, timeout time.Duration) (*HTTP, error) {
	return NewHTTPS(addr, timeout, false, "")
}

// NewHTTPS returns a new HTTPS points writer with default options and HTTPS configured
func NewHTTPS(addr string, timeout time.Duration, unsafeSsl bool, caCerts string) (*HTTP, error) {
	tlsConfig, err := createTlsConfig(caCerts)

	conf := client.HTTPConfig{
		Addr:               addr,
		Timeout:            timeout,
		InsecureSkipVerify: unsafeSsl,
		TLSConfig:          tlsConfig,
	}

	c, err := client.NewHTTPClient(conf)
	if err != nil {
		return nil, err
	}
	return &HTTP{c: c}, nil
}

// WritePoints writes points over HTTP transport.
func (h *HTTP) WritePoints(p *coordinator.WritePointsRequest) (err error) {
	bp, _ := client.NewBatchPoints(client.BatchPointsConfig{
		Database:        p.Database,
		RetentionPolicy: p.RetentionPolicy,
	})
	for _, pt := range p.Points {
		bp.AddPoint(client.NewPointFrom(pt))
	}
	err = h.c.Write(bp)
	return
}

func createTlsConfig(caCerts string) (*tls.Config, error) {
	if caCerts == "" {
		return nil, nil
	}
	return loadCaCerts(caCerts)
}

func loadCaCerts(caCerts string) (*tls.Config, error) {
	caCert, err := ioutil.ReadFile(caCerts)
	if err != nil {
		return nil, err
	}
	caCertPool := x509.NewCertPool()
	caCertPool.AppendCertsFromPEM(caCert)

	return &tls.Config{
		RootCAs: caCertPool,
	}, nil
}
