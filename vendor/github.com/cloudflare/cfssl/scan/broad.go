package scan

import (
	"crypto/tls"
	"crypto/x509"
	"net"
	"sync"
	"time"

	"github.com/cloudflare/cfssl/bundler"
)

// Broad contains scanners for large swaths of TLS hosts on the internet.
var Broad = &Family{
	Description: "Large scale scans of TLS hosts",
	Scanners: map[string]*Scanner{
		"IntermediateCAs": {
			"Scans a CIDR IP range for unknown Intermediate CAs",
			intermediateCAScan,
		},
	},
}

func incrementBytes(bytes []byte) {
	lsb := len(bytes) - 1
	bytes[lsb]++
	if bytes[lsb] == 0 {
		incrementBytes(bytes[:lsb])
	}
}

var (
	caBundleFile  = "/etc/cfssl/ca-bundle.crt"
	intBundleFile = "/etc/cfssl/int-bundle.crt"
	numWorkers    = 32
	timeout       = time.Second
)

// intermediateCAScan scans for new intermediate CAs not in the trust store.
func intermediateCAScan(addr, hostname string) (grade Grade, output Output, err error) {
	cidr, port, _ := net.SplitHostPort(addr)
	_, ipnet, err := net.ParseCIDR(cidr)
	if err != nil {
		return Skipped, nil, nil
	}
	b, err := bundler.NewBundler(caBundleFile, intBundleFile)
	if err != nil {
		return
	}
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	dialer := &net.Dialer{Timeout: timeout}
	config := &tls.Config{InsecureSkipVerify: true}
	addrs := make(chan string)
	chains := make(chan []*x509.Certificate, numWorkers)
	go func() {
		for chain := range chains {
			b.Bundle(chain, nil, bundler.Force)
		}
	}()
	for i := 0; i < numWorkers; i++ {
		go func() {
			for addr := range addrs {
				conn, err := tls.DialWithDialer(dialer, Network, addr, config)
				if err != nil {
					continue
				}
				conn.Close()
				if conn.ConnectionState().HandshakeComplete {
					chains <- conn.ConnectionState().PeerCertificates
				}
			}
			wg.Done()
		}()
	}
	for ip := ipnet.IP.To16(); ipnet.Contains(ip); incrementBytes(ip) {
		addrs <- net.JoinHostPort(ip.String(), port)
	}
	close(addrs)
	wg.Wait()
	close(chains)
	grade = Good
	return
}
