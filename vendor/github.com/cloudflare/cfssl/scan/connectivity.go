package scan

import (
	"bufio"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
)

// Connectivity contains scanners testing basic connectivity to the host
var Connectivity = &Family{
	Description: "Scans for basic connectivity with the host through DNS and TCP/TLS dials",
	Scanners: map[string]*Scanner{
		"DNSLookup": {
			"Host can be resolved through DNS",
			dnsLookupScan,
		},
		"CloudFlareStatus": {
			"Host is on CloudFlare",
			onCloudFlareScan,
		},
		"TCPDial": {
			"Host accepts TCP connection",
			tcpDialScan,
		},
		"TLSDial": {
			"Host can perform TLS handshake",
			tlsDialScan,
		},
	},
}

// dnsLookupScan tests that DNS resolution of the host returns at least one address
func dnsLookupScan(addr, hostname string) (grade Grade, output Output, err error) {
	addrs, err := net.LookupHost(hostname)
	if err != nil {
		return
	}

	if len(addrs) == 0 {
		err = errors.New("no addresses found for host")
	}

	grade, output = Good, addrs
	return
}

var (
	cfNets    []*net.IPNet
	cfNetsErr error
)

func initOnCloudFlareScan() ([]*net.IPNet, error) {
	// Propogate previous errors and don't attempt to re-download.
	if cfNetsErr != nil {
		return nil, cfNetsErr
	}

	// Don't re-download ranges if we already have them.
	if len(cfNets) > 0 {
		return cfNets, nil
	}

	// Download CloudFlare CIDR ranges and parse them.
	v4resp, err := Client.Get("https://www.cloudflare.com/ips-v4")
	if err != nil {
		cfNetsErr = fmt.Errorf("Couldn't download CloudFlare IPs: %v", err)
		return nil, cfNetsErr
	}
	defer v4resp.Body.Close()

	v6resp, err := Client.Get("https://www.cloudflare.com/ips-v6")
	if err != nil {
		cfNetsErr = fmt.Errorf("Couldn't download CloudFlare IPs: %v", err)
		return nil, cfNetsErr
	}
	defer v6resp.Body.Close()

	scanner := bufio.NewScanner(io.MultiReader(v4resp.Body, v6resp.Body))
	for scanner.Scan() {
		_, ipnet, err := net.ParseCIDR(scanner.Text())
		if err != nil {
			cfNetsErr = fmt.Errorf("Couldn't parse CIDR range: %v", err)
			return nil, cfNetsErr
		}
		cfNets = append(cfNets, ipnet)
	}
	if err := scanner.Err(); err != nil {
		cfNetsErr = fmt.Errorf("Couldn't read IP bodies: %v", err)
		return nil, cfNetsErr
	}

	return cfNets, nil
}

func onCloudFlareScan(addr, hostname string) (grade Grade, output Output, err error) {
	var cloudflareNets []*net.IPNet
	if cloudflareNets, err = initOnCloudFlareScan(); err != nil {
		grade = Skipped
		return
	}

	_, addrs, err := dnsLookupScan(addr, hostname)
	if err != nil {
		return
	}

	cfStatus := make(map[string]bool)
	grade = Good
	for _, addr := range addrs.([]string) {
		ip := net.ParseIP(addr)
		for _, cfNet := range cloudflareNets {
			if cfNet.Contains(ip) {
				cfStatus[addr] = true
				break
			}
		}
		if !cfStatus[addr] {
			cfStatus[addr] = false
			grade = Bad
		}
	}

	output = cfStatus
	return
}

// tcpDialScan tests that the host can be connected to through TCP.
func tcpDialScan(addr, hostname string) (grade Grade, output Output, err error) {
	conn, err := Dialer.Dial(Network, addr)
	if err != nil {
		return
	}
	conn.Close()
	grade = Good
	return
}

// tlsDialScan tests that the host can perform a TLS Handshake
// and warns if the server's certificate can't be verified.
func tlsDialScan(addr, hostname string) (grade Grade, output Output, err error) {
	var conn *tls.Conn
	config := defaultTLSConfig(hostname)

	if conn, err = tls.DialWithDialer(Dialer, Network, addr, config); err != nil {
		return
	}
	conn.Close()

	config.InsecureSkipVerify = false
	if conn, err = tls.DialWithDialer(Dialer, Network, addr, config); err != nil {
		grade = Warning
		return
	}
	conn.Close()

	grade = Good
	return
}
