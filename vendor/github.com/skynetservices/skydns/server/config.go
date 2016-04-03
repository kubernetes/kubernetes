// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/miekg/dns"
)

const (
	SCacheCapacity = 10000
	RCacheCapacity = 100000
	RCacheTtl      = 60
)

// Config provides options to the SkyDNS resolver.
type Config struct {
	// The ip:port SkyDNS should be listening on for incoming DNS requests.
	DnsAddr string `json:"dns_addr,omitempty"`
	// bind to port(s) activated by systemd. If set to true, this overrides DnsAddr.
	Systemd bool `json:"systemd,omitempty"`
	// The domain SkyDNS is authoritative for, defaults to skydns.local.
	Domain string `json:"domain,omitempty"`
	// Domain pointing to a key where service info is stored when being queried
	// for local.dns.skydns.local.
	Local string `json:"local,omitempty"`
	// The hostmaster responsible for this domain, defaults to hostmaster.<Domain>.
	Hostmaster string `json:"hostmaster,omitempty"`
	DNSSEC     string `json:"dnssec,omitempty"`
	// Round robin A/AAAA replies. Default is true.
	RoundRobin bool `json:"round_robin,omitempty"`
	// List of ip:port, seperated by commas of recursive nameservers to forward queries to.
	Nameservers []string `json:"nameservers,omitempty"`
	// Never provide a recursive service.
	NoRec       bool          `json:norec,omitempty"`
	ReadTimeout time.Duration `json:"read_timeout,omitempty"`
	// Default priority on SRV records when none is given. Defaults to 10.
	Priority uint16 `json:"priority"`
	// Default TTL, in seconds, when none is given in etcd. Defaults to 3600.
	Ttl uint32 `json:"ttl,omitempty"`
	// Minimum TTL, in seconds, for NXDOMAIN responses. Defaults to 300.
	MinTtl uint32 `json:"min_ttl,omitempty"`
	// SCache, capacity of the signature cache in signatures stored.
	SCache int `json:"scache,omitempty"`
	// RCache, capacity of response cache in resource records stored.
	RCache int `json:"rcache,omitempty"`
	// RCacheTtl, how long to cache in seconds.
	RCacheTtl int `json:"rcache_ttl,omitempty"`
	// How many labels a name should have before we allow forwarding. Default to 2.
	Ndots int `json:"ndot,omitempty"`

	// DNSSEC key material
	PubKey  *dns.DNSKEY    `json:"-"`
	KeyTag  uint16         `json:"-"`
	PrivKey dns.PrivateKey `json:"-"`

	Verbose bool `json:"-"`

	// some predefined string "constants"
	localDomain string // "local.dns." + config.Domain
	dnsDomain   string // "ns.dns". + config.Domain

	// Stub zones support. Pointer to a map that we refresh when we see
	// an update. Map contains domainname -> nameserver:port
	stub *map[string][]string
}

func SetDefaults(config *Config) error {
	if config.ReadTimeout == 0 {
		config.ReadTimeout = 2 * time.Second
	}
	if config.DnsAddr == "" {
		config.DnsAddr = "127.0.0.1:53"
	}
	if config.Domain == "" {
		config.Domain = "skydns.local."
	}
	if config.Hostmaster == "" {
		config.Hostmaster = appendDomain("hostmaster", config.Domain)
	}
	// People probably don't know that SOA's email addresses cannot
	// contain @-signs, replace them with dots
	config.Hostmaster = dns.Fqdn(strings.Replace(config.Hostmaster, "@", ".", -1))
	if config.MinTtl == 0 {
		config.MinTtl = 60
	}
	if config.Ttl == 0 {
		config.Ttl = 3600
	}
	if config.Priority == 0 {
		config.Priority = 10
	}
	if config.RCache < 0 {
		config.RCache = 0
	}
	if config.SCache < 0 {
		config.SCache = 0
	}
	if config.RCacheTtl == 0 {
		config.RCacheTtl = RCacheTtl
	}
	if config.Ndots <= 0 {
		config.Ndots = 2
	}

	if len(config.Nameservers) == 0 {
		c, err := dns.ClientConfigFromFile("/etc/resolv.conf")
		if err != nil {
			return err
		}
		for _, s := range c.Servers {
			config.Nameservers = append(config.Nameservers, net.JoinHostPort(s, c.Port))
		}
	}
	config.Domain = dns.Fqdn(strings.ToLower(config.Domain))
	if config.DNSSEC != "" {
		// For some reason the + are replaces by spaces in etcd. Re-replace them
		keyfile := strings.Replace(config.DNSSEC, " ", "+", -1)
		k, p, err := ParseKeyFile(keyfile)
		if err != nil {
			return err
		}
		if k.Header().Name != dns.Fqdn(config.Domain) {
			return fmt.Errorf("ownername of DNSKEY must match SkyDNS domain")
		}
		k.Header().Ttl = config.Ttl
		config.PubKey = k
		config.KeyTag = k.KeyTag()
		config.PrivKey = p
	}
	config.localDomain = appendDomain("local.dns", config.Domain)
	config.dnsDomain = appendDomain("ns.dns", config.Domain)
	stubmap := make(map[string][]string)
	config.stub = &stubmap
	return nil
}

func appendDomain(s1, s2 string) string {
	if len(s2) > 0 && s2[0] == '.' {
		return s1 + s2
	}
	return s1 + "." + s2
}
