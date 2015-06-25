// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/miekg/dns"

	backendetcd "github.com/skynetservices/skydns/backends/etcd"
	"github.com/skynetservices/skydns/msg"
	"github.com/skynetservices/skydns/server"
	"github.com/skynetservices/skydns/stats"
)

var (
	tlskey     = ""
	tlspem     = ""
	cacert     = ""
	config     = &server.Config{ReadTimeout: 0, Domain: "", DnsAddr: "", DNSSEC: ""}
	nameserver = ""
	machine    = ""
	discover   = false
	stub       = false
)

func env(key, def string) string {
	if x := os.Getenv(key); x != "" {
		return x
	}
	return def
}

func init() {
	flag.StringVar(&config.Domain, "domain", env("SKYDNS_DOMAIN", "skydns.local."), "domain to anchor requests to (SKYDNS_DOMAIN)")
	flag.StringVar(&config.DnsAddr, "addr", env("SKYDNS_ADDR", "127.0.0.1:53"), "ip:port to bind to (SKYDNS_ADDR)")
	flag.StringVar(&nameserver, "nameservers", env("SKYDNS_NAMESERVERS", ""), "nameserver address(es) to forward (non-local) queries to e.g. 8.8.8.8:53,8.8.4.4:53")
	flag.BoolVar(&config.NoRec, "no-rec", false, "do not provide a recursive service")
	flag.StringVar(&machine, "machines", env("ETCD_MACHINES", ""), "machine address(es) running etcd")
	flag.StringVar(&config.DNSSEC, "dnssec", "", "basename of DNSSEC key file e.q. Kskydns.local.+005+38250")
	flag.StringVar(&config.Local, "local", "", "optional unique value for this skydns instance")
	flag.StringVar(&tlskey, "tls-key", env("ETCD_TLSKEY", ""), "TLS Private Key path")
	flag.StringVar(&tlspem, "tls-pem", env("ETCD_TLSPEM", ""), "X509 Certificate")
	flag.StringVar(&cacert, "ca-cert", env("ETCD_CACERT", ""), "CA Certificate")
	flag.DurationVar(&config.ReadTimeout, "rtimeout", 2*time.Second, "read timeout")
	flag.BoolVar(&config.RoundRobin, "round-robin", true, "round robin A/AAAA replies")
	flag.BoolVar(&discover, "discover", false, "discover new machines by watching /v2/_etcd/machines")
	flag.BoolVar(&stub, "stubzones", false, "support stub zones")
	flag.BoolVar(&config.Verbose, "verbose", false, "log queries")
	flag.BoolVar(&config.Systemd, "systemd", false, "bind to socket(s) activated by systemd (ignore -addr)")

	// TTl
	// Minttl
	flag.StringVar(&config.Hostmaster, "hostmaster", "hostmaster@skydns.local.", "hostmaster email address to use")
	flag.IntVar(&config.SCache, "scache", server.SCacheCapacity, "capacity of the signature cache")
	flag.IntVar(&config.RCache, "rcache", 0, "capacity of the response cache") // default to 0 for now
	flag.IntVar(&config.RCacheTtl, "rcache-ttl", server.RCacheTtl, "TTL of the response cache")
}

func main() {
	flag.Parse()
	machines := strings.Split(machine, ",")
	client := newClient(machines, tlspem, tlskey, cacert)

	if nameserver != "" {
		for _, hostPort := range strings.Split(nameserver, ",") {
			if err := validateHostPort(hostPort); err != nil {
				log.Fatalf("skydns: nameserver is invalid: %s", err)
			}
			config.Nameservers = append(config.Nameservers, hostPort)
		}
	}
	if err := validateHostPort(config.DnsAddr); err != nil {
		log.Fatalf("skydns: addr is invalid: %s", err)
	}

	if err := loadConfig(client, config); err != nil {
		log.Fatalf("skydns: %s", err)
	}
	server.SetDefaults(config)

	if config.Local != "" {
		config.Local = dns.Fqdn(config.Local)
	}

	backend := backendetcd.NewBackend(client, &backendetcd.Config{
		Ttl:      config.Ttl,
		Priority: config.Priority,
	})
	s := server.New(backend, config)

	if discover {
		go func() {
			recv := make(chan *etcd.Response)
			go client.Watch("/_etcd/machines/", 0, true, recv, nil)
			duration := 1 * time.Second
			for {
				select {
				case n := <-recv:
					if n != nil {
						if client := updateClient(n, tlskey, tlspem, cacert); client != nil {
							backend.UpdateClient(client)
						}
						duration = 1 * time.Second // reset
					} else {
						// we can see an n == nil, probably when we can't connect to etcd.
						log.Printf("skydns: etcd machine cluster update failed, sleeping %s + ~3s", duration)
						time.Sleep(duration + (time.Duration(rand.Float32() * 3e9))) // Add some random.
						duration *= 2
						if duration > 32*time.Second {
							duration = 32 * time.Second
						}
					}
				}
			}
		}()
	}

	if stub {
		s.UpdateStubZones()
		go func() {
			recv := make(chan *etcd.Response)
			go client.Watch(msg.Path(config.Domain)+"/dns/stub/", 0, true, recv, nil)
			duration := 1 * time.Second
			for {
				select {
				case n := <-recv:
					if n != nil {
						s.UpdateStubZones()
						log.Printf("skydns: stubzone update")
						duration = 1 * time.Second // reset
					} else {
						// we can see an n == nil, probably when we can't connect to etcd.
						log.Printf("skydns: stubzone update failed, sleeping %s + ~3s", duration)
						time.Sleep(duration + (time.Duration(rand.Float32() * 3e9))) // Add some random.
						duration *= 2
						if duration > 32*time.Second {
							duration = 32 * time.Second
						}
					}
				}
			}
		}()
	}

	stats.Collect()  // Graphite
	server.Metrics() // Prometheus

	if err := s.Run(); err != nil {
		log.Fatalf("skydns: %s", err)
	}
}

func loadConfig(client *etcd.Client, config *server.Config) error {
	// Override what isn't set yet from the command line.
	n, err := client.Get("/skydns/config", false, false)
	if err != nil {
		log.Printf("skydns: falling back to default configuration, could not read from etcd: %s", err)
		return nil
	}
	if err := json.Unmarshal([]byte(n.Node.Value), config); err != nil {
		return fmt.Errorf("failed to unmarshal config: %s", err.Error())
	}
	return nil
}

func validateHostPort(hostPort string) error {
	host, port, err := net.SplitHostPort(hostPort)
	if err != nil {
		return err
	}
	if ip := net.ParseIP(host); ip == nil {
		return fmt.Errorf("bad IP address: %s", host)
	}

	if p, _ := strconv.Atoi(port); p < 1 || p > 65535 {
		return fmt.Errorf("bad port number %s", port)
	}
	return nil
}

func newClient(machines []string, tlsCert, tlsKey, tlsCACert string) (client *etcd.Client) {
	// set default if not specified in env
	if len(machines) == 1 && machines[0] == "" {
		machines[0] = "http://127.0.0.1:4001"
	}
	if strings.HasPrefix(machines[0], "https://") {
		var err error
		// TODO(miek): machines is local, the rest is global, ugly.
		if client, err = etcd.NewTLSClient(machines, tlsCert, tlsKey, tlsCACert); err != nil {
			// TODO(miek): would be nice if this wasn't a fatal error
			log.Fatalf("skydns: failure to connect: %s", err)
		}
		return client
	}
	return etcd.NewClient(machines)
}

// updateClient updates the client with the machines found in v2/_etcd/machines.
func updateClient(resp *etcd.Response, tlsCert, tlsKey, tlsCACert string) (client *etcd.Client) {
	machines := make([]string, 0, 3)
	for _, m := range resp.Node.Nodes {
		u, e := url.Parse(m.Value)
		if e != nil {
			continue
		}
		// etcd=bla&raft=bliep
		// TODO(miek): surely there is a better way to do this
		ms := strings.Split(u.String(), "&")
		if len(ms) == 0 {
			continue
		}
		if len(ms[0]) < 5 {
			continue
		}
		machines = append(machines, ms[0][5:])
	}
	// When CoreOS (and thus etcd) crash this call back seems to also trigger, but we
	// don't have any machines. Don't trigger this update then, as a) it
	// crashes SkyDNS and b) potentially leaves with no machines to connect to.
	// Keep the old ones and hope they still work and wait for another update
	// in the future.
	if len(machines) > 0 {
		log.Printf("skydns: setting new etcd cluster to %v", machines)
		return newClient(machines, tlsCert, tlsKey, tlsCACert)
	}
	return nil
}
