// Copyright (c) 2014 The SkyDNS Authors. All rights reserved.
// Use of this source code is governed by The MIT License (MIT) that can be
// found in the LICENSE file.

package server

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"strconv"
	"testing"

	"github.com/miekg/dns"
)

var metricsDone = false

func newMetricServer(t *testing.T) *server {
	s := newTestServer(t, false)

	// There is no graceful way (yet) in the http package to
	// shutdown a http server (if started with http.ListenAndServe)
	// so once this is running. It is running until we shut the
	// entire test
	prometheusPort = "12300"
	prometheusSubsystem = "test"
	prometheusNamespace = "test"

	Metrics()
	metricsDone = true

	return s
}

func query(n string, t uint16) {
	m := new(dns.Msg)
	m.SetQuestion(n, t)
	dns.Exchange(m, "127.0.0.1:"+StrPort)
}

func scrape(t *testing.T, key string) int {
	resp, err := http.Get("http://localhost:12300/metrics")
	if err != nil {
		t.Logf("could not get metrics")
		return -1
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return -1
	}

	// Find value for key.
	n := bytes.Index(body, []byte(key))
	if n == -1 {
		return -1
	}

	i := n
	for i < len(body) {
		if body[i] == '\n' {
			break
		}
		if body[i] == ' ' {
			n = i + 1
		}
		i++
	}
	value, err := strconv.Atoi(string(body[n:i]))
	if err != nil {
		t.Fatal("failed to get value")
	}
	return value
}

// This test needs to be first, see comment in NewMetricServer.
func TestMetricsOff(t *testing.T) {
	s := newTestServer(t, false)
	defer s.Stop()

	v := scrape(t, "test_dns_request_count{type=\"udp\"}")
	if v != -1 {
		t.Fatalf("expecting -1, got %d", v)
	}
}

func TestMetricRequests(t *testing.T) {
	s := newMetricServer(t)
	defer s.Stop()

	query("miek.nl.", dns.TypeMX)
	v := scrape(t, "test_dns_request_count{type=\"udp\"}")
	if v != 1 {
		t.Fatalf("expecting %d, got %d", 1, v)
	}
	v = scrape(t, "test_dns_request_count{type=\"tcp\"}")
	if v != -1 {	// if not hit, is does not show up in the metrics page.
		t.Fatalf("expecting %d, got %d for", -1, v)
	}

	// external requests are not counted in the nxdomain/nodata metrics
	query("aaaaaa.miek.nl.", dns.TypeSRV)
	v = scrape(t, "test_dns_request_count{type=\"udp\"}")
	if v != 2 {
		t.Fatalf("expecting %d, got %d", 2, v)
	}
}
