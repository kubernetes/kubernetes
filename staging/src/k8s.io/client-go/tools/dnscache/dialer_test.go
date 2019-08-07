package dnscache

import (
	"context"
	"fmt"
	"net"
	"reflect"
	"testing"
	"time"
)

type testConn struct {
	host string
	net.Conn
}
type testDialer struct {
	count int64
}

func (d *testDialer) Dial(network, address string) (net.Conn, error) {
	return d.DialContext(context.Background(), network, address)
}

func (d *testDialer) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	d.count++
	return &testConn{
		host: address,
	}, nil
}

type testResolver struct {
	count int64
	delay time.Duration
	ips   map[string][]net.IPAddr
}

func (r *testResolver) LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error) {
	if r.delay > 0 {
		time.Sleep(r.delay)
	}
	r.count++
	addrs, ok := r.ips[host]
	if !ok {
		return nil, fmt.Errorf("not found")
	}
	return addrs, nil
}

func TestDialerWithForceRefreshTimes(t *testing.T) {
	td := &testDialer{}
	tr := &testResolver{
		ips: map[string][]net.IPAddr{
			"apiserver.kube-system.svc": []net.IPAddr{
				{IP: net.IPv4(1, 2, 3, 4)},
			},
		},
	}

	config := &DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		ForceRefreshTimes: 5,
	}
	d, err := NewDialer(config)
	if err != nil {
		t.Fatalf("Can't create dialer because %v", err)
	}

	times := config.ForceRefreshTimes * 10

	for i := int64(0); i < times; i++ {
		conn, err := d.DialContext(context.Background(), "tcp", "apiserver.kube-system.svc:80")
		if err != nil {
			t.Fatalf("Can't create conn because %v", err)
		}

		tc, ok := conn.(*testConn)
		if !ok {
			t.Fatalf("Can't convert interface %s to testConn", reflect.TypeOf(conn).String())
		}
		if tc.host != "1.2.3.4:80" {
			t.Fatalf("Receive wrong host %s with ip %s", "apiserver.kube-system.svc", tc.host)
		}
		if tr.count != (i/config.ForceRefreshTimes + 1) {
			t.Fatalf("The count of dns query %d is not equal to expected %d", tr.count, i/config.ForceRefreshTimes+1)
		}
		if td.count != i+1 {
			t.Fatalf("The count of created conn %d is not equal to expected %d", td.count, i+1)
		}
	}

	s, ok := d.(Stats)
	if !ok {
		t.Fatalf("Can't convert dialer %s to stats", reflect.TypeOf(d).String())
	}
	stats := s.Stats()
	expected := DialerStats{
		TotalConn:          50,
		CacheMiss:          0,
		CacheHit:           50,
		DNSQuery:           10,
		SuccessfulDNSQuery: 10,
	}
	if !reflect.DeepEqual(stats, expected) {
		t.Fatalf("Dialer stats %+v is not equal to %+v", stats, expected)
	}
}

func TestDialerWithCacheHit(t *testing.T) {
	td := &testDialer{}
	tr := &testResolver{
		ips: map[string][]net.IPAddr{
			"apiserver.kube-system.svc": []net.IPAddr{
				{IP: net.IPv4(1, 2, 3, 4)},
			},
		},
	}

	config := &DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		MinCacheDuration:  time.Millisecond * 200,
		MaxCacheDuration:  time.Second,
		ForceRefreshTimes: 10000,
	}
	d, err := NewDialer(config)
	if err != nil {
		t.Fatalf("Can't create dialer because %v", err)
	}

	times := int64(10)

	for i := int64(0); i < times; i++ {
		conn, err := d.DialContext(context.Background(), "tcp", "apiserver.kube-system.svc:80")
		if err != nil {
			t.Fatalf("Can't create conn because %v", err)
		}

		tc, ok := conn.(*testConn)
		if !ok {
			t.Fatalf("Can't convert interface %s to testConn", reflect.TypeOf(conn).String())
		}
		if tc.host != "1.2.3.4:80" {
			t.Fatalf("Receive wrong host %s with ip %s", "apiserver.kube-system.svc", tc.host)
		}
		if tr.count != 1 {
			t.Fatalf("The count of dns query %d is not equal to expected %d", tr.count, 1)
		}
		if td.count != i+1 {
			t.Fatalf("The count of created conn %d is not equal to expected %d", td.count, i+1)
		}
	}
	s, ok := d.(Stats)
	if !ok {
		t.Fatalf("Can't convert dialer %s to stats", reflect.TypeOf(d).String())
	}
	stats := s.Stats()
	expected := DialerStats{
		TotalConn:          10,
		CacheMiss:          0,
		CacheHit:           10,
		DNSQuery:           1,
		SuccessfulDNSQuery: 1,
	}
	if !reflect.DeepEqual(stats, expected) {
		t.Fatalf("Dialer stats %+v is not equal to %+v", stats, expected)
	}
}

func TestDialerWithCacheMiss(t *testing.T) {
	td := &testDialer{}
	tr := &testResolver{
		ips: map[string][]net.IPAddr{
			"apiserver.kube-system.svc": []net.IPAddr{
				{IP: net.IPv4(1, 2, 3, 4)},
			},
		},
	}

	config := &DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		MinCacheDuration:  time.Millisecond * 200,
		MaxCacheDuration:  time.Millisecond * 300,
		ForceRefreshTimes: 10000,
	}
	d, err := NewDialer(config)
	if err != nil {
		t.Fatalf("Can't create dialer because %v", err)
	}

	times := int64(10)

	for i := int64(0); i < times; i++ {
		conn, err := d.DialContext(context.Background(), "tcp", "apiserver.kube-system.svc:80")
		if err != nil {
			t.Fatalf("Can't create conn because %v", err)
		}

		tc, ok := conn.(*testConn)
		if !ok {
			t.Fatalf("Can't convert interface %s to testConn", reflect.TypeOf(conn).String())
		}
		if tc.host != "1.2.3.4:80" {
			t.Fatalf("Receive wrong host %s with ip %s", "apiserver.kube-system.svc", tc.host)
		}
		if tr.count != i+1 {
			t.Fatalf("The count of dns query %d is not equal to expected %d", tr.count, i+1)
		}
		if td.count != i+1 {
			t.Fatalf("The count of created conn %d is not equal to expected %d", td.count, i+1)
		}
		time.Sleep(config.MaxCacheDuration)
	}
	s, ok := d.(Stats)
	if !ok {
		t.Fatalf("Can't convert dialer %s to stats", reflect.TypeOf(d).String())
	}
	stats := s.Stats()
	expected := DialerStats{
		TotalConn:          10,
		CacheMiss:          0,
		CacheHit:           10,
		DNSQuery:           10,
		SuccessfulDNSQuery: 10,
	}
	if !reflect.DeepEqual(stats, expected) {
		t.Fatalf("Dialer stats %+v is not equal to %+v", stats, expected)
	}
}

func TestDialerWithExceptions(t *testing.T) {
	td := &testDialer{}
	tr := &testResolver{
		ips: map[string][]net.IPAddr{
			"apiserver.kube-system.svc": []net.IPAddr{
				{IP: net.IPv4(1, 2, 3, 4)},
			},
		},
	}

	_, err := NewDialer(&DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		MinCacheDuration:  time.Millisecond * 400,
		MaxCacheDuration:  time.Millisecond * 300,
		ForceRefreshTimes: 10000,
	})
	if err == nil || err.Error() != "min cache duration(400ms) should less than or equal to max cache duration(300ms)" {
		t.Fatalf("Create error is not expected: %v", err)
	}

	config := &DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		MinCacheDuration:  time.Millisecond * 200,
		MaxCacheDuration:  time.Millisecond * 300,
		ForceRefreshTimes: 10000,
	}
	d, err := NewDialer(config)
	if err != nil {
		t.Fatalf("Can't create dialer because %v", err)
	}

	_, err = d.Dial("tcp4", "apiserver.kube-system.svc")
	if err == nil || err.Error() != "address apiserver.kube-system.svc: missing port in address" {
		t.Fatalf("Dial error is not expected: %v", err)
	}
	s, ok := d.(Stats)
	if !ok {
		t.Fatalf("Can't convert dialer %s to stats", reflect.TypeOf(d).String())
	}
	stats := s.Stats()
	expected := DialerStats{
		TotalConn:          1,
		CacheMiss:          0,
		CacheHit:           0,
		DNSQuery:           0,
		SuccessfulDNSQuery: 0,
	}
	if !reflect.DeepEqual(stats, expected) {
		t.Fatalf("Dialer stats %+v is not equal to %+v", stats, expected)
	}
}

func TestDialerWithMultipleDomains(t *testing.T) {
	td := &testDialer{}
	domains := []string{
		"apiserver.kube-system.svc1",
		"apiserver.kube-system.svc2",
		"apiserver.kube-system.svc3",
		"apiserver.kube-system.svc4",
		"apiserver.kube-system.svc5",
	}
	dips := map[string][]net.IPAddr{
		domains[0]: []net.IPAddr{
			{IP: net.IPv4(1, 1, 3, 4)},
			{IP: net.IPv4(1, 2, 3, 4)},
			{IP: net.IPv4(1, 3, 3, 4)},
		},
		domains[1]: []net.IPAddr{
			{IP: net.IPv4(2, 1, 3, 4)},
		},
		domains[2]: []net.IPAddr{
			{IP: net.IPv4(3, 1, 3, 4)},
			{IP: net.IPv4(3, 2, 3, 4)},
		},
		// This domain has no IPs, always fail in querying.
		domains[3]: []net.IPAddr{},
		domains[4]: []net.IPAddr{
			{IP: net.IPv4(5, 1, 3, 4)},
			{IP: net.IPv4(5, 2, 3, 4)},
			{IP: net.IPv4(5, 3, 3, 4)},
			{IP: net.IPv4(5, 4, 3, 4)},
			{IP: net.IPv4(5, 5, 3, 4)},
		},
	}
	tr := &testResolver{
		ips: dips,
	}

	config := &DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		MinCacheDuration:  time.Millisecond * 300,
		MaxCacheDuration:  time.Millisecond * 300,
		ForceRefreshTimes: 5,
	}
	d, err := NewDialer(config)
	if err != nil {
		t.Fatalf("Can't create dialer because %v", err)
	}

	times := int64(1000)

	dnsCount := int64(0)
	connCount := int64(0)
	for i := int64(0); i < times; i++ {
		for _, domain := range domains {
			ips := dips[domain]
			conn, err := d.DialContext(context.Background(), "tcp4", domain+":443")
			if domain == "apiserver.kube-system.svc4" {
				if err != nil && err.Error() == "no dns records for host apiserver.kube-system.svc4" {
					dnsCount++
					continue
				}
				t.Fatalf("Should not create conn because apiserver.kube-system.svc4 has no records")
			}
			if err != nil {
				t.Fatalf("Can't create conn because %v", err)
			}
			tc, ok := conn.(*testConn)
			if !ok {
				t.Fatalf("Can't convert interface %s to testConn", reflect.TypeOf(conn).String())
			}
			if len(ips) > 0 {
				index := i % config.ForceRefreshTimes % int64(len(ips))
				if tc.host != ips[index].String()+":443" {
					t.Fatalf("Receive wrong host %s(%s) with ip %s", domain, ips[index].String(), tc.host)
				}
				if i%config.ForceRefreshTimes == 0 {
					dnsCount++
				}
			} else {
				if tc.host != domain+":443" {
					t.Fatalf("Receive wrong host %s with domain %s", domain, tc.host)
				}
				dnsCount++
			}

			if tr.count != dnsCount {
				t.Fatalf("The count of dns query %d is not equal to expected %d", tr.count, dnsCount)
			}
			connCount++
			if td.count != connCount {
				t.Fatalf("The count of created conn %d is not equal to expected %d", td.count, connCount)
			}
		}
	}
	s, ok := d.(Stats)
	if !ok {
		t.Fatalf("Can't convert dialer %s to stats", reflect.TypeOf(d).String())
	}
	stats := s.Stats()
	expected := DialerStats{
		TotalConn:          5000,
		CacheMiss:          1000,
		CacheHit:           4000,
		DNSQuery:           1800,
		SuccessfulDNSQuery: 800,
	}
	if !reflect.DeepEqual(stats, expected) {
		t.Fatalf("Dialer stats %+v is not equal to %+v", stats, expected)
	}
}

func TestDialerWithParallelAccess(t *testing.T) {
	td := &testDialer{}
	domains := []string{
		"apiserver.kube-system.svc",
	}
	dips := map[string][]net.IPAddr{
		domains[0]: []net.IPAddr{
			{IP: net.IPv4(1, 1, 3, 4)},
			{IP: net.IPv4(1, 2, 3, 4)},
			{IP: net.IPv4(1, 3, 3, 4)},
		},
	}
	tr := &testResolver{
		ips:   dips,
		delay: time.Millisecond * 900,
	}

	config := &DialerConfig{
		Dialer:            td,
		Resolver:          tr,
		MinCacheDuration:  time.Millisecond * 300,
		MaxCacheDuration:  time.Millisecond * 300,
		ForceRefreshTimes: 10,
	}
	d, err := NewDialer(config)
	if err != nil {
		t.Fatalf("Can't create dialer because %v", err)
	}

	times := int64(100)

	for i := int64(0); i < times; i++ {
		go func(i int64) {
			ctx, _ := context.WithTimeout(context.Background(), time.Second)
			conn, err := d.DialContext(ctx, "tcp", "apiserver.kube-system.svc:80")
			if err != nil {
				t.Fatalf("Can't create conn because %v", err)
			}
			tc, ok := conn.(*testConn)
			if !ok {
				t.Fatalf("Can't convert interface %s to testConn", reflect.TypeOf(conn).String())
			}
			if tc.host != "1.1.3.4:80" && tc.host != "1.2.3.4:80" && tc.host != "1.3.3.4:80" {
				t.Fatalf("Receive wrong host %s with ip %s", "apiserver.kube-system.svc", tc.host)
			}
		}(i)
	}

	time.Sleep(time.Second)

	if tr.count != 1 {
		t.Fatalf("The count of dns query %d is not equal to expected %d", tr.count, 1)
	}
	if td.count != 100 {
		t.Fatalf("The count of created conn %d is not equal to expected %d", td.count, 100)
	}

	s, ok := d.(Stats)
	if !ok {
		t.Fatalf("Can't convert dialer %s to stats", reflect.TypeOf(d).String())
	}
	stats := s.Stats()
	expected := DialerStats{
		TotalConn:          100,
		CacheMiss:          0,
		CacheHit:           100,
		DNSQuery:           1,
		SuccessfulDNSQuery: 1,
	}
	if !reflect.DeepEqual(stats, expected) {
		t.Fatalf("Dialer stats %+v is not equal to %+v", stats, expected)
	}
}
