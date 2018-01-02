package discovery

import (
	"fmt"
	"testing"
	"time"
)

func TestDiscoveryOpts(t *testing.T) {
	clusterOpts := map[string]string{"discovery.heartbeat": "10", "discovery.ttl": "5"}
	heartbeat, ttl, err := discoveryOpts(clusterOpts)
	if err == nil {
		t.Fatal("discovery.ttl < discovery.heartbeat must fail")
	}

	clusterOpts = map[string]string{"discovery.heartbeat": "10", "discovery.ttl": "10"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err == nil {
		t.Fatal("discovery.ttl == discovery.heartbeat must fail")
	}

	clusterOpts = map[string]string{"discovery.heartbeat": "-10", "discovery.ttl": "10"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err == nil {
		t.Fatal("negative discovery.heartbeat must fail")
	}

	clusterOpts = map[string]string{"discovery.heartbeat": "10", "discovery.ttl": "-10"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err == nil {
		t.Fatal("negative discovery.ttl must fail")
	}

	clusterOpts = map[string]string{"discovery.heartbeat": "invalid"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err == nil {
		t.Fatal("invalid discovery.heartbeat must fail")
	}

	clusterOpts = map[string]string{"discovery.ttl": "invalid"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err == nil {
		t.Fatal("invalid discovery.ttl must fail")
	}

	clusterOpts = map[string]string{"discovery.heartbeat": "10", "discovery.ttl": "20"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err != nil {
		t.Fatal(err)
	}

	if heartbeat != 10*time.Second {
		t.Fatalf("Heartbeat - Expected : %v, Actual : %v", 10*time.Second, heartbeat)
	}

	if ttl != 20*time.Second {
		t.Fatalf("TTL - Expected : %v, Actual : %v", 20*time.Second, ttl)
	}

	clusterOpts = map[string]string{"discovery.heartbeat": "10"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err != nil {
		t.Fatal(err)
	}

	if heartbeat != 10*time.Second {
		t.Fatalf("Heartbeat - Expected : %v, Actual : %v", 10*time.Second, heartbeat)
	}

	expected := 10 * defaultDiscoveryTTLFactor * time.Second
	if ttl != expected {
		t.Fatalf("TTL - Expected : %v, Actual : %v", expected, ttl)
	}

	clusterOpts = map[string]string{"discovery.ttl": "30"}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err != nil {
		t.Fatal(err)
	}

	if ttl != 30*time.Second {
		t.Fatalf("TTL - Expected : %v, Actual : %v", 30*time.Second, ttl)
	}

	expected = 30 * time.Second / defaultDiscoveryTTLFactor
	if heartbeat != expected {
		t.Fatalf("Heartbeat - Expected : %v, Actual : %v", expected, heartbeat)
	}

	discoveryTTL := fmt.Sprintf("%d", defaultDiscoveryTTLFactor-1)
	clusterOpts = map[string]string{"discovery.ttl": discoveryTTL}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err == nil && heartbeat == 0 {
		t.Fatal("discovery.heartbeat must be positive")
	}

	clusterOpts = map[string]string{}
	heartbeat, ttl, err = discoveryOpts(clusterOpts)
	if err != nil {
		t.Fatal(err)
	}

	if heartbeat != defaultDiscoveryHeartbeat {
		t.Fatalf("Heartbeat - Expected : %v, Actual : %v", defaultDiscoveryHeartbeat, heartbeat)
	}

	expected = defaultDiscoveryHeartbeat * defaultDiscoveryTTLFactor
	if ttl != expected {
		t.Fatalf("TTL - Expected : %v, Actual : %v", expected, ttl)
	}
}
