package metrics

import (
	"reflect"
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	conf := DefaultConfig("service")
	if conf.ServiceName != "service" {
		t.Fatalf("Bad name")
	}
	if conf.HostName == "" {
		t.Fatalf("missing hostname")
	}
	if !conf.EnableHostname || !conf.EnableRuntimeMetrics {
		t.Fatalf("expect true")
	}
	if conf.EnableTypePrefix {
		t.Fatalf("expect false")
	}
	if conf.TimerGranularity != time.Millisecond {
		t.Fatalf("bad granularity")
	}
	if conf.ProfileInterval != time.Second {
		t.Fatalf("bad interval")
	}
}

func Test_GlobalMetrics_SetGauge(t *testing.T) {
	m := &MockSink{}
	globalMetrics = &Metrics{sink: m}

	k := []string{"test"}
	v := float32(42.0)
	SetGauge(k, v)

	if !reflect.DeepEqual(m.keys[0], k) {
		t.Fatalf("key not equal")
	}
	if !reflect.DeepEqual(m.vals[0], v) {
		t.Fatalf("val not equal")
	}
}

func Test_GlobalMetrics_EmitKey(t *testing.T) {
	m := &MockSink{}
	globalMetrics = &Metrics{sink: m}

	k := []string{"test"}
	v := float32(42.0)
	EmitKey(k, v)

	if !reflect.DeepEqual(m.keys[0], k) {
		t.Fatalf("key not equal")
	}
	if !reflect.DeepEqual(m.vals[0], v) {
		t.Fatalf("val not equal")
	}
}

func Test_GlobalMetrics_IncrCounter(t *testing.T) {
	m := &MockSink{}
	globalMetrics = &Metrics{sink: m}

	k := []string{"test"}
	v := float32(42.0)
	IncrCounter(k, v)

	if !reflect.DeepEqual(m.keys[0], k) {
		t.Fatalf("key not equal")
	}
	if !reflect.DeepEqual(m.vals[0], v) {
		t.Fatalf("val not equal")
	}
}

func Test_GlobalMetrics_AddSample(t *testing.T) {
	m := &MockSink{}
	globalMetrics = &Metrics{sink: m}

	k := []string{"test"}
	v := float32(42.0)
	AddSample(k, v)

	if !reflect.DeepEqual(m.keys[0], k) {
		t.Fatalf("key not equal")
	}
	if !reflect.DeepEqual(m.vals[0], v) {
		t.Fatalf("val not equal")
	}
}

func Test_GlobalMetrics_MeasureSince(t *testing.T) {
	m := &MockSink{}
	globalMetrics = &Metrics{sink: m}
	globalMetrics.TimerGranularity = time.Millisecond

	k := []string{"test"}
	now := time.Now()
	MeasureSince(k, now)

	if !reflect.DeepEqual(m.keys[0], k) {
		t.Fatalf("key not equal")
	}
	if m.vals[0] > 0.1 {
		t.Fatalf("val too large %v", m.vals[0])
	}
}
