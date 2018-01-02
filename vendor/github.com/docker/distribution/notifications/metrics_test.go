package notifications

import (
	"encoding/json"
	"expvar"
	"testing"
)

func TestMetricsExpvar(t *testing.T) {
	endpointsVar := expvar.Get("registry").(*expvar.Map).Get("notifications").(*expvar.Map).Get("endpoints")

	var v interface{}
	if err := json.Unmarshal([]byte(endpointsVar.String()), &v); err != nil {
		t.Fatalf("unexpected error unmarshaling endpoints: %v", err)
	}
	if v != nil {
		t.Fatalf("expected nil, got %#v", v)
	}

	NewEndpoint("x", "y", EndpointConfig{})

	if err := json.Unmarshal([]byte(endpointsVar.String()), &v); err != nil {
		t.Fatalf("unexpected error unmarshaling endpoints: %v", err)
	}
	if slice, ok := v.([]interface{}); !ok || len(slice) != 1 {
		t.Logf("expected one-element []interface{}, got %#v", v)
	}
}
