package registry

import (
	"reflect"
	"testing"

	"github.com/docker/distribution/configuration"
)

// Tests to ensure nextProtos returns the correct protocols when:
// * config.HTTP.HTTP2.Disabled is not explicitly set => [h2 http/1.1]
// * config.HTTP.HTTP2.Disabled is explicitly set to false [h2 http/1.1]
// * config.HTTP.HTTP2.Disabled is explicitly set to true [http/1.1]
func TestNextProtos(t *testing.T) {
	config := &configuration.Configuration{}
	protos := nextProtos(config)
	if !reflect.DeepEqual(protos, []string{"h2", "http/1.1"}) {
		t.Fatalf("expected protos to equal [h2 http/1.1], got %s", protos)
	}
	config.HTTP.HTTP2.Disabled = false
	protos = nextProtos(config)
	if !reflect.DeepEqual(protos, []string{"h2", "http/1.1"}) {
		t.Fatalf("expected protos to equal [h2 http/1.1], got %s", protos)
	}
	config.HTTP.HTTP2.Disabled = true
	protos = nextProtos(config)
	if !reflect.DeepEqual(protos, []string{"http/1.1"}) {
		t.Fatalf("expected protos to equal [http/1.1], got %s", protos)
	}
}
