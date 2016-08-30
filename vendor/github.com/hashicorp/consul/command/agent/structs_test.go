package agent

import (
	"testing"
	"time"

	"github.com/hashicorp/consul/consul/structs"
)

func TestAgentStructs_HealthCheck(t *testing.T) {
	def := CheckDefinition{}
	check := def.HealthCheck("node1")

	// Health checks default to critical state
	if check.Status != structs.HealthCritical {
		t.Fatalf("bad: %v", check.Status)
	}
}

func TestAgentStructs_CheckTypes(t *testing.T) {
	svc := new(ServiceDefinition)

	// Singular Check field works
	svc.Check = CheckType{
		Script:   "/foo/bar",
		Interval: 10 * time.Second,
	}

	// Returns HTTP checks
	svc.Checks = append(svc.Checks, &CheckType{
		HTTP:     "http://foo/bar",
		Interval: 10 * time.Second,
	})

	// Returns Script checks
	svc.Checks = append(svc.Checks, &CheckType{
		Script:   "/foo/bar",
		Interval: 10 * time.Second,
	})

	// Returns TTL checks
	svc.Checks = append(svc.Checks, &CheckType{
		TTL: 10 * time.Second,
	})

	// Does not return invalid checks
	svc.Checks = append(svc.Checks, &CheckType{})

	if len(svc.CheckTypes()) != 4 {
		t.Fatalf("bad: %#v", svc)
	}
}
