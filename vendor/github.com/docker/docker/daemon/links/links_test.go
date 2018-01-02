package links

import (
	"fmt"
	"strings"
	"testing"

	"github.com/docker/go-connections/nat"
)

// Just to make life easier
func newPortNoError(proto, port string) nat.Port {
	p, _ := nat.NewPort(proto, port)
	return p
}

func TestLinkNaming(t *testing.T) {
	ports := make(nat.PortSet)
	ports[newPortNoError("tcp", "6379")] = struct{}{}

	link := NewLink("172.0.17.3", "172.0.17.2", "/db/docker-1", nil, ports)

	rawEnv := link.ToEnv()
	env := make(map[string]string, len(rawEnv))
	for _, e := range rawEnv {
		parts := strings.Split(e, "=")
		if len(parts) != 2 {
			t.FailNow()
		}
		env[parts[0]] = parts[1]
	}

	value, ok := env["DOCKER_1_PORT"]

	if !ok {
		t.Fatal("DOCKER_1_PORT not found in env")
	}

	if value != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected 172.0.17.2:6379, got %s", env["DOCKER_1_PORT"])
	}
}

func TestLinkNew(t *testing.T) {
	ports := make(nat.PortSet)
	ports[newPortNoError("tcp", "6379")] = struct{}{}

	link := NewLink("172.0.17.3", "172.0.17.2", "/db/docker", nil, ports)

	if link.Name != "/db/docker" {
		t.Fail()
	}
	if link.ParentIP != "172.0.17.3" {
		t.Fail()
	}
	if link.ChildIP != "172.0.17.2" {
		t.Fail()
	}
	for _, p := range link.Ports {
		if p != newPortNoError("tcp", "6379") {
			t.Fail()
		}
	}
}

func TestLinkEnv(t *testing.T) {
	ports := make(nat.PortSet)
	ports[newPortNoError("tcp", "6379")] = struct{}{}

	link := NewLink("172.0.17.3", "172.0.17.2", "/db/docker", []string{"PASSWORD=gordon"}, ports)

	rawEnv := link.ToEnv()
	env := make(map[string]string, len(rawEnv))
	for _, e := range rawEnv {
		parts := strings.Split(e, "=")
		if len(parts) != 2 {
			t.FailNow()
		}
		env[parts[0]] = parts[1]
	}
	if env["DOCKER_PORT"] != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected 172.0.17.2:6379, got %s", env["DOCKER_PORT"])
	}
	if env["DOCKER_PORT_6379_TCP"] != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected tcp://172.0.17.2:6379, got %s", env["DOCKER_PORT_6379_TCP"])
	}
	if env["DOCKER_PORT_6379_TCP_PROTO"] != "tcp" {
		t.Fatalf("Expected tcp, got %s", env["DOCKER_PORT_6379_TCP_PROTO"])
	}
	if env["DOCKER_PORT_6379_TCP_ADDR"] != "172.0.17.2" {
		t.Fatalf("Expected 172.0.17.2, got %s", env["DOCKER_PORT_6379_TCP_ADDR"])
	}
	if env["DOCKER_PORT_6379_TCP_PORT"] != "6379" {
		t.Fatalf("Expected 6379, got %s", env["DOCKER_PORT_6379_TCP_PORT"])
	}
	if env["DOCKER_NAME"] != "/db/docker" {
		t.Fatalf("Expected /db/docker, got %s", env["DOCKER_NAME"])
	}
	if env["DOCKER_ENV_PASSWORD"] != "gordon" {
		t.Fatalf("Expected gordon, got %s", env["DOCKER_ENV_PASSWORD"])
	}
}

func TestLinkMultipleEnv(t *testing.T) {
	ports := make(nat.PortSet)
	ports[newPortNoError("tcp", "6379")] = struct{}{}
	ports[newPortNoError("tcp", "6380")] = struct{}{}
	ports[newPortNoError("tcp", "6381")] = struct{}{}

	link := NewLink("172.0.17.3", "172.0.17.2", "/db/docker", []string{"PASSWORD=gordon"}, ports)

	rawEnv := link.ToEnv()
	env := make(map[string]string, len(rawEnv))
	for _, e := range rawEnv {
		parts := strings.Split(e, "=")
		if len(parts) != 2 {
			t.FailNow()
		}
		env[parts[0]] = parts[1]
	}
	if env["DOCKER_PORT"] != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected 172.0.17.2:6379, got %s", env["DOCKER_PORT"])
	}
	if env["DOCKER_PORT_6379_TCP_START"] != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected tcp://172.0.17.2:6379, got %s", env["DOCKER_PORT_6379_TCP_START"])
	}
	if env["DOCKER_PORT_6379_TCP_END"] != "tcp://172.0.17.2:6381" {
		t.Fatalf("Expected tcp://172.0.17.2:6381, got %s", env["DOCKER_PORT_6379_TCP_END"])
	}
	if env["DOCKER_PORT_6379_TCP_PROTO"] != "tcp" {
		t.Fatalf("Expected tcp, got %s", env["DOCKER_PORT_6379_TCP_PROTO"])
	}
	if env["DOCKER_PORT_6379_TCP_ADDR"] != "172.0.17.2" {
		t.Fatalf("Expected 172.0.17.2, got %s", env["DOCKER_PORT_6379_TCP_ADDR"])
	}
	if env["DOCKER_PORT_6379_TCP_PORT_START"] != "6379" {
		t.Fatalf("Expected 6379, got %s", env["DOCKER_PORT_6379_TCP_PORT_START"])
	}
	if env["DOCKER_PORT_6379_TCP_PORT_END"] != "6381" {
		t.Fatalf("Expected 6381, got %s", env["DOCKER_PORT_6379_TCP_PORT_END"])
	}
	if env["DOCKER_NAME"] != "/db/docker" {
		t.Fatalf("Expected /db/docker, got %s", env["DOCKER_NAME"])
	}
	if env["DOCKER_ENV_PASSWORD"] != "gordon" {
		t.Fatalf("Expected gordon, got %s", env["DOCKER_ENV_PASSWORD"])
	}
}

func TestLinkPortRangeEnv(t *testing.T) {
	ports := make(nat.PortSet)
	ports[newPortNoError("tcp", "6379")] = struct{}{}
	ports[newPortNoError("tcp", "6380")] = struct{}{}
	ports[newPortNoError("tcp", "6381")] = struct{}{}

	link := NewLink("172.0.17.3", "172.0.17.2", "/db/docker", []string{"PASSWORD=gordon"}, ports)

	rawEnv := link.ToEnv()
	env := make(map[string]string, len(rawEnv))
	for _, e := range rawEnv {
		parts := strings.Split(e, "=")
		if len(parts) != 2 {
			t.FailNow()
		}
		env[parts[0]] = parts[1]
	}

	if env["DOCKER_PORT"] != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected 172.0.17.2:6379, got %s", env["DOCKER_PORT"])
	}
	if env["DOCKER_PORT_6379_TCP_START"] != "tcp://172.0.17.2:6379" {
		t.Fatalf("Expected tcp://172.0.17.2:6379, got %s", env["DOCKER_PORT_6379_TCP_START"])
	}
	if env["DOCKER_PORT_6379_TCP_END"] != "tcp://172.0.17.2:6381" {
		t.Fatalf("Expected tcp://172.0.17.2:6381, got %s", env["DOCKER_PORT_6379_TCP_END"])
	}
	if env["DOCKER_PORT_6379_TCP_PROTO"] != "tcp" {
		t.Fatalf("Expected tcp, got %s", env["DOCKER_PORT_6379_TCP_PROTO"])
	}
	if env["DOCKER_PORT_6379_TCP_ADDR"] != "172.0.17.2" {
		t.Fatalf("Expected 172.0.17.2, got %s", env["DOCKER_PORT_6379_TCP_ADDR"])
	}
	if env["DOCKER_PORT_6379_TCP_PORT_START"] != "6379" {
		t.Fatalf("Expected 6379, got %s", env["DOCKER_PORT_6379_TCP_PORT_START"])
	}
	if env["DOCKER_PORT_6379_TCP_PORT_END"] != "6381" {
		t.Fatalf("Expected 6381, got %s", env["DOCKER_PORT_6379_TCP_PORT_END"])
	}
	if env["DOCKER_NAME"] != "/db/docker" {
		t.Fatalf("Expected /db/docker, got %s", env["DOCKER_NAME"])
	}
	if env["DOCKER_ENV_PASSWORD"] != "gordon" {
		t.Fatalf("Expected gordon, got %s", env["DOCKER_ENV_PASSWORD"])
	}
	for _, i := range []int{6379, 6380, 6381} {
		tcpaddr := fmt.Sprintf("DOCKER_PORT_%d_TCP_ADDR", i)
		tcpport := fmt.Sprintf("DOCKER_PORT_%d_TCP_PORT", i)
		tcpproto := fmt.Sprintf("DOCKER_PORT_%d_TCP_PROTO", i)
		tcp := fmt.Sprintf("DOCKER_PORT_%d_TCP", i)
		if env[tcpaddr] != "172.0.17.2" {
			t.Fatalf("Expected env %s  = 172.0.17.2, got %s", tcpaddr, env[tcpaddr])
		}
		if env[tcpport] != fmt.Sprintf("%d", i) {
			t.Fatalf("Expected env %s  = %d, got %s", tcpport, i, env[tcpport])
		}
		if env[tcpproto] != "tcp" {
			t.Fatalf("Expected env %s  = tcp, got %s", tcpproto, env[tcpproto])
		}
		if env[tcp] != fmt.Sprintf("tcp://172.0.17.2:%d", i) {
			t.Fatalf("Expected env %s  = tcp://172.0.17.2:%d, got %s", tcp, i, env[tcp])
		}
	}
}
