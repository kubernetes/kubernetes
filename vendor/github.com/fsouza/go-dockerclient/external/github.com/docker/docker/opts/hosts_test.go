package opts

import (
	"runtime"
	"testing"
)

func TestParseHost(t *testing.T) {
	invalid := map[string]string{
		"anything":              "Invalid bind address format: anything",
		"something with spaces": "Invalid bind address format: something with spaces",
		"://":                "Invalid bind address format: ://",
		"unknown://":         "Invalid bind address format: unknown://",
		"tcp://:port":        "Invalid bind address format: :port",
		"tcp://invalid":      "Invalid bind address format: invalid",
		"tcp://invalid:port": "Invalid bind address format: invalid:port",
	}
	const defaultHTTPHost = "tcp://127.0.0.1:2375"
	var defaultHOST = "unix:///var/run/docker.sock"

	if runtime.GOOS == "windows" {
		defaultHOST = defaultHTTPHost
	}
	valid := map[string]string{
		"":                         defaultHOST,
		"fd://":                    "fd://",
		"fd://something":           "fd://something",
		"tcp://host:":              "tcp://host:2375",
		"tcp://":                   "tcp://localhost:2375",
		"tcp://:2375":              "tcp://localhost:2375", // default ip address
		"tcp://:2376":              "tcp://localhost:2376", // default ip address
		"tcp://0.0.0.0:8080":       "tcp://0.0.0.0:8080",
		"tcp://192.168.0.0:12000":  "tcp://192.168.0.0:12000",
		"tcp://192.168:8080":       "tcp://192.168:8080",
		"tcp://0.0.0.0:1234567890": "tcp://0.0.0.0:1234567890", // yeah it's valid :P
		"tcp://docker.com:2375":    "tcp://docker.com:2375",
		"unix://":                  "unix:///var/run/docker.sock", // default unix:// value
		"unix://path/to/socket":    "unix://path/to/socket",
	}

	for value, errorMessage := range invalid {
		if _, err := ParseHost(defaultHTTPHost, value); err == nil || err.Error() != errorMessage {
			t.Fatalf("Expected an error for %v with [%v], got [%v]", value, errorMessage, err)
		}
	}
	for value, expected := range valid {
		if actual, err := ParseHost(defaultHTTPHost, value); err != nil || actual != expected {
			t.Fatalf("Expected for %v [%v], got [%v, %v]", value, expected, actual, err)
		}
	}
}

func TestParseDockerDaemonHost(t *testing.T) {
	var (
		defaultHTTPHost  = "tcp://localhost:2375"
		defaultHTTPSHost = "tcp://localhost:2376"
		defaultUnix      = "/var/run/docker.sock"
		defaultHOST      = "unix:///var/run/docker.sock"
	)
	if runtime.GOOS == "windows" {
		defaultHOST = defaultHTTPHost
	}
	invalids := map[string]string{
		"0.0.0.0":                       "Invalid bind address format: 0.0.0.0",
		"tcp:a.b.c.d":                   "Invalid bind address format: tcp:a.b.c.d",
		"tcp:a.b.c.d/path":              "Invalid bind address format: tcp:a.b.c.d/path",
		"udp://127.0.0.1":               "Invalid bind address format: udp://127.0.0.1",
		"udp://127.0.0.1:2375":          "Invalid bind address format: udp://127.0.0.1:2375",
		"tcp://unix:///run/docker.sock": "Invalid bind address format: unix",
		"tcp":  "Invalid bind address format: tcp",
		"unix": "Invalid bind address format: unix",
		"fd":   "Invalid bind address format: fd",
	}
	valids := map[string]string{
		"0.0.0.1:":                    "tcp://0.0.0.1:2375",
		"0.0.0.1:5555":                "tcp://0.0.0.1:5555",
		"0.0.0.1:5555/path":           "tcp://0.0.0.1:5555/path",
		"[::1]:":                      "tcp://[::1]:2375",
		"[::1]:5555/path":             "tcp://[::1]:5555/path",
		"[0:0:0:0:0:0:0:1]:":          "tcp://[0:0:0:0:0:0:0:1]:2375",
		"[0:0:0:0:0:0:0:1]:5555/path": "tcp://[0:0:0:0:0:0:0:1]:5555/path",
		":6666":                   "tcp://localhost:6666",
		":6666/path":              "tcp://localhost:6666/path",
		"":                        defaultHOST,
		" ":                       defaultHOST,
		"  ":                      defaultHOST,
		"tcp://":                  defaultHTTPHost,
		"tcp://:7777":             "tcp://localhost:7777",
		"tcp://:7777/path":        "tcp://localhost:7777/path",
		" tcp://:7777/path ":      "tcp://localhost:7777/path",
		"unix:///run/docker.sock": "unix:///run/docker.sock",
		"unix://":                 "unix:///var/run/docker.sock",
		"fd://":                   "fd://",
		"fd://something":          "fd://something",
		"localhost:":              "tcp://localhost:2375",
		"localhost:5555":          "tcp://localhost:5555",
		"localhost:5555/path":     "tcp://localhost:5555/path",
	}
	for invalidAddr, expectedError := range invalids {
		if addr, err := parseDockerDaemonHost(defaultHTTPHost, defaultHTTPSHost, defaultUnix, "", invalidAddr); err == nil || err.Error() != expectedError {
			t.Errorf("tcp %v address expected error %v return, got %s and addr %v", invalidAddr, expectedError, err, addr)
		}
	}
	for validAddr, expectedAddr := range valids {
		if addr, err := parseDockerDaemonHost(defaultHTTPHost, defaultHTTPSHost, defaultUnix, "", validAddr); err != nil || addr != expectedAddr {
			t.Errorf("%v -> expected %v, got (%v) addr (%v)", validAddr, expectedAddr, err, addr)
		}
	}
}

func TestParseTCP(t *testing.T) {
	var (
		defaultHTTPHost = "tcp://127.0.0.1:2376"
	)
	invalids := map[string]string{
		"0.0.0.0":              "Invalid bind address format: 0.0.0.0",
		"tcp:a.b.c.d":          "Invalid bind address format: tcp:a.b.c.d",
		"tcp:a.b.c.d/path":     "Invalid bind address format: tcp:a.b.c.d/path",
		"udp://127.0.0.1":      "Invalid proto, expected tcp: udp://127.0.0.1",
		"udp://127.0.0.1:2375": "Invalid proto, expected tcp: udp://127.0.0.1:2375",
	}
	valids := map[string]string{
		"":                            defaultHTTPHost,
		"tcp://":                      defaultHTTPHost,
		"0.0.0.1:":                    "tcp://0.0.0.1:2376",
		"0.0.0.1:5555":                "tcp://0.0.0.1:5555",
		"0.0.0.1:5555/path":           "tcp://0.0.0.1:5555/path",
		":6666":                       "tcp://127.0.0.1:6666",
		":6666/path":                  "tcp://127.0.0.1:6666/path",
		"tcp://:7777":                 "tcp://127.0.0.1:7777",
		"tcp://:7777/path":            "tcp://127.0.0.1:7777/path",
		"[::1]:":                      "tcp://[::1]:2376",
		"[::1]:5555":                  "tcp://[::1]:5555",
		"[::1]:5555/path":             "tcp://[::1]:5555/path",
		"[0:0:0:0:0:0:0:1]:":          "tcp://[0:0:0:0:0:0:0:1]:2376",
		"[0:0:0:0:0:0:0:1]:5555":      "tcp://[0:0:0:0:0:0:0:1]:5555",
		"[0:0:0:0:0:0:0:1]:5555/path": "tcp://[0:0:0:0:0:0:0:1]:5555/path",
		"localhost:":                  "tcp://localhost:2376",
		"localhost:5555":              "tcp://localhost:5555",
		"localhost:5555/path":         "tcp://localhost:5555/path",
	}
	for invalidAddr, expectedError := range invalids {
		if addr, err := parseTCPAddr(invalidAddr, defaultHTTPHost); err == nil || err.Error() != expectedError {
			t.Errorf("tcp %v address expected error %v return, got %s and addr %v", invalidAddr, expectedError, err, addr)
		}
	}
	for validAddr, expectedAddr := range valids {
		if addr, err := parseTCPAddr(validAddr, defaultHTTPHost); err != nil || addr != expectedAddr {
			t.Errorf("%v -> expected %v, got %v and addr %v", validAddr, expectedAddr, err, addr)
		}
	}
}

func TestParseInvalidUnixAddrInvalid(t *testing.T) {
	if _, err := parseUnixAddr("tcp://127.0.0.1", "unix:///var/run/docker.sock"); err == nil || err.Error() != "Invalid proto, expected unix: tcp://127.0.0.1" {
		t.Fatalf("Expected an error, got %v", err)
	}
	if _, err := parseUnixAddr("unix://tcp://127.0.0.1", "/var/run/docker.sock"); err == nil || err.Error() != "Invalid proto, expected unix: tcp://127.0.0.1" {
		t.Fatalf("Expected an error, got %v", err)
	}
	if v, err := parseUnixAddr("", "/var/run/docker.sock"); err != nil || v != "unix:///var/run/docker.sock" {
		t.Fatalf("Expected an %v, got %v", v, "unix:///var/run/docker.sock")
	}
}
