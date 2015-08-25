package parsers

import (
	"strings"
	"testing"
)

func TestParseHost(t *testing.T) {
	var (
		defaultHTTPHost = "127.0.0.1"
		defaultUnix     = "/var/run/docker.sock"
	)
	invalids := map[string]string{
		"0.0.0.0":              "Invalid bind address format: 0.0.0.0",
		"tcp://":               "Invalid proto, expected tcp: ",
		"tcp:a.b.c.d":          "Invalid bind address format: tcp:a.b.c.d",
		"tcp:a.b.c.d/path":     "Invalid bind address format: tcp:a.b.c.d/path",
		"udp://127.0.0.1":      "Invalid bind address format: udp://127.0.0.1",
		"udp://127.0.0.1:2375": "Invalid bind address format: udp://127.0.0.1:2375",
	}
	valids := map[string]string{
		"0.0.0.1:5555":      "tcp://0.0.0.1:5555",
		"0.0.0.1:5555/path": "tcp://0.0.0.1:5555/path",
		":6666":             "tcp://127.0.0.1:6666",
		":6666/path":        "tcp://127.0.0.1:6666/path",
		"tcp://:7777":       "tcp://127.0.0.1:7777",
		"tcp://:7777/path":  "tcp://127.0.0.1:7777/path",
		"":                  "unix:///var/run/docker.sock",
		"unix:///run/docker.sock": "unix:///run/docker.sock",
		"unix://":                 "unix:///var/run/docker.sock",
		"fd://":                   "fd://",
		"fd://something":          "fd://something",
	}
	for invalidAddr, expectedError := range invalids {
		if addr, err := ParseHost(defaultHTTPHost, defaultUnix, invalidAddr); err == nil || err.Error() != expectedError {
			t.Errorf("tcp %v address expected error %v return, got %s and addr %v", invalidAddr, expectedError, err, addr)
		}
	}
	for validAddr, expectedAddr := range valids {
		if addr, err := ParseHost(defaultHTTPHost, defaultUnix, validAddr); err != nil || addr != expectedAddr {
			t.Errorf("%v -> expected %v, got %v", validAddr, expectedAddr, addr)
		}
	}
}

func TestParseInvalidUnixAddrInvalid(t *testing.T) {
	if _, err := ParseUnixAddr("unix://tcp://127.0.0.1", "unix:///var/run/docker.sock"); err == nil || err.Error() != "Invalid proto, expected unix: tcp://127.0.0.1" {
		t.Fatalf("Expected an error, got %v", err)
	}
}

func TestParseRepositoryTag(t *testing.T) {
	if repo, tag := ParseRepositoryTag("root"); repo != "root" || tag != "" {
		t.Errorf("Expected repo: '%s' and tag: '%s', got '%s' and '%s'", "root", "", repo, tag)
	}
	if repo, tag := ParseRepositoryTag("root:tag"); repo != "root" || tag != "tag" {
		t.Errorf("Expected repo: '%s' and tag: '%s', got '%s' and '%s'", "root", "tag", repo, tag)
	}
	if repo, digest := ParseRepositoryTag("root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"); repo != "root" || digest != "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Errorf("Expected repo: '%s' and digest: '%s', got '%s' and '%s'", "root", "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", repo, digest)
	}
	if repo, tag := ParseRepositoryTag("user/repo"); repo != "user/repo" || tag != "" {
		t.Errorf("Expected repo: '%s' and tag: '%s', got '%s' and '%s'", "user/repo", "", repo, tag)
	}
	if repo, tag := ParseRepositoryTag("user/repo:tag"); repo != "user/repo" || tag != "tag" {
		t.Errorf("Expected repo: '%s' and tag: '%s', got '%s' and '%s'", "user/repo", "tag", repo, tag)
	}
	if repo, digest := ParseRepositoryTag("user/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"); repo != "user/repo" || digest != "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Errorf("Expected repo: '%s' and digest: '%s', got '%s' and '%s'", "user/repo", "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", repo, digest)
	}
	if repo, tag := ParseRepositoryTag("url:5000/repo"); repo != "url:5000/repo" || tag != "" {
		t.Errorf("Expected repo: '%s' and tag: '%s', got '%s' and '%s'", "url:5000/repo", "", repo, tag)
	}
	if repo, tag := ParseRepositoryTag("url:5000/repo:tag"); repo != "url:5000/repo" || tag != "tag" {
		t.Errorf("Expected repo: '%s' and tag: '%s', got '%s' and '%s'", "url:5000/repo", "tag", repo, tag)
	}
	if repo, digest := ParseRepositoryTag("url:5000/repo@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"); repo != "url:5000/repo" || digest != "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Errorf("Expected repo: '%s' and digest: '%s', got '%s' and '%s'", "url:5000/repo", "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", repo, digest)
	}
}

func TestParsePortMapping(t *testing.T) {
	if _, err := PartParser("ip:public:private", "192.168.1.1:80"); err == nil {
		t.Fatalf("Expected an error, got %v", err)
	}
	data, err := PartParser("ip:public:private", "192.168.1.1:80:8080")
	if err != nil {
		t.Fatal(err)
	}

	if len(data) != 3 {
		t.FailNow()
	}
	if data["ip"] != "192.168.1.1" {
		t.Fail()
	}
	if data["public"] != "80" {
		t.Fail()
	}
	if data["private"] != "8080" {
		t.Fail()
	}
}

func TestParseKeyValueOpt(t *testing.T) {
	invalids := map[string]string{
		"":    "Unable to parse key/value option: ",
		"key": "Unable to parse key/value option: key",
	}
	for invalid, expectedError := range invalids {
		if _, _, err := ParseKeyValueOpt(invalid); err == nil || err.Error() != expectedError {
			t.Fatalf("Expected error %v for %v, got %v", expectedError, invalid, err)
		}
	}
	valids := map[string][]string{
		"key=value":               {"key", "value"},
		" key = value ":           {"key", "value"},
		"key=value1=value2":       {"key", "value1=value2"},
		" key = value1 = value2 ": {"key", "value1 = value2"},
	}
	for valid, expectedKeyValue := range valids {
		key, value, err := ParseKeyValueOpt(valid)
		if err != nil {
			t.Fatal(err)
		}
		if key != expectedKeyValue[0] || value != expectedKeyValue[1] {
			t.Fatalf("Expected {%v: %v} got {%v: %v}", expectedKeyValue[0], expectedKeyValue[1], key, value)
		}
	}
}

func TestParsePortRange(t *testing.T) {
	if start, end, err := ParsePortRange("8000-8080"); err != nil || start != 8000 || end != 8080 {
		t.Fatalf("Error: %s or Expecting {start,end} values {8000,8080} but found {%d,%d}.", err, start, end)
	}
}

func TestParsePortRangeEmpty(t *testing.T) {
	if _, _, err := ParsePortRange(""); err == nil || err.Error() != "Empty string specified for ports." {
		t.Fatalf("Expected error 'Empty string specified for ports.', got %v", err)
	}
}

func TestParsePortRangeWithNoRange(t *testing.T) {
	start, end, err := ParsePortRange("8080")
	if err != nil {
		t.Fatal(err)
	}
	if start != 8080 || end != 8080 {
		t.Fatalf("Expected start and end to be the same and equal to 8080, but were %v and %v", start, end)
	}
}

func TestParsePortRangeIncorrectRange(t *testing.T) {
	if _, _, err := ParsePortRange("9000-8080"); err == nil || !strings.Contains(err.Error(), "Invalid range specified for the Port") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}
}

func TestParsePortRangeIncorrectEndRange(t *testing.T) {
	if _, _, err := ParsePortRange("8000-a"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}

	if _, _, err := ParsePortRange("8000-30a"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}
}

func TestParsePortRangeIncorrectStartRange(t *testing.T) {
	if _, _, err := ParsePortRange("a-8000"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}

	if _, _, err := ParsePortRange("30a-8000"); err == nil || !strings.Contains(err.Error(), "invalid syntax") {
		t.Fatalf("Expecting error 'Invalid range specified for the Port' but received %s.", err)
	}
}

func TestParseLink(t *testing.T) {
	name, alias, err := ParseLink("name:alias")
	if err != nil {
		t.Fatalf("Expected not to error out on a valid name:alias format but got: %v", err)
	}
	if name != "name" {
		t.Fatalf("Link name should have been name, got %s instead", name)
	}
	if alias != "alias" {
		t.Fatalf("Link alias should have been alias, got %s instead", alias)
	}
	// short format definition
	name, alias, err = ParseLink("name")
	if err != nil {
		t.Fatalf("Expected not to error out on a valid name only format but got: %v", err)
	}
	if name != "name" {
		t.Fatalf("Link name should have been name, got %s instead", name)
	}
	if alias != "name" {
		t.Fatalf("Link alias should have been name, got %s instead", alias)
	}
	// empty string link definition is not allowed
	if _, _, err := ParseLink(""); err == nil || !strings.Contains(err.Error(), "empty string specified for links") {
		t.Fatalf("Expected error 'empty string specified for links' but got: %v", err)
	}
	// more than two colons are not allowed
	if _, _, err := ParseLink("link:alias:wrong"); err == nil || !strings.Contains(err.Error(), "bad format for links: link:alias:wrong") {
		t.Fatalf("Expected error 'bad format for links: link:alias:wrong' but got: %v", err)
	}
}
