package parsers

import (
	"strings"
	"testing"
)

func TestParseHost(t *testing.T) {
	var (
		defaultHttpHost = "127.0.0.1"
		defaultUnix     = "/var/run/docker.sock"
	)
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "0.0.0.0"); err == nil {
		t.Errorf("tcp 0.0.0.0 address expected error return, but err == nil, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "tcp://"); err == nil {
		t.Errorf("default tcp:// address expected error return, but err == nil, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "0.0.0.1:5555"); err != nil || addr != "tcp://0.0.0.1:5555" {
		t.Errorf("0.0.0.1:5555 -> expected tcp://0.0.0.1:5555, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, ":6666"); err != nil || addr != "tcp://127.0.0.1:6666" {
		t.Errorf(":6666 -> expected tcp://127.0.0.1:6666, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "tcp://:7777"); err != nil || addr != "tcp://127.0.0.1:7777" {
		t.Errorf("tcp://:7777 -> expected tcp://127.0.0.1:7777, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, ""); err != nil || addr != "unix:///var/run/docker.sock" {
		t.Errorf("empty argument -> expected unix:///var/run/docker.sock, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "unix:///var/run/docker.sock"); err != nil || addr != "unix:///var/run/docker.sock" {
		t.Errorf("unix:///var/run/docker.sock -> expected unix:///var/run/docker.sock, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "unix://"); err != nil || addr != "unix:///var/run/docker.sock" {
		t.Errorf("unix:///var/run/docker.sock -> expected unix:///var/run/docker.sock, got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "udp://127.0.0.1"); err == nil {
		t.Errorf("udp protocol address expected error return, but err == nil. Got %s", addr)
	}
	if addr, err := ParseHost(defaultHttpHost, defaultUnix, "udp://127.0.0.1:2375"); err == nil {
		t.Errorf("udp protocol address expected error return, but err == nil. Got %s", addr)
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

func TestParsePortRange(t *testing.T) {
	if start, end, err := ParsePortRange("8000-8080"); err != nil || start != 8000 || end != 8080 {
		t.Fatalf("Error: %s or Expecting {start,end} values {8000,8080} but found {%d,%d}.", err, start, end)
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
