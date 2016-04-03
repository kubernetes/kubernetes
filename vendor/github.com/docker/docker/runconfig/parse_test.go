package runconfig

import (
	"fmt"
	"io/ioutil"
	"strings"
	"testing"

	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/nat"
	"github.com/docker/docker/pkg/parsers"
)

func parseRun(args []string) (*Config, *HostConfig, *flag.FlagSet, error) {
	cmd := flag.NewFlagSet("run", flag.ContinueOnError)
	cmd.SetOutput(ioutil.Discard)
	cmd.Usage = nil
	return Parse(cmd, args)
}

func parse(t *testing.T, args string) (*Config, *HostConfig, error) {
	config, hostConfig, _, err := parseRun(strings.Split(args+" ubuntu bash", " "))
	return config, hostConfig, err
}

func mustParse(t *testing.T, args string) (*Config, *HostConfig) {
	config, hostConfig, err := parse(t, args)
	if err != nil {
		t.Fatal(err)
	}
	return config, hostConfig
}

// check if (a == c && b == d) || (a == d && b == c)
// because maps are randomized
func compareRandomizedStrings(a, b, c, d string) error {
	if a == c && b == d {
		return nil
	}
	if a == d && b == c {
		return nil
	}
	return fmt.Errorf("strings don't match")
}
func TestParseRunLinks(t *testing.T) {
	if _, hostConfig := mustParse(t, "--link a:b"); len(hostConfig.Links) == 0 || hostConfig.Links[0] != "a:b" {
		t.Fatalf("Error parsing links. Expected []string{\"a:b\"}, received: %v", hostConfig.Links)
	}
	if _, hostConfig := mustParse(t, "--link a:b --link c:d"); len(hostConfig.Links) < 2 || hostConfig.Links[0] != "a:b" || hostConfig.Links[1] != "c:d" {
		t.Fatalf("Error parsing links. Expected []string{\"a:b\", \"c:d\"}, received: %v", hostConfig.Links)
	}
	if _, hostConfig := mustParse(t, ""); len(hostConfig.Links) != 0 {
		t.Fatalf("Error parsing links. No link expected, received: %v", hostConfig.Links)
	}
}

func TestParseRunAttach(t *testing.T) {
	if config, _ := mustParse(t, "-a stdin"); !config.AttachStdin || config.AttachStdout || config.AttachStderr {
		t.Fatalf("Error parsing attach flags. Expect only Stdin enabled. Received: in: %v, out: %v, err: %v", config.AttachStdin, config.AttachStdout, config.AttachStderr)
	}
	if config, _ := mustParse(t, "-a stdin -a stdout"); !config.AttachStdin || !config.AttachStdout || config.AttachStderr {
		t.Fatalf("Error parsing attach flags. Expect only Stdin and Stdout enabled. Received: in: %v, out: %v, err: %v", config.AttachStdin, config.AttachStdout, config.AttachStderr)
	}
	if config, _ := mustParse(t, "-a stdin -a stdout -a stderr"); !config.AttachStdin || !config.AttachStdout || !config.AttachStderr {
		t.Fatalf("Error parsing attach flags. Expect all attach enabled. Received: in: %v, out: %v, err: %v", config.AttachStdin, config.AttachStdout, config.AttachStderr)
	}
	if config, _ := mustParse(t, ""); config.AttachStdin || !config.AttachStdout || !config.AttachStderr {
		t.Fatalf("Error parsing attach flags. Expect Stdin disabled. Received: in: %v, out: %v, err: %v", config.AttachStdin, config.AttachStdout, config.AttachStderr)
	}
	if config, _ := mustParse(t, "-i"); !config.AttachStdin || !config.AttachStdout || !config.AttachStderr {
		t.Fatalf("Error parsing attach flags. Expect Stdin enabled. Received: in: %v, out: %v, err: %v", config.AttachStdin, config.AttachStdout, config.AttachStderr)
	}

	if _, _, err := parse(t, "-a"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a` should be an error but is not")
	}
	if _, _, err := parse(t, "-a invalid"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a invalid` should be an error but is not")
	}
	if _, _, err := parse(t, "-a invalid -a stdout"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a stdout -a invalid` should be an error but is not")
	}
	if _, _, err := parse(t, "-a stdout -a stderr -d"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a stdout -a stderr -d` should be an error but is not")
	}
	if _, _, err := parse(t, "-a stdin -d"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a stdin -d` should be an error but is not")
	}
	if _, _, err := parse(t, "-a stdout -d"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a stdout -d` should be an error but is not")
	}
	if _, _, err := parse(t, "-a stderr -d"); err == nil {
		t.Fatalf("Error parsing attach flags, `-a stderr -d` should be an error but is not")
	}
	if _, _, err := parse(t, "-d --rm"); err == nil {
		t.Fatalf("Error parsing attach flags, `-d --rm` should be an error but is not")
	}
}

func TestParseRunVolumes(t *testing.T) {
	if config, hostConfig := mustParse(t, "-v /tmp"); hostConfig.Binds != nil {
		t.Fatalf("Error parsing volume flags, `-v /tmp` should not mount-bind anything. Received %v", hostConfig.Binds)
	} else if _, exists := config.Volumes["/tmp"]; !exists {
		t.Fatalf("Error parsing volume flags, `-v /tmp` is missing from volumes. Received %v", config.Volumes)
	}

	if config, hostConfig := mustParse(t, "-v /tmp -v /var"); hostConfig.Binds != nil {
		t.Fatalf("Error parsing volume flags, `-v /tmp -v /var` should not mount-bind anything. Received %v", hostConfig.Binds)
	} else if _, exists := config.Volumes["/tmp"]; !exists {
		t.Fatalf("Error parsing volume flags, `-v /tmp` is missing from volumes. Received %v", config.Volumes)
	} else if _, exists := config.Volumes["/var"]; !exists {
		t.Fatalf("Error parsing volume flags, `-v /var` is missing from volumes. Received %v", config.Volumes)
	}

	if _, hostConfig := mustParse(t, "-v /hostTmp:/containerTmp"); hostConfig.Binds == nil || hostConfig.Binds[0] != "/hostTmp:/containerTmp" {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp` should mount-bind /hostTmp into /containeTmp. Received %v", hostConfig.Binds)
	}

	if _, hostConfig := mustParse(t, "-v /hostTmp:/containerTmp -v /hostVar:/containerVar"); hostConfig.Binds == nil || compareRandomizedStrings(hostConfig.Binds[0], hostConfig.Binds[1], "/hostTmp:/containerTmp", "/hostVar:/containerVar") != nil {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp -v /hostVar:/containerVar` should mount-bind /hostTmp into /containeTmp and /hostVar into /hostContainer. Received %v", hostConfig.Binds)
	}

	if _, hostConfig := mustParse(t, "-v /hostTmp:/containerTmp:ro -v /hostVar:/containerVar:rw"); hostConfig.Binds == nil || compareRandomizedStrings(hostConfig.Binds[0], hostConfig.Binds[1], "/hostTmp:/containerTmp:ro", "/hostVar:/containerVar:rw") != nil {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp:ro -v /hostVar:/containerVar:rw` should mount-bind /hostTmp into /containeTmp and /hostVar into /hostContainer. Received %v", hostConfig.Binds)
	}

	if _, hostConfig := mustParse(t, "-v /containerTmp:ro -v /containerVar:rw"); hostConfig.Binds == nil || compareRandomizedStrings(hostConfig.Binds[0], hostConfig.Binds[1], "/containerTmp:ro", "/containerVar:rw") != nil {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp:ro -v /hostVar:/containerVar:rw` should mount-bind /hostTmp into /containeTmp and /hostVar into /hostContainer. Received %v", hostConfig.Binds)
	}

	if _, hostConfig := mustParse(t, "-v /hostTmp:/containerTmp:ro,Z -v /hostVar:/containerVar:rw,Z"); hostConfig.Binds == nil || compareRandomizedStrings(hostConfig.Binds[0], hostConfig.Binds[1], "/hostTmp:/containerTmp:ro,Z", "/hostVar:/containerVar:rw,Z") != nil {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp:ro,Z -v /hostVar:/containerVar:rw,Z` should mount-bind /hostTmp into /containeTmp and /hostVar into /hostContainer. Received %v", hostConfig.Binds)
	}

	if _, hostConfig := mustParse(t, "-v /hostTmp:/containerTmp:Z -v /hostVar:/containerVar:z"); hostConfig.Binds == nil || compareRandomizedStrings(hostConfig.Binds[0], hostConfig.Binds[1], "/hostTmp:/containerTmp:Z", "/hostVar:/containerVar:z") != nil {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp:Z -v /hostVar:/containerVar:z` should mount-bind /hostTmp into /containeTmp and /hostVar into /hostContainer. Received %v", hostConfig.Binds)
	}

	if config, hostConfig := mustParse(t, "-v /hostTmp:/containerTmp -v /containerVar"); hostConfig.Binds == nil || len(hostConfig.Binds) > 1 || hostConfig.Binds[0] != "/hostTmp:/containerTmp" {
		t.Fatalf("Error parsing volume flags, `-v /hostTmp:/containerTmp -v /containerVar` should mount-bind only /hostTmp into /containeTmp. Received %v", hostConfig.Binds)
	} else if _, exists := config.Volumes["/containerVar"]; !exists {
		t.Fatalf("Error parsing volume flags, `-v /containerVar` is missing from volumes. Received %v", config.Volumes)
	}

	if config, hostConfig := mustParse(t, ""); hostConfig.Binds != nil {
		t.Fatalf("Error parsing volume flags, without volume, nothing should be mount-binded. Received %v", hostConfig.Binds)
	} else if len(config.Volumes) != 0 {
		t.Fatalf("Error parsing volume flags, without volume, no volume should be present. Received %v", config.Volumes)
	}

	if _, _, err := parse(t, "-v /"); err == nil {
		t.Fatalf("Expected error, but got none")
	}

	if _, _, err := parse(t, "-v /:/"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v /:/` should fail but didn't")
	}
	if _, _, err := parse(t, "-v"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v` should fail but didn't")
	}
	if _, _, err := parse(t, "-v /tmp:"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v /tmp:` should fail but didn't")
	}
	if _, _, err := parse(t, "-v /tmp::"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v /tmp::` should fail but didn't")
	}
	if _, _, err := parse(t, "-v :"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v :` should fail but didn't")
	}
	if _, _, err := parse(t, "-v ::"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v ::` should fail but didn't")
	}
	if _, _, err := parse(t, "-v /tmp:/tmp:/tmp:/tmp"); err == nil {
		t.Fatalf("Error parsing volume flags, `-v /tmp:/tmp:/tmp:/tmp` should fail but didn't")
	}
}

func TestParseLxcConfOpt(t *testing.T) {
	opts := []string{"lxc.utsname=docker", "lxc.utsname = docker "}

	for _, o := range opts {
		k, v, err := parsers.ParseKeyValueOpt(o)
		if err != nil {
			t.FailNow()
		}
		if k != "lxc.utsname" {
			t.Fail()
		}
		if v != "docker" {
			t.Fail()
		}
	}

	// With parseRun too
	_, hostconfig, _, err := parseRun([]string{"lxc.utsname=docker", "lxc.utsname = docker ", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	for _, lxcConf := range hostconfig.LxcConf.Slice() {
		if lxcConf.Key != "lxc.utsname" || lxcConf.Value != "docker" {
			t.Fail()
		}
	}

}

func TestNetHostname(t *testing.T) {
	if _, _, _, err := parseRun([]string{"-h=name", "img", "cmd"}); err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}

	if _, _, _, err := parseRun([]string{"--net=host", "img", "cmd"}); err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}

	if _, _, _, err := parseRun([]string{"-h=name", "--net=bridge", "img", "cmd"}); err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}

	if _, _, _, err := parseRun([]string{"-h=name", "--net=none", "img", "cmd"}); err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}

	if _, _, _, err := parseRun([]string{"-h=name", "--net=host", "img", "cmd"}); err != ErrConflictNetworkHostname {
		t.Fatalf("Expected error ErrConflictNetworkHostname, got: %s", err)
	}

	if _, _, _, err := parseRun([]string{"-h=name", "--net=container:other", "img", "cmd"}); err != ErrConflictNetworkHostname {
		t.Fatalf("Expected error ErrConflictNetworkHostname, got: %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container", "img", "cmd"}); err == nil || err.Error() != "--net: invalid net mode: invalid container format container:<name|id>" {
		t.Fatalf("Expected error with --net=container, got : %v", err)
	}
	if _, _, _, err := parseRun([]string{"--net=weird", "img", "cmd"}); err == nil || err.Error() != "--net: invalid net mode: invalid --net: weird" {
		t.Fatalf("Expected error with --net=weird, got: %s", err)
	}
}

func TestConflictContainerNetworkAndLinks(t *testing.T) {
	if _, _, _, err := parseRun([]string{"--net=container:other", "--link=zip:zap", "img", "cmd"}); err != ErrConflictContainerNetworkAndLinks {
		t.Fatalf("Expected error ErrConflictContainerNetworkAndLinks, got: %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=host", "--link=zip:zap", "img", "cmd"}); err != ErrConflictHostNetworkAndLinks {
		t.Fatalf("Expected error ErrConflictHostNetworkAndLinks, got: %s", err)
	}
}

func TestConflictNetworkModeAndOptions(t *testing.T) {
	if _, _, _, err := parseRun([]string{"--net=host", "--dns=8.8.8.8", "img", "cmd"}); err != ErrConflictNetworkAndDns {
		t.Fatalf("Expected error ErrConflictNetworkAndDns, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container:other", "--dns=8.8.8.8", "img", "cmd"}); err != ErrConflictNetworkAndDns {
		t.Fatalf("Expected error ErrConflictNetworkAndDns, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=host", "--add-host=name:8.8.8.8", "img", "cmd"}); err != ErrConflictNetworkHosts {
		t.Fatalf("Expected error ErrConflictNetworkAndDns, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container:other", "--add-host=name:8.8.8.8", "img", "cmd"}); err != ErrConflictNetworkHosts {
		t.Fatalf("Expected error ErrConflictNetworkAndDns, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=host", "--mac-address=92:d0:c6:0a:29:33", "img", "cmd"}); err != ErrConflictContainerNetworkAndMac {
		t.Fatalf("Expected error ErrConflictContainerNetworkAndMac, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container:other", "--mac-address=92:d0:c6:0a:29:33", "img", "cmd"}); err != ErrConflictContainerNetworkAndMac {
		t.Fatalf("Expected error ErrConflictContainerNetworkAndMac, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container:other", "-P", "img", "cmd"}); err != ErrConflictNetworkPublishPorts {
		t.Fatalf("Expected error ErrConflictNetworkPublishPorts, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container:other", "-p", "8080", "img", "cmd"}); err != ErrConflictNetworkPublishPorts {
		t.Fatalf("Expected error ErrConflictNetworkPublishPorts, got %s", err)
	}
	if _, _, _, err := parseRun([]string{"--net=container:other", "--expose", "8000-9000", "img", "cmd"}); err != ErrConflictNetworkExposePorts {
		t.Fatalf("Expected error ErrConflictNetworkExposePorts, got %s", err)
	}
}

// Simple parse with MacAddress validatation
func TestParseWithMacAddress(t *testing.T) {
	invalidMacAddress := "--mac-address=invalidMacAddress"
	validMacAddress := "--mac-address=92:d0:c6:0a:29:33"
	if _, _, _, err := parseRun([]string{invalidMacAddress, "img", "cmd"}); err != nil && err.Error() != "invalidMacAddress is not a valid mac address" {
		t.Fatalf("Expected an error with %v mac-address, got %v", invalidMacAddress, err)
	}
	if config, _ := mustParse(t, validMacAddress); config.MacAddress != "92:d0:c6:0a:29:33" {
		t.Fatalf("Expected the config to have '92:d0:c6:0a:29:33' as MacAddress, got '%v'", config.MacAddress)
	}
}

func TestParseWithMemory(t *testing.T) {
	invalidMemory := "--memory=invalid"
	validMemory := "--memory=1G"
	if _, _, _, err := parseRun([]string{invalidMemory, "img", "cmd"}); err != nil && err.Error() != "invalid size: 'invalid'" {
		t.Fatalf("Expected an error with '%v' Memory, got '%v'", invalidMemory, err)
	}
	if _, hostconfig := mustParse(t, validMemory); hostconfig.Memory != 1073741824 {
		t.Fatalf("Expected the config to have '1G' as Memory, got '%v'", hostconfig.Memory)
	}
}

func TestParseWithMemorySwap(t *testing.T) {
	invalidMemory := "--memory-swap=invalid"
	validMemory := "--memory-swap=1G"
	anotherValidMemory := "--memory-swap=-1"
	if _, _, _, err := parseRun([]string{invalidMemory, "img", "cmd"}); err == nil || err.Error() != "invalid size: 'invalid'" {
		t.Fatalf("Expected an error with '%v' MemorySwap, got '%v'", invalidMemory, err)
	}
	if _, hostconfig := mustParse(t, validMemory); hostconfig.MemorySwap != 1073741824 {
		t.Fatalf("Expected the config to have '1073741824' as MemorySwap, got '%v'", hostconfig.MemorySwap)
	}
	if _, hostconfig := mustParse(t, anotherValidMemory); hostconfig.MemorySwap != -1 {
		t.Fatalf("Expected the config to have '-1' as MemorySwap, got '%v'", hostconfig.MemorySwap)
	}
}

func TestParseHostname(t *testing.T) {
	hostname := "--hostname=hostname"
	hostnameWithDomain := "--hostname=hostname.domainname"
	hostnameWithDomainTld := "--hostname=hostname.domainname.tld"
	if config, _ := mustParse(t, hostname); config.Hostname != "hostname" && config.Domainname != "" {
		t.Fatalf("Expected the config to have 'hostname' as hostname, got '%v'", config.Hostname)
	}
	if config, _ := mustParse(t, hostnameWithDomain); config.Hostname != "hostname" && config.Domainname != "domainname" {
		t.Fatalf("Expected the config to have 'hostname' as hostname, got '%v'", config.Hostname)
	}
	if config, _ := mustParse(t, hostnameWithDomainTld); config.Hostname != "hostname" && config.Domainname != "domainname.tld" {
		t.Fatalf("Expected the config to have 'hostname' as hostname, got '%v'", config.Hostname)
	}
}

func TestParseWithExpose(t *testing.T) {
	invalids := map[string]string{
		":":                   "Invalid port format for --expose: :",
		"8080:9090":           "Invalid port format for --expose: 8080:9090",
		"/tcp":                "Invalid range format for --expose: /tcp, error: Empty string specified for ports.",
		"/udp":                "Invalid range format for --expose: /udp, error: Empty string specified for ports.",
		"NaN/tcp":             `Invalid range format for --expose: NaN/tcp, error: strconv.ParseUint: parsing "NaN": invalid syntax`,
		"NaN-NaN/tcp":         `Invalid range format for --expose: NaN-NaN/tcp, error: strconv.ParseUint: parsing "NaN": invalid syntax`,
		"8080-NaN/tcp":        `Invalid range format for --expose: 8080-NaN/tcp, error: strconv.ParseUint: parsing "NaN": invalid syntax`,
		"1234567890-8080/tcp": `Invalid range format for --expose: 1234567890-8080/tcp, error: strconv.ParseUint: parsing "1234567890": value out of range`,
	}
	valids := map[string][]nat.Port{
		"8080/tcp":      {"8080/tcp"},
		"8080/udp":      {"8080/udp"},
		"8080/ncp":      {"8080/ncp"},
		"8080-8080/udp": {"8080/udp"},
		"8080-8082/tcp": {"8080/tcp", "8081/tcp", "8082/tcp"},
	}
	for expose, expectedError := range invalids {
		if _, _, _, err := parseRun([]string{fmt.Sprintf("--expose=%v", expose), "img", "cmd"}); err == nil || err.Error() != expectedError {
			t.Fatalf("Expected error '%v' with '--expose=%v', got '%v'", expectedError, expose, err)
		}
	}
	for expose, exposedPorts := range valids {
		config, _, _, err := parseRun([]string{fmt.Sprintf("--expose=%v", expose), "img", "cmd"})
		if err != nil {
			t.Fatal(err)
		}
		if len(config.ExposedPorts) != len(exposedPorts) {
			t.Fatalf("Expected %v exposed port, got %v", len(exposedPorts), len(config.ExposedPorts))
		}
		for _, port := range exposedPorts {
			if _, ok := config.ExposedPorts[port]; !ok {
				t.Fatalf("Expected %v, got %v", exposedPorts, config.ExposedPorts)
			}
		}
	}
	// Merge with actual published port
	config, _, _, err := parseRun([]string{"--publish=80", "--expose=80-81/tcp", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(config.ExposedPorts) != 2 {
		t.Fatalf("Expected 2 exposed ports, got %v", config.ExposedPorts)
	}
	ports := []nat.Port{"80/tcp", "81/tcp"}
	for _, port := range ports {
		if _, ok := config.ExposedPorts[port]; !ok {
			t.Fatalf("Expected %v, got %v", ports, config.ExposedPorts)
		}
	}
}

func TestParseDevice(t *testing.T) {
	valids := map[string]DeviceMapping{
		"/dev/snd": {
			PathOnHost:        "/dev/snd",
			PathInContainer:   "/dev/snd",
			CgroupPermissions: "rwm",
		},
		"/dev/snd:/something": {
			PathOnHost:        "/dev/snd",
			PathInContainer:   "/something",
			CgroupPermissions: "rwm",
		},
		"/dev/snd:/something:ro": {
			PathOnHost:        "/dev/snd",
			PathInContainer:   "/something",
			CgroupPermissions: "ro",
		},
	}
	for device, deviceMapping := range valids {
		_, hostconfig, _, err := parseRun([]string{fmt.Sprintf("--device=%v", device), "img", "cmd"})
		if err != nil {
			t.Fatal(err)
		}
		if len(hostconfig.Devices) != 1 {
			t.Fatalf("Expected 1 devices, got %v", hostconfig.Devices)
		}
		if hostconfig.Devices[0] != deviceMapping {
			t.Fatalf("Expected %v, got %v", deviceMapping, hostconfig.Devices)
		}
	}

}

func TestParseModes(t *testing.T) {
	// ipc ko
	if _, _, _, err := parseRun([]string{"--ipc=container:", "img", "cmd"}); err == nil || err.Error() != "--ipc: invalid IPC mode" {
		t.Fatalf("Expected an error with message '--ipc: invalid IPC mode', got %v", err)
	}
	// ipc ok
	_, hostconfig, _, err := parseRun([]string{"--ipc=host", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if !hostconfig.IpcMode.Valid() {
		t.Fatalf("Expected a valid IpcMode, got %v", hostconfig.IpcMode)
	}
	// pid ko
	if _, _, _, err := parseRun([]string{"--pid=container:", "img", "cmd"}); err == nil || err.Error() != "--pid: invalid PID mode" {
		t.Fatalf("Expected an error with message '--pid: invalid PID mode', got %v", err)
	}
	// pid ok
	_, hostconfig, _, err = parseRun([]string{"--pid=host", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if !hostconfig.PidMode.Valid() {
		t.Fatalf("Expected a valid PidMode, got %v", hostconfig.PidMode)
	}
	// uts ko
	if _, _, _, err := parseRun([]string{"--uts=container:", "img", "cmd"}); err == nil || err.Error() != "--uts: invalid UTS mode" {
		t.Fatalf("Expected an error with message '--uts: invalid UTS mode', got %v", err)
	}
	// uts ok
	_, hostconfig, _, err = parseRun([]string{"--uts=host", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if !hostconfig.UTSMode.Valid() {
		t.Fatalf("Expected a valid UTSMode, got %v", hostconfig.UTSMode)
	}
}

func TestParseRestartPolicy(t *testing.T) {
	invalids := map[string]string{
		"something":          "invalid restart policy something",
		"always:2":           "maximum restart count not valid with restart policy of \"always\"",
		"always:2:3":         "maximum restart count not valid with restart policy of \"always\"",
		"on-failure:invalid": `strconv.ParseInt: parsing "invalid": invalid syntax`,
		"on-failure:2:5":     "restart count format is not valid, usage: 'on-failure:N' or 'on-failure'",
	}
	valids := map[string]RestartPolicy{
		"": {},
		"always": {
			Name:              "always",
			MaximumRetryCount: 0,
		},
		"on-failure:1": {
			Name:              "on-failure",
			MaximumRetryCount: 1,
		},
	}
	for restart, expectedError := range invalids {
		if _, _, _, err := parseRun([]string{fmt.Sprintf("--restart=%s", restart), "img", "cmd"}); err == nil || err.Error() != expectedError {
			t.Fatalf("Expected an error with message '%v' for %v, got %v", expectedError, restart, err)
		}
	}
	for restart, expected := range valids {
		_, hostconfig, _, err := parseRun([]string{fmt.Sprintf("--restart=%v", restart), "img", "cmd"})
		if err != nil {
			t.Fatal(err)
		}
		if hostconfig.RestartPolicy != expected {
			t.Fatalf("Expected %v, got %v", expected, hostconfig.RestartPolicy)
		}
	}
}

func TestParseLoggingOpts(t *testing.T) {
	// logging opts ko
	if _, _, _, err := parseRun([]string{"--log-driver=none", "--log-opt=anything", "img", "cmd"}); err == nil || err.Error() != "Invalid logging opts for driver none" {
		t.Fatalf("Expected an error with message 'Invalid logging opts for driver none', got %v", err)
	}
	// logging opts ok
	_, hostconfig, _, err := parseRun([]string{"--log-driver=syslog", "--log-opt=something", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if hostconfig.LogConfig.Type != "syslog" || len(hostconfig.LogConfig.Config) != 1 {
		t.Fatalf("Expected a 'syslog' LogConfig with one config, got %v", hostconfig.RestartPolicy)
	}
}

func TestParseEnvfileVariables(t *testing.T) {
	// env ko
	if _, _, _, err := parseRun([]string{"--env-file=nonexistent", "img", "cmd"}); err == nil || err.Error() != "open nonexistent: no such file or directory" {
		t.Fatalf("Expected an error with message 'open nonexistent: no such file or directory', got %v", err)
	}
	// env ok
	config, _, _, err := parseRun([]string{"--env-file=fixtures/valid.env", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(config.Env) != 1 || config.Env[0] != "ENV1=value1" {
		t.Fatalf("Expected a a config with [ENV1=value1], got %v", config.Env)
	}
	config, _, _, err = parseRun([]string{"--env-file=fixtures/valid.env", "--env=ENV2=value2", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(config.Env) != 2 || config.Env[0] != "ENV1=value1" || config.Env[1] != "ENV2=value2" {
		t.Fatalf("Expected a a config with [ENV1=value1 ENV2=value2], got %v", config.Env)
	}
}

func TestParseLabelfileVariables(t *testing.T) {
	// label ko
	if _, _, _, err := parseRun([]string{"--label-file=nonexistent", "img", "cmd"}); err == nil || err.Error() != "open nonexistent: no such file or directory" {
		t.Fatalf("Expected an error with message 'open nonexistent: no such file or directory', got %v", err)
	}
	// label ok
	config, _, _, err := parseRun([]string{"--label-file=fixtures/valid.label", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(config.Labels) != 1 || config.Labels["LABEL1"] != "value1" {
		t.Fatalf("Expected a a config with [LABEL1:value1], got %v", config.Labels)
	}
	config, _, _, err = parseRun([]string{"--label-file=fixtures/valid.label", "--label=LABEL2=value2", "img", "cmd"})
	if err != nil {
		t.Fatal(err)
	}
	if len(config.Labels) != 2 || config.Labels["LABEL1"] != "value1" || config.Labels["LABEL2"] != "value2" {
		t.Fatalf("Expected a a config with [LABEL1:value1 LABEL2:value2], got %v", config.Labels)
	}
}

func TestParseEntryPoint(t *testing.T) {
	config, _, _, err := parseRun([]string{"--entrypoint=anything", "cmd", "img"})
	if err != nil {
		t.Fatal(err)
	}
	if config.Entrypoint.Len() != 1 && config.Entrypoint.parts[0] != "anything" {
		t.Fatalf("Expected entrypoint 'anything', got %v", config.Entrypoint)
	}
}
