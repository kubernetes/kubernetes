package config

import (
	"io/ioutil"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/docker/docker/daemon/discovery"
	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/testutil"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
)

func TestDaemonConfigurationNotFound(t *testing.T) {
	_, err := MergeDaemonConfigurations(&Config{}, nil, "/tmp/foo-bar-baz-docker")
	if err == nil || !os.IsNotExist(err) {
		t.Fatalf("expected does not exist error, got %v", err)
	}
}

func TestDaemonBrokenConfiguration(t *testing.T) {
	f, err := ioutil.TempFile("", "docker-config-")
	if err != nil {
		t.Fatal(err)
	}

	configFile := f.Name()
	f.Write([]byte(`{"Debug": tru`))
	f.Close()

	_, err = MergeDaemonConfigurations(&Config{}, nil, configFile)
	if err == nil {
		t.Fatalf("expected error, got %v", err)
	}
}

func TestParseClusterAdvertiseSettings(t *testing.T) {
	if runtime.GOOS == "solaris" {
		t.Skip("ClusterSettings not supported on Solaris\n")
	}
	_, err := ParseClusterAdvertiseSettings("something", "")
	if err != discovery.ErrDiscoveryDisabled {
		t.Fatalf("expected discovery disabled error, got %v\n", err)
	}

	_, err = ParseClusterAdvertiseSettings("", "something")
	if err == nil {
		t.Fatalf("expected discovery store error, got %v\n", err)
	}

	_, err = ParseClusterAdvertiseSettings("etcd", "127.0.0.1:8080")
	if err != nil {
		t.Fatal(err)
	}
}

func TestFindConfigurationConflicts(t *testing.T) {
	config := map[string]interface{}{"authorization-plugins": "foobar"}
	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)

	flags.String("authorization-plugins", "", "")
	assert.NoError(t, flags.Set("authorization-plugins", "asdf"))

	testutil.ErrorContains(t,
		findConfigurationConflicts(config, flags),
		"authorization-plugins: (from flag: asdf, from file: foobar)")
}

func TestFindConfigurationConflictsWithNamedOptions(t *testing.T) {
	config := map[string]interface{}{"hosts": []string{"qwer"}}
	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)

	var hosts []string
	flags.VarP(opts.NewNamedListOptsRef("hosts", &hosts, opts.ValidateHost), "host", "H", "Daemon socket(s) to connect to")
	assert.NoError(t, flags.Set("host", "tcp://127.0.0.1:4444"))
	assert.NoError(t, flags.Set("host", "unix:///var/run/docker.sock"))

	testutil.ErrorContains(t, findConfigurationConflicts(config, flags), "hosts")
}

func TestDaemonConfigurationMergeConflicts(t *testing.T) {
	f, err := ioutil.TempFile("", "docker-config-")
	if err != nil {
		t.Fatal(err)
	}

	configFile := f.Name()
	f.Write([]byte(`{"debug": true}`))
	f.Close()

	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)
	flags.Bool("debug", false, "")
	flags.Set("debug", "false")

	_, err = MergeDaemonConfigurations(&Config{}, flags, configFile)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "debug") {
		t.Fatalf("expected debug conflict, got %v", err)
	}
}

func TestDaemonConfigurationMergeConcurrent(t *testing.T) {
	f, err := ioutil.TempFile("", "docker-config-")
	if err != nil {
		t.Fatal(err)
	}

	configFile := f.Name()
	f.Write([]byte(`{"max-concurrent-downloads": 1}`))
	f.Close()

	_, err = MergeDaemonConfigurations(&Config{}, nil, configFile)
	if err != nil {
		t.Fatal("expected error, got nil")
	}
}

func TestDaemonConfigurationMergeConcurrentError(t *testing.T) {
	f, err := ioutil.TempFile("", "docker-config-")
	if err != nil {
		t.Fatal(err)
	}

	configFile := f.Name()
	f.Write([]byte(`{"max-concurrent-downloads": -1}`))
	f.Close()

	_, err = MergeDaemonConfigurations(&Config{}, nil, configFile)
	if err == nil {
		t.Fatalf("expected no error, got error %v", err)
	}
}

func TestDaemonConfigurationMergeConflictsWithInnerStructs(t *testing.T) {
	f, err := ioutil.TempFile("", "docker-config-")
	if err != nil {
		t.Fatal(err)
	}

	configFile := f.Name()
	f.Write([]byte(`{"tlscacert": "/etc/certificates/ca.pem"}`))
	f.Close()

	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)
	flags.String("tlscacert", "", "")
	flags.Set("tlscacert", "~/.docker/ca.pem")

	_, err = MergeDaemonConfigurations(&Config{}, flags, configFile)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "tlscacert") {
		t.Fatalf("expected tlscacert conflict, got %v", err)
	}
}

func TestFindConfigurationConflictsWithUnknownKeys(t *testing.T) {
	config := map[string]interface{}{"tls-verify": "true"}
	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)

	flags.Bool("tlsverify", false, "")
	err := findConfigurationConflicts(config, flags)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "the following directives don't match any configuration option: tls-verify") {
		t.Fatalf("expected tls-verify conflict, got %v", err)
	}
}

func TestFindConfigurationConflictsWithMergedValues(t *testing.T) {
	var hosts []string
	config := map[string]interface{}{"hosts": "tcp://127.0.0.1:2345"}
	flags := pflag.NewFlagSet("base", pflag.ContinueOnError)
	flags.VarP(opts.NewNamedListOptsRef("hosts", &hosts, nil), "host", "H", "")

	err := findConfigurationConflicts(config, flags)
	if err != nil {
		t.Fatal(err)
	}

	flags.Set("host", "unix:///var/run/docker.sock")
	err = findConfigurationConflicts(config, flags)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "hosts: (from flag: [unix:///var/run/docker.sock], from file: tcp://127.0.0.1:2345)") {
		t.Fatalf("expected hosts conflict, got %v", err)
	}
}

func TestValidateConfigurationErrors(t *testing.T) {
	minusNumber := -10
	testCases := []struct {
		config *Config
	}{
		{
			config: &Config{
				CommonConfig: CommonConfig{
					Labels: []string{"one"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					Labels: []string{"foo=bar", "one"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					DNS: []string{"1.1.1.1o"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					DNS: []string{"2.2.2.2", "1.1.1.1o"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					DNSSearch: []string{"123456"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					DNSSearch: []string{"a.b.c", "123456"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					MaxConcurrentDownloads: &minusNumber,
					// This is weird...
					ValuesSet: map[string]interface{}{
						"max-concurrent-downloads": -1,
					},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					MaxConcurrentUploads: &minusNumber,
					// This is weird...
					ValuesSet: map[string]interface{}{
						"max-concurrent-uploads": -1,
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		err := Validate(tc.config)
		if err == nil {
			t.Fatalf("expected error, got nil for config %v", tc.config)
		}
	}
}

func TestValidateConfiguration(t *testing.T) {
	minusNumber := 4
	testCases := []struct {
		config *Config
	}{
		{
			config: &Config{
				CommonConfig: CommonConfig{
					Labels: []string{"one=two"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					DNS: []string{"1.1.1.1"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					DNSSearch: []string{"a.b.c"},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					MaxConcurrentDownloads: &minusNumber,
					// This is weird...
					ValuesSet: map[string]interface{}{
						"max-concurrent-downloads": -1,
					},
				},
			},
		},
		{
			config: &Config{
				CommonConfig: CommonConfig{
					MaxConcurrentUploads: &minusNumber,
					// This is weird...
					ValuesSet: map[string]interface{}{
						"max-concurrent-uploads": -1,
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		err := Validate(tc.config)
		if err != nil {
			t.Fatalf("expected no error, got error %v", err)
		}
	}
}

func TestModifiedDiscoverySettings(t *testing.T) {
	cases := []struct {
		current  *Config
		modified *Config
		expected bool
	}{
		{
			current:  discoveryConfig("foo", "bar", map[string]string{}),
			modified: discoveryConfig("foo", "bar", map[string]string{}),
			expected: false,
		},
		{
			current:  discoveryConfig("foo", "bar", map[string]string{"foo": "bar"}),
			modified: discoveryConfig("foo", "bar", map[string]string{"foo": "bar"}),
			expected: false,
		},
		{
			current:  discoveryConfig("foo", "bar", map[string]string{}),
			modified: discoveryConfig("foo", "bar", nil),
			expected: false,
		},
		{
			current:  discoveryConfig("foo", "bar", nil),
			modified: discoveryConfig("foo", "bar", map[string]string{}),
			expected: false,
		},
		{
			current:  discoveryConfig("foo", "bar", nil),
			modified: discoveryConfig("baz", "bar", nil),
			expected: true,
		},
		{
			current:  discoveryConfig("foo", "bar", nil),
			modified: discoveryConfig("foo", "baz", nil),
			expected: true,
		},
		{
			current:  discoveryConfig("foo", "bar", nil),
			modified: discoveryConfig("foo", "bar", map[string]string{"foo": "bar"}),
			expected: true,
		},
	}

	for _, c := range cases {
		got := ModifiedDiscoverySettings(c.current, c.modified.ClusterStore, c.modified.ClusterAdvertise, c.modified.ClusterOpts)
		if c.expected != got {
			t.Fatalf("expected %v, got %v: current config %v, new config %v", c.expected, got, c.current, c.modified)
		}
	}
}

func discoveryConfig(backendAddr, advertiseAddr string, opts map[string]string) *Config {
	return &Config{
		CommonConfig: CommonConfig{
			ClusterStore:     backendAddr,
			ClusterAdvertise: advertiseAddr,
			ClusterOpts:      opts,
		},
	}
}
