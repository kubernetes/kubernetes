// +build !windows

package config

import (
	"testing"

	"github.com/docker/docker/opts"
	"github.com/docker/docker/pkg/testutil/tempfile"
	"github.com/docker/go-units"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetConflictFreeConfiguration(t *testing.T) {
	configFileData := string([]byte(`
		{
			"debug": true,
			"default-ulimits": {
				"nofile": {
					"Name": "nofile",
					"Hard": 2048,
					"Soft": 1024
				}
			},
			"log-opts": {
				"tag": "test_tag"
			}
		}`))

	file := tempfile.NewTempFile(t, "docker-config", configFileData)
	defer file.Remove()

	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)
	var debug bool
	flags.BoolVarP(&debug, "debug", "D", false, "")
	flags.Var(opts.NewNamedUlimitOpt("default-ulimits", nil), "default-ulimit", "")
	flags.Var(opts.NewNamedMapOpts("log-opts", nil, nil), "log-opt", "")

	cc, err := getConflictFreeConfiguration(file.Name(), flags)
	require.NoError(t, err)

	assert.True(t, cc.Debug)

	expectedUlimits := map[string]*units.Ulimit{
		"nofile": {
			Name: "nofile",
			Hard: 2048,
			Soft: 1024,
		},
	}

	assert.Equal(t, expectedUlimits, cc.Ulimits)
}

func TestDaemonConfigurationMerge(t *testing.T) {
	configFileData := string([]byte(`
		{
			"debug": true,
			"default-ulimits": {
				"nofile": {
					"Name": "nofile",
					"Hard": 2048,
					"Soft": 1024
				}
			},
			"log-opts": {
				"tag": "test_tag"
			}
		}`))

	file := tempfile.NewTempFile(t, "docker-config", configFileData)
	defer file.Remove()

	c := &Config{
		CommonConfig: CommonConfig{
			AutoRestart: true,
			LogConfig: LogConfig{
				Type:   "syslog",
				Config: map[string]string{"tag": "test"},
			},
		},
	}

	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)

	var debug bool
	flags.BoolVarP(&debug, "debug", "D", false, "")
	flags.Var(opts.NewNamedUlimitOpt("default-ulimits", nil), "default-ulimit", "")
	flags.Var(opts.NewNamedMapOpts("log-opts", nil, nil), "log-opt", "")

	cc, err := MergeDaemonConfigurations(c, flags, file.Name())
	require.NoError(t, err)

	assert.True(t, cc.Debug)
	assert.True(t, cc.AutoRestart)

	expectedLogConfig := LogConfig{
		Type:   "syslog",
		Config: map[string]string{"tag": "test_tag"},
	}

	assert.Equal(t, expectedLogConfig, cc.LogConfig)

	expectedUlimits := map[string]*units.Ulimit{
		"nofile": {
			Name: "nofile",
			Hard: 2048,
			Soft: 1024,
		},
	}

	assert.Equal(t, expectedUlimits, cc.Ulimits)
}

func TestDaemonConfigurationMergeShmSize(t *testing.T) {
	data := string([]byte(`
		{
			"default-shm-size": "1g"
		}`))

	file := tempfile.NewTempFile(t, "docker-config", data)
	defer file.Remove()

	c := &Config{}

	flags := pflag.NewFlagSet("test", pflag.ContinueOnError)
	shmSize := opts.MemBytes(DefaultShmSize)
	flags.Var(&shmSize, "default-shm-size", "")

	cc, err := MergeDaemonConfigurations(c, flags, file.Name())
	require.NoError(t, err)

	expectedValue := 1 * 1024 * 1024 * 1024
	if cc.ShmSize.Value() != int64(expectedValue) {
		t.Fatalf("expected default shm size %d, got %d", expectedValue, cc.ShmSize.Value())
	}
}
