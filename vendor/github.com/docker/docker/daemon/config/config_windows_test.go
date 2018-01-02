// +build windows

package config

import (
	"io/ioutil"
	"testing"

	"github.com/docker/docker/opts"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDaemonConfigurationMerge(t *testing.T) {
	f, err := ioutil.TempFile("", "docker-config-")
	if err != nil {
		t.Fatal(err)
	}

	configFile := f.Name()

	f.Write([]byte(`
		{
			"debug": true,
			"log-opts": {
				"tag": "test_tag"
			}
		}`))

	f.Close()

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
	flags.Var(opts.NewNamedMapOpts("log-opts", nil, nil), "log-opt", "")

	cc, err := MergeDaemonConfigurations(c, flags, configFile)
	require.NoError(t, err)

	assert.True(t, cc.Debug)
	assert.True(t, cc.AutoRestart)

	expectedLogConfig := LogConfig{
		Type:   "syslog",
		Config: map[string]string{"tag": "test_tag"},
	}

	assert.Equal(t, expectedLogConfig, cc.LogConfig)
}
