// +build !windows

package config

import (
	"testing"

	"github.com/docker/docker/api/types"
)

func TestCommonUnixValidateConfigurationErrors(t *testing.T) {
	testCases := []struct {
		config *Config
	}{
		// Can't override the stock runtime
		{
			config: &Config{
				CommonUnixConfig: CommonUnixConfig{
					Runtimes: map[string]types.Runtime{
						StockRuntimeName: {},
					},
				},
			},
		},
		// Default runtime should be present in runtimes
		{
			config: &Config{
				CommonUnixConfig: CommonUnixConfig{
					Runtimes: map[string]types.Runtime{
						"foo": {},
					},
					DefaultRuntime: "bar",
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

func TestCommonUnixGetInitPath(t *testing.T) {
	testCases := []struct {
		config           *Config
		expectedInitPath string
	}{
		{
			config: &Config{
				InitPath: "some-init-path",
			},
			expectedInitPath: "some-init-path",
		},
		{
			config: &Config{
				CommonUnixConfig: CommonUnixConfig{
					DefaultInitBinary: "foo-init-bin",
				},
			},
			expectedInitPath: "foo-init-bin",
		},
		{
			config: &Config{
				InitPath: "init-path-A",
				CommonUnixConfig: CommonUnixConfig{
					DefaultInitBinary: "init-path-B",
				},
			},
			expectedInitPath: "init-path-A",
		},
		{
			config:           &Config{},
			expectedInitPath: "docker-init",
		},
	}
	for _, tc := range testCases {
		initPath := tc.config.GetInitPath()
		if initPath != tc.expectedInitPath {
			t.Fatalf("expected initPath to be %v, got %v", tc.expectedInitPath, initPath)
		}
	}
}
