// +build go1.7

package session

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/internal/sdktesting"
)

func TestSession_loadCSMConfig(t *testing.T) {
	defConfigFiles := []string{
		filepath.Join("testdata", "csm_shared_config"),
	}
	cases := map[string]struct {
		Envs        map[string]string
		ConfigFiles []string
		CSMProfile  string

		Expect csmConfig
		Err    string
	}{
		"no config": {
			Envs:        map[string]string{},
			Expect:      csmConfig{},
			ConfigFiles: defConfigFiles,
			CSMProfile:  "aws_csm_empty",
		},
		"env enabled": {
			Envs: map[string]string{
				"AWS_CSM_ENABLED":   "true",
				"AWS_CSM_PORT":      "4321",
				"AWS_CSM_HOST":      "ahost",
				"AWS_CSM_CLIENT_ID": "client id",
			},
			Expect: csmConfig{
				Enabled:  true,
				Port:     "4321",
				Host:     "ahost",
				ClientID: "client id",
			},
		},
		"shared cfg enabled": {
			ConfigFiles: defConfigFiles,
			Expect: csmConfig{
				Enabled:  true,
				Port:     "1234",
				Host:     "bar",
				ClientID: "foo",
			},
		},
		"mixed cfg, use env": {
			Envs: map[string]string{
				"AWS_CSM_ENABLED": "true",
			},
			ConfigFiles: defConfigFiles,
			Expect: csmConfig{
				Enabled: true,
			},
		},
		"mixed cfg, use env disabled": {
			Envs: map[string]string{
				"AWS_CSM_ENABLED": "false",
			},
			ConfigFiles: defConfigFiles,
			Expect: csmConfig{
				Enabled: false,
			},
		},
		"mixed cfg, use shared config": {
			Envs: map[string]string{
				"AWS_CSM_PORT": "4321",
			},
			ConfigFiles: defConfigFiles,
			Expect: csmConfig{
				Enabled:  true,
				Port:     "1234",
				Host:     "bar",
				ClientID: "foo",
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			restoreFn := sdktesting.StashEnv()
			defer restoreFn()

			if len(c.CSMProfile) != 0 {
				csmProfile := csmProfileName
				defer func() {
					csmProfileName = csmProfile
				}()
				csmProfileName = c.CSMProfile
			}

			for name, v := range c.Envs {
				os.Setenv(name, v)
			}

			envCfg, err := loadEnvConfig()
			if err != nil {
				t.Fatalf("failed to load the envcfg, %v", err)
			}
			csmCfg, err := loadCSMConfig(envCfg, c.ConfigFiles)
			if len(c.Err) != 0 {
				if err == nil {
					t.Fatalf("expect error, got none")
				}
				if e, a := c.Err, err.Error(); !strings.Contains(a, e) {
					t.Errorf("expect %v in error %v", e, a)
				}
				return
			}

			if e, a := c.Expect, csmCfg; e != a {
				t.Errorf("expect %v CSM config got %v", e, a)
			}
		})
	}
}
