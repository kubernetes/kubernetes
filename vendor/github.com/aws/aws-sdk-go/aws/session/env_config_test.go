package session

import (
	"os"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/awstesting"
)

func TestLoadEnvConfig_Creds(t *testing.T) {
	env := awstesting.StashEnv()
	defer awstesting.PopEnv(env)

	cases := []struct {
		Env map[string]string
		Val credentials.Value
	}{
		{
			Env: map[string]string{
				"AWS_ACCESS_KEY": "AKID",
			},
			Val: credentials.Value{},
		},
		{
			Env: map[string]string{
				"AWS_ACCESS_KEY_ID": "AKID",
			},
			Val: credentials.Value{},
		},
		{
			Env: map[string]string{
				"AWS_SECRET_KEY": "SECRET",
			},
			Val: credentials.Value{},
		},
		{
			Env: map[string]string{
				"AWS_SECRET_ACCESS_KEY": "SECRET",
			},
			Val: credentials.Value{},
		},
		{
			Env: map[string]string{
				"AWS_ACCESS_KEY_ID":     "AKID",
				"AWS_SECRET_ACCESS_KEY": "SECRET",
			},
			Val: credentials.Value{
				AccessKeyID: "AKID", SecretAccessKey: "SECRET",
				ProviderName: "EnvConfigCredentials",
			},
		},
		{
			Env: map[string]string{
				"AWS_ACCESS_KEY": "AKID",
				"AWS_SECRET_KEY": "SECRET",
			},
			Val: credentials.Value{
				AccessKeyID: "AKID", SecretAccessKey: "SECRET",
				ProviderName: "EnvConfigCredentials",
			},
		},
		{
			Env: map[string]string{
				"AWS_ACCESS_KEY":    "AKID",
				"AWS_SECRET_KEY":    "SECRET",
				"AWS_SESSION_TOKEN": "TOKEN",
			},
			Val: credentials.Value{
				AccessKeyID: "AKID", SecretAccessKey: "SECRET", SessionToken: "TOKEN",
				ProviderName: "EnvConfigCredentials",
			},
		},
	}

	for _, c := range cases {
		os.Clearenv()

		for k, v := range c.Env {
			os.Setenv(k, v)
		}

		cfg := loadEnvConfig()
		if !reflect.DeepEqual(c.Val, cfg.Creds) {
			t.Errorf("expect credentials to match.\n%s",
				awstesting.SprintExpectActual(c.Val, cfg.Creds))
		}
	}
}

func TestLoadEnvConfig(t *testing.T) {
	env := awstesting.StashEnv()
	defer awstesting.PopEnv(env)

	cases := []struct {
		Env                 map[string]string
		UseSharedConfigCall bool
		Config              envConfig
	}{
		{
			Env: map[string]string{
				"AWS_REGION":  "region",
				"AWS_PROFILE": "profile",
			},
			Config: envConfig{
				Region: "region", Profile: "profile",
			},
		},
		{
			Env: map[string]string{
				"AWS_REGION":          "region",
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_PROFILE":         "profile",
				"AWS_DEFAULT_PROFILE": "default_profile",
			},
			Config: envConfig{
				Region: "region", Profile: "profile",
			},
		},
		{
			Env: map[string]string{
				"AWS_REGION":          "region",
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_PROFILE":         "profile",
				"AWS_DEFAULT_PROFILE": "default_profile",
				"AWS_SDK_LOAD_CONFIG": "1",
			},
			Config: envConfig{
				Region: "region", Profile: "profile",
				EnableSharedConfig: true,
			},
		},
		{
			Env: map[string]string{
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_DEFAULT_PROFILE": "default_profile",
			},
		},
		{
			Env: map[string]string{
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_DEFAULT_PROFILE": "default_profile",
				"AWS_SDK_LOAD_CONFIG": "1",
			},
			Config: envConfig{
				Region: "default_region", Profile: "default_profile",
				EnableSharedConfig: true,
			},
		},
		{
			Env: map[string]string{
				"AWS_REGION":  "region",
				"AWS_PROFILE": "profile",
			},
			Config: envConfig{
				Region: "region", Profile: "profile",
				EnableSharedConfig: true,
			},
			UseSharedConfigCall: true,
		},
		{
			Env: map[string]string{
				"AWS_REGION":          "region",
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_PROFILE":         "profile",
				"AWS_DEFAULT_PROFILE": "default_profile",
			},
			Config: envConfig{
				Region: "region", Profile: "profile",
				EnableSharedConfig: true,
			},
			UseSharedConfigCall: true,
		},
		{
			Env: map[string]string{
				"AWS_REGION":          "region",
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_PROFILE":         "profile",
				"AWS_DEFAULT_PROFILE": "default_profile",
				"AWS_SDK_LOAD_CONFIG": "1",
			},
			Config: envConfig{
				Region: "region", Profile: "profile",
				EnableSharedConfig: true,
			},
			UseSharedConfigCall: true,
		},
		{
			Env: map[string]string{
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_DEFAULT_PROFILE": "default_profile",
			},
			Config: envConfig{
				Region: "default_region", Profile: "default_profile",
				EnableSharedConfig: true,
			},
			UseSharedConfigCall: true,
		},
		{
			Env: map[string]string{
				"AWS_DEFAULT_REGION":  "default_region",
				"AWS_DEFAULT_PROFILE": "default_profile",
				"AWS_SDK_LOAD_CONFIG": "1",
			},
			Config: envConfig{
				Region: "default_region", Profile: "default_profile",
				EnableSharedConfig: true,
			},
			UseSharedConfigCall: true,
		},
		{
			Env: map[string]string{
				"AWS_CA_BUNDLE": "custom_ca_bundle",
			},
			Config: envConfig{
				CustomCABundle: "custom_ca_bundle",
			},
		},
		{
			Env: map[string]string{
				"AWS_CA_BUNDLE": "custom_ca_bundle",
			},
			Config: envConfig{
				CustomCABundle:     "custom_ca_bundle",
				EnableSharedConfig: true,
			},
			UseSharedConfigCall: true,
		},
		{
			Env: map[string]string{
				"AWS_SHARED_CREDENTIALS_FILE": "/path/to/credentials/file",
				"AWS_CONFIG_FILE":             "/path/to/config/file",
			},
			Config: envConfig{
				SharedCredentialsFile: "/path/to/credentials/file",
				SharedConfigFile:      "/path/to/config/file",
			},
		},
	}

	for _, c := range cases {
		os.Clearenv()

		for k, v := range c.Env {
			os.Setenv(k, v)
		}

		var cfg envConfig
		if c.UseSharedConfigCall {
			cfg = loadSharedEnvConfig()
		} else {
			cfg = loadEnvConfig()
		}

		if !reflect.DeepEqual(c.Config, cfg) {
			t.Errorf("expect config to match.\n%s",
				awstesting.SprintExpectActual(c.Config, cfg))
		}
	}
}

func TestSetEnvValue(t *testing.T) {
	env := awstesting.StashEnv()
	defer awstesting.PopEnv(env)

	os.Setenv("empty_key", "")
	os.Setenv("second_key", "2")
	os.Setenv("third_key", "3")

	var dst string
	setFromEnvVal(&dst, []string{
		"empty_key", "first_key", "second_key", "third_key",
	})

	if e, a := "2", dst; e != a {
		t.Errorf("expect %s value from environment, got %s", e, a)
	}
}
