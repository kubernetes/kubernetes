package config

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

var expiry = 1 * time.Minute

var invalidProfileConfig = &Config{
	Signing: &Signing{
		Profiles: map[string]*SigningProfile{
			"invalid": {
				Usage:  []string{"wiretapping"},
				Expiry: expiry,
			},
			"empty": {},
		},
		Default: &SigningProfile{
			Usage:  []string{"digital signature"},
			Expiry: expiry,
		},
	},
}

var invalidDefaultConfig = &Config{
	Signing: &Signing{
		Profiles: map[string]*SigningProfile{
			"key usage": {
				Usage: []string{"digital signature"},
			},
		},
		Default: &SigningProfile{
			Usage: []string{"s/mime"},
		},
	},
}

var validConfig = &Config{
	Signing: &Signing{
		Profiles: map[string]*SigningProfile{
			"valid": {
				Usage:  []string{"digital signature"},
				Expiry: expiry,
			},
		},
		Default: &SigningProfile{
			Usage:  []string{"digital signature"},
			Expiry: expiry,
		},
	},
}

var validMixedConfig = `
{
	"signing": {
		"profiles": {
			"CA": {
				"auth_key": "sample",
				"remote": "localhost"
			},
			"email": {
				"usages": ["s/mime"],
				"expiry": "720h"
			}
		},
		"default": {
			"usages": ["digital signature", "email protection"],
			"expiry": "8000h"
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	},
	"remotes": {
		"localhost": "127.0.0.1:8888"
	}
}`

var validMinimalRemoteConfig = `
{
	"signing": {
		"default": {
			"auth_key": "sample",
			"remote": "localhost"
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	},
	"remotes": {
		"localhost": "127.0.0.1:8888"
	}
}`

var validMinimalRemoteConfig2 = `
{
	"signing": {
		"default": {
			"auth_remote":{
			    "auth_key": "sample",
			    "remote": "localhost"
		    }
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	},
	"remotes": {
		"localhost": "127.0.0.1:8888"
	}
}`

var invalidRemoteConfig = `
{
	"signing": {
		"default": {
			"auth_remotes_typos":{
			    "auth_key": "sample",
			    "remote": "localhost"
		    }
		}
	},
	"auth_keys": {
		"sample": {
			"type":"standard",
			"key":"0123456789ABCDEF0123456789ABCDEF"
		}
	},
	"remotes": {
		"localhost": "127.0.0.1:8888"
	}
}`

var validMinimalLocalConfig = `
{
	"signing": {
		"default": {
			"usages": ["digital signature", "email protection"],
			"expiry": "8000h"
		}
	}
}`

func TestInvalidProfile(t *testing.T) {
	if invalidProfileConfig.Signing.Profiles["invalid"].validProfile(false) {
		t.Fatal("invalid profile accepted as valid")
	}

	if invalidProfileConfig.Signing.Profiles["empty"].validProfile(false) {
		t.Fatal("invalid profile accepted as valid")
	}

	if invalidProfileConfig.Valid() {
		t.Fatal("invalid config accepted as valid")
	}

	if !invalidProfileConfig.Signing.Profiles["invalid"].validProfile(true) {
		t.Fatal("invalid profile should be a valid default profile")
	}
}

func TestRemoteProfiles(t *testing.T) {
	var validRemoteProfile = &SigningProfile{
		RemoteName:   "localhost",
		RemoteServer: "localhost:8080",
	}

	var invalidRemoteProfile = &SigningProfile{
		RemoteName: "localhost",
	}

	var invalidRemoteAuthProfile = &SigningProfile{
		RemoteName:   "localhost",
		RemoteServer: "localhost:8080",
		AuthKeyName:  "blahblah",
	}

	if !validRemoteProfile.validProfile(true) ||
		!validRemoteProfile.validProfile(false) {
		t.Fatal("valid remote profile is rejected.")
	}

	if invalidRemoteProfile.validProfile(true) ||
		invalidRemoteProfile.validProfile(false) {
		t.Fatal("invalid remote profile is accepted.")
	}

	if invalidRemoteAuthProfile.validProfile(true) ||
		invalidRemoteAuthProfile.validProfile(false) {
		t.Fatal("invalid remote profile is accepted.")
	}
}

func TestInvalidDefault(t *testing.T) {
	if invalidDefaultConfig.Signing.Default.validProfile(true) {
		t.Fatal("invalid default accepted as valid")
	}

	if invalidDefaultConfig.Valid() {
		t.Fatal("invalid config accepted as valid")
	}

	if !invalidDefaultConfig.Signing.Default.validProfile(false) {
		t.Fatal("invalid default profile should be a valid profile")
	}
}

func TestValidConfig(t *testing.T) {
	if !validConfig.Valid() {
		t.Fatal("Valid config is not valid")
	}
	bytes, _ := json.Marshal(validConfig)
	fmt.Printf("%v", string(bytes))
}

func TestDefaultConfig(t *testing.T) {
	if !DefaultConfig().validProfile(false) {
		t.Fatal("global default signing profile should be a valid profile.")
	}

	if !DefaultConfig().validProfile(true) {
		t.Fatal("global default signing profile should be a valid default profile")
	}
}

func TestParse(t *testing.T) {
	var validProfiles = []*SigningProfile{
		{
			ExpiryString: "8760h",
		},
		{
			ExpiryString: "168h",
		},
		{
			ExpiryString: "300s",
		},
	}

	var invalidProfiles = []*SigningProfile{
		nil,
		{},
		{
			ExpiryString: "",
		},
		{
			ExpiryString: "365d",
		},
		{
			ExpiryString: "1y",
		},
		{
			ExpiryString: "one year",
		},
	}

	for _, p := range validProfiles {
		if p.populate(nil) != nil {
			t.Fatalf("Failed to parse ExpiryString=%s", p.ExpiryString)
		}
	}

	for _, p := range invalidProfiles {
		if p.populate(nil) == nil {
			if p != nil {
				t.Fatalf("ExpiryString=%s should not be parseable", p.ExpiryString)
			}
			t.Fatalf("Nil profile should not be parseable")
		}
	}

}

func TestLoadFile(t *testing.T) {
	validConfigFiles := []string{
		"testdata/valid_config.json",
		"testdata/valid_config_auth.json",
		"testdata/valid_config_no_default.json",
		"testdata/valid_config_auth_no_default.json",
	}

	for _, configFile := range validConfigFiles {
		_, err := LoadFile(configFile)
		if err != nil {
			t.Fatal("Load valid config file failed.", configFile, "error is ", err)
		}
	}
}

func TestLoadInvalidConfigFile(t *testing.T) {
	invalidConfigFiles := []string{"", "testdata/no_such_file",
		"testdata/invalid_default.json",
		"testdata/invalid_profiles.json",
		"testdata/invalid_usage.json",
		"testdata/invalid_config.json",
		"testdata/invalid_auth.json",
		"testdata/invalid_auth_bad_key.json",
		"testdata/invalid_no_auth_keys.json",
		"testdata/invalid_remote.json",
		"testdata/invalid_no_remotes.json",
	}
	for _, configFile := range invalidConfigFiles {
		_, err := LoadFile(configFile)
		if err == nil {
			t.Fatal("Invalid config is loaded.", configFile)
		}
	}
}

func TestNeedLocalSigner(t *testing.T) {

	c, err := LoadConfig([]byte(validMixedConfig))
	if err != nil {
		t.Fatal("load valid config failed:", err)
	}

	// This signing config needs both local signer and remote signer.
	if c.Signing.NeedsLocalSigner() != true {
		t.Fatal("incorrect NeedsLocalSigner().")
	}

	if c.Signing.NeedsRemoteSigner() != true {
		t.Fatal("incorrect NeedsRemoteSigner()")
	}

	remoteConfig, err := LoadConfig([]byte(validMinimalRemoteConfig))
	if err != nil {
		t.Fatal("Load valid config failed:", err)
	}

	if remoteConfig.Signing.NeedsLocalSigner() != false {
		t.Fatal("incorrect NeedsLocalSigner().")
	}

	if remoteConfig.Signing.NeedsRemoteSigner() != true {
		t.Fatal("incorrect NeedsRemoteSigner().")
	}

	localConfig, err := LoadConfig([]byte(validMinimalLocalConfig))
	if localConfig.Signing.NeedsLocalSigner() != true {
		t.Fatal("incorrect NeedsLocalSigner().")
	}

	if localConfig.Signing.NeedsRemoteSigner() != false {
		t.Fatal("incorrect NeedsRemoteSigner().")
	}
}

func TestOverrideRemotes(t *testing.T) {
	c, err := LoadConfig([]byte(validMixedConfig))
	if err != nil {
		t.Fatal("load valid config failed:", err)
	}

	host := "localhost:8888"
	c.Signing.OverrideRemotes(host)

	if c.Signing.Default.RemoteServer != host {
		t.Fatal("should override default profile's RemoteServer")
	}

	for _, p := range c.Signing.Profiles {
		if p.RemoteServer != host {
			t.Fatal("failed to override profile's RemoteServer")
		}
	}

}

func TestAuthRemoteConfig(t *testing.T) {
	c, err := LoadConfig([]byte(validMinimalRemoteConfig2))
	if err != nil {
		t.Fatal("load valid config failed:", err)
	}

	if c.Signing.Default.RemoteServer != "127.0.0.1:8888" {
		t.Fatal("load valid config failed: incorrect remote server")
	}

	host := "localhost:8888"
	c.Signing.OverrideRemotes(host)

	if c.Signing.Default.RemoteServer != host {
		t.Fatal("should override default profile's RemoteServer")
	}

	for _, p := range c.Signing.Profiles {
		if p.RemoteServer != host {
			t.Fatal("failed to override profile's RemoteServer")
		}
	}
}

func TestBadAuthRemoteConfig(t *testing.T) {
	_, err := LoadConfig([]byte(invalidRemoteConfig))
	if err == nil {
		t.Fatal("load invalid config should failed")
	}
}
