package cliconfig

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/docker/docker/pkg/homedir"
)

func TestMissingFile(t *testing.T) {
	tmpHome, _ := ioutil.TempDir("", "config-test")

	config, err := Load(tmpHome)
	if err != nil {
		t.Fatalf("Failed loading on missing file: %q", err)
	}

	// Now save it and make sure it shows up in new form
	err = config.Save()
	if err != nil {
		t.Fatalf("Failed to save: %q", err)
	}

	buf, err := ioutil.ReadFile(filepath.Join(tmpHome, ConfigFileName))
	if !strings.Contains(string(buf), `"auths":`) {
		t.Fatalf("Should have save in new form: %s", string(buf))
	}
}

func TestSaveFileToDirs(t *testing.T) {
	tmpHome, _ := ioutil.TempDir("", "config-test")

	tmpHome += "/.docker"

	config, err := Load(tmpHome)
	if err != nil {
		t.Fatalf("Failed loading on missing file: %q", err)
	}

	// Now save it and make sure it shows up in new form
	err = config.Save()
	if err != nil {
		t.Fatalf("Failed to save: %q", err)
	}

	buf, err := ioutil.ReadFile(filepath.Join(tmpHome, ConfigFileName))
	if !strings.Contains(string(buf), `"auths":`) {
		t.Fatalf("Should have save in new form: %s", string(buf))
	}
}

func TestEmptyFile(t *testing.T) {
	tmpHome, _ := ioutil.TempDir("", "config-test")
	fn := filepath.Join(tmpHome, ConfigFileName)
	ioutil.WriteFile(fn, []byte(""), 0600)

	_, err := Load(tmpHome)
	if err == nil {
		t.Fatalf("Was supposed to fail")
	}
}

func TestEmptyJson(t *testing.T) {
	tmpHome, _ := ioutil.TempDir("", "config-test")
	fn := filepath.Join(tmpHome, ConfigFileName)
	ioutil.WriteFile(fn, []byte("{}"), 0600)

	config, err := Load(tmpHome)
	if err != nil {
		t.Fatalf("Failed loading on empty json file: %q", err)
	}

	// Now save it and make sure it shows up in new form
	err = config.Save()
	if err != nil {
		t.Fatalf("Failed to save: %q", err)
	}

	buf, err := ioutil.ReadFile(filepath.Join(tmpHome, ConfigFileName))
	if !strings.Contains(string(buf), `"auths":`) {
		t.Fatalf("Should have save in new form: %s", string(buf))
	}
}

func TestOldJson(t *testing.T) {
	if runtime.GOOS == "windows" {
		return
	}

	tmpHome, _ := ioutil.TempDir("", "config-test")
	defer os.RemoveAll(tmpHome)

	homeKey := homedir.Key()
	homeVal := homedir.Get()

	defer func() { os.Setenv(homeKey, homeVal) }()
	os.Setenv(homeKey, tmpHome)

	fn := filepath.Join(tmpHome, oldConfigfile)
	js := `{"https://index.docker.io/v1/":{"auth":"am9lam9lOmhlbGxv","email":"user@example.com"}}`
	ioutil.WriteFile(fn, []byte(js), 0600)

	config, err := Load(tmpHome)
	if err != nil {
		t.Fatalf("Failed loading on empty json file: %q", err)
	}

	ac := config.AuthConfigs["https://index.docker.io/v1/"]
	if ac.Email != "user@example.com" || ac.Username != "joejoe" || ac.Password != "hello" {
		t.Fatalf("Missing data from parsing:\n%q", config)
	}

	// Now save it and make sure it shows up in new form
	err = config.Save()
	if err != nil {
		t.Fatalf("Failed to save: %q", err)
	}

	buf, err := ioutil.ReadFile(filepath.Join(tmpHome, ConfigFileName))
	if !strings.Contains(string(buf), `"auths":`) ||
		!strings.Contains(string(buf), "user@example.com") {
		t.Fatalf("Should have save in new form: %s", string(buf))
	}
}

func TestNewJson(t *testing.T) {
	tmpHome, _ := ioutil.TempDir("", "config-test")
	fn := filepath.Join(tmpHome, ConfigFileName)
	js := ` { "auths": { "https://index.docker.io/v1/": { "auth": "am9lam9lOmhlbGxv", "email": "user@example.com" } } }`
	ioutil.WriteFile(fn, []byte(js), 0600)

	config, err := Load(tmpHome)
	if err != nil {
		t.Fatalf("Failed loading on empty json file: %q", err)
	}

	ac := config.AuthConfigs["https://index.docker.io/v1/"]
	if ac.Email != "user@example.com" || ac.Username != "joejoe" || ac.Password != "hello" {
		t.Fatalf("Missing data from parsing:\n%q", config)
	}

	// Now save it and make sure it shows up in new form
	err = config.Save()
	if err != nil {
		t.Fatalf("Failed to save: %q", err)
	}

	buf, err := ioutil.ReadFile(filepath.Join(tmpHome, ConfigFileName))
	if !strings.Contains(string(buf), `"auths":`) ||
		!strings.Contains(string(buf), "user@example.com") {
		t.Fatalf("Should have save in new form: %s", string(buf))
	}
}
