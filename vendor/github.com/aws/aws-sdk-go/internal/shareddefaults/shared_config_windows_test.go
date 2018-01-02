package shareddefaults_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/internal/shareddefaults"
)

func TestSharedCredsFilename(t *testing.T) {
	env := awstesting.StashEnv()
	defer awstesting.PopEnv(env)

	os.Setenv("HOME", "home_dir")
	os.Setenv("USERPROFILE", "profile_dir")

	expect := filepath.Join("profile_dir", ".aws", "credentials")

	name := shareddefaults.SharedCredentialsFilename()
	if e, a := expect, name; e != a {
		t.Errorf("expect %q shared creds filename, got %q", e, a)
	}
}

func TestSharedConfigFilename(t *testing.T) {
	env := awstesting.StashEnv()
	defer awstesting.PopEnv(env)

	os.Setenv("HOME", "home_dir")
	os.Setenv("USERPROFILE", "profile_dir")

	expect := filepath.Join("profile_dir", ".aws", "config")

	name := shareddefaults.SharedConfigFilename()
	if e, a := expect, name; e != a {
		t.Errorf("expect %q shared config filename, got %q", e, a)
	}
}
