package session

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/go-ini/ini"
	"github.com/stretchr/testify/assert"
)

var (
	testConfigFilename      = filepath.Join("testdata", "shared_config")
	testConfigOtherFilename = filepath.Join("testdata", "shared_config_other")
)

func TestLoadSharedConfig(t *testing.T) {
	cases := []struct {
		Filenames []string
		Profile   string
		Expected  sharedConfig
		Err       error
	}{
		{
			Filenames: []string{"file_not_exists"},
			Profile:   "default",
		},
		{
			Filenames: []string{testConfigFilename},
			Expected: sharedConfig{
				Region: "default_region",
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "config_file_load_order",
			Expected: sharedConfig{
				Region: "shared_config_region",
				Creds: credentials.Value{
					AccessKeyID:     "shared_config_akid",
					SecretAccessKey: "shared_config_secret",
					ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
				},
			},
		},
		{
			Filenames: []string{testConfigFilename, testConfigOtherFilename},
			Profile:   "config_file_load_order",
			Expected: sharedConfig{
				Region: "shared_config_other_region",
				Creds: credentials.Value{
					AccessKeyID:     "shared_config_other_akid",
					SecretAccessKey: "shared_config_other_secret",
					ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigOtherFilename),
				},
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "assume_role",
			Expected: sharedConfig{
				AssumeRole: assumeRoleConfig{
					RoleARN:       "assume_role_role_arn",
					SourceProfile: "complete_creds",
				},
				AssumeRoleSource: &sharedConfig{
					Creds: credentials.Value{
						AccessKeyID:     "complete_creds_akid",
						SecretAccessKey: "complete_creds_secret",
						ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
					},
				},
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "assume_role_invalid_source_profile",
			Expected: sharedConfig{
				AssumeRole: assumeRoleConfig{
					RoleARN:       "assume_role_invalid_source_profile_role_arn",
					SourceProfile: "profile_not_exists",
				},
			},
			Err: SharedConfigAssumeRoleError{RoleARN: "assume_role_invalid_source_profile_role_arn"},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "assume_role_w_creds",
			Expected: sharedConfig{
				Creds: credentials.Value{
					AccessKeyID:     "assume_role_w_creds_akid",
					SecretAccessKey: "assume_role_w_creds_secret",
					ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
				},
				AssumeRole: assumeRoleConfig{
					RoleARN:         "assume_role_w_creds_role_arn",
					SourceProfile:   "assume_role_w_creds",
					ExternalID:      "1234",
					RoleSessionName: "assume_role_w_creds_session_name",
				},
				AssumeRoleSource: &sharedConfig{
					Creds: credentials.Value{
						AccessKeyID:     "assume_role_w_creds_akid",
						SecretAccessKey: "assume_role_w_creds_secret",
						ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
					},
				},
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "assume_role_wo_creds",
			Expected: sharedConfig{
				AssumeRole: assumeRoleConfig{
					RoleARN:       "assume_role_wo_creds_role_arn",
					SourceProfile: "assume_role_wo_creds",
				},
			},
			Err: SharedConfigAssumeRoleError{RoleARN: "assume_role_wo_creds_role_arn"},
		},
		{
			Filenames: []string{filepath.Join("testdata", "shared_config_invalid_ini")},
			Profile:   "profile_name",
			Err:       SharedConfigLoadError{Filename: filepath.Join("testdata", "shared_config_invalid_ini")},
		},
	}

	for i, c := range cases {
		cfg, err := loadSharedConfig(c.Profile, c.Filenames)
		if c.Err != nil {
			assert.Contains(t, err.Error(), c.Err.Error(), "expected error, %d", i)
			continue
		}

		assert.NoError(t, err, "unexpected error, %d", i)
		assert.Equal(t, c.Expected, cfg, "not equal, %d", i)
	}
}

func TestLoadSharedConfigFromFile(t *testing.T) {
	filename := testConfigFilename
	f, err := ini.Load(filename)
	if err != nil {
		t.Fatalf("failed to load test config file, %s, %v", filename, err)
	}
	iniFile := sharedConfigFile{IniData: f, Filename: filename}

	cases := []struct {
		Profile  string
		Expected sharedConfig
		Err      error
	}{
		{
			Profile:  "default",
			Expected: sharedConfig{Region: "default_region"},
		},
		{
			Profile:  "alt_profile_name",
			Expected: sharedConfig{Region: "alt_profile_name_region"},
		},
		{
			Profile:  "short_profile_name_first",
			Expected: sharedConfig{Region: "short_profile_name_first_short"},
		},
		{
			Profile:  "partial_creds",
			Expected: sharedConfig{},
		},
		{
			Profile: "complete_creds",
			Expected: sharedConfig{
				Creds: credentials.Value{
					AccessKeyID:     "complete_creds_akid",
					SecretAccessKey: "complete_creds_secret",
					ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
				},
			},
		},
		{
			Profile: "complete_creds_with_token",
			Expected: sharedConfig{
				Creds: credentials.Value{
					AccessKeyID:     "complete_creds_with_token_akid",
					SecretAccessKey: "complete_creds_with_token_secret",
					SessionToken:    "complete_creds_with_token_token",
					ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
				},
			},
		},
		{
			Profile: "full_profile",
			Expected: sharedConfig{
				Creds: credentials.Value{
					AccessKeyID:     "full_profile_akid",
					SecretAccessKey: "full_profile_secret",
					ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
				},
				Region: "full_profile_region",
			},
		},
		{
			Profile:  "partial_assume_role",
			Expected: sharedConfig{},
		},
		{
			Profile: "assume_role",
			Expected: sharedConfig{
				AssumeRole: assumeRoleConfig{
					RoleARN:       "assume_role_role_arn",
					SourceProfile: "complete_creds",
				},
			},
		},
		{
			Profile: "assume_role_w_mfa",
			Expected: sharedConfig{
				AssumeRole: assumeRoleConfig{
					RoleARN:       "assume_role_role_arn",
					SourceProfile: "complete_creds",
					MFASerial:     "0123456789",
				},
			},
		},
		{
			Profile: "does_not_exists",
			Err:     SharedConfigProfileNotExistsError{Profile: "does_not_exists"},
		},
	}

	for i, c := range cases {
		cfg := sharedConfig{}

		err := cfg.setFromIniFile(c.Profile, iniFile)
		if c.Err != nil {
			assert.Contains(t, err.Error(), c.Err.Error(), "expected error, %d", i)
			continue
		}

		assert.NoError(t, err, "unexpected error, %d", i)
		assert.Equal(t, c.Expected, cfg, "not equal, %d", i)
	}
}

func TestLoadSharedConfigIniFiles(t *testing.T) {
	cases := []struct {
		Filenames []string
		Expected  []sharedConfigFile
	}{
		{
			Filenames: []string{"not_exists", testConfigFilename},
			Expected: []sharedConfigFile{
				{Filename: testConfigFilename},
			},
		},
		{
			Filenames: []string{testConfigFilename, testConfigOtherFilename},
			Expected: []sharedConfigFile{
				{Filename: testConfigFilename},
				{Filename: testConfigOtherFilename},
			},
		},
	}

	for i, c := range cases {
		files, err := loadSharedConfigIniFiles(c.Filenames)
		assert.NoError(t, err, "unexpected error, %d", i)
		assert.Equal(t, len(c.Expected), len(files), "expected num files, %d", i)

		for i, expectedFile := range c.Expected {
			assert.Equal(t, expectedFile.Filename, files[i].Filename)
		}
	}
}
