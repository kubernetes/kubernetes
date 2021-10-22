// +build go1.7

package session

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/internal/ini"
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
			Expected:  sharedConfig{},
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
				RoleARN:           "assume_role_role_arn",
				SourceProfileName: "complete_creds",
				SourceProfile: &sharedConfig{
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
				RoleARN:           "assume_role_invalid_source_profile_role_arn",
				SourceProfileName: "profile_not_exists",
			},
			Err: SharedConfigAssumeRoleError{
				RoleARN:       "assume_role_invalid_source_profile_role_arn",
				SourceProfile: "profile_not_exists",
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "assume_role_w_creds",
			Expected: sharedConfig{
				RoleARN:           "assume_role_w_creds_role_arn",
				ExternalID:        "1234",
				RoleSessionName:   "assume_role_w_creds_session_name",
				SourceProfileName: "assume_role_w_creds",
				SourceProfile: &sharedConfig{
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
				RoleARN:           "assume_role_wo_creds_role_arn",
				SourceProfileName: "assume_role_wo_creds",
			},
			Err: SharedConfigAssumeRoleError{
				RoleARN:       "assume_role_wo_creds_role_arn",
				SourceProfile: "assume_role_wo_creds",
			},
		},
		{
			Filenames: []string{filepath.Join("testdata", "shared_config_invalid_ini")},
			Profile:   "profile_name",
			Err:       SharedConfigLoadError{Filename: filepath.Join("testdata", "shared_config_invalid_ini")},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "assume_role_with_credential_source",
			Expected: sharedConfig{
				RoleARN:          "assume_role_with_credential_source_role_arn",
				CredentialSource: credSourceEc2Metadata,
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "multiple_assume_role",
			Expected: sharedConfig{
				RoleARN:           "multiple_assume_role_role_arn",
				SourceProfileName: "assume_role",
				SourceProfile: &sharedConfig{
					RoleARN:           "assume_role_role_arn",
					SourceProfileName: "complete_creds",
					SourceProfile: &sharedConfig{
						Creds: credentials.Value{
							AccessKeyID:     "complete_creds_akid",
							SecretAccessKey: "complete_creds_secret",
							ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", testConfigFilename),
						},
					},
				},
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "multiple_assume_role_with_credential_source",
			Expected: sharedConfig{
				RoleARN:           "multiple_assume_role_with_credential_source_role_arn",
				SourceProfileName: "assume_role_with_credential_source",
				SourceProfile: &sharedConfig{
					RoleARN:          "assume_role_with_credential_source_role_arn",
					CredentialSource: credSourceEc2Metadata,
				},
			},
		},
		{
			Filenames: []string{testConfigOtherFilename, testConfigFilename},
			Profile:   "multiple_assume_role_with_credential_source2",
			Expected: sharedConfig{
				RoleARN:           "multiple_assume_role_with_credential_source2_role_arn",
				SourceProfileName: "multiple_assume_role_with_credential_source",
				SourceProfile: &sharedConfig{
					RoleARN:           "multiple_assume_role_with_credential_source_role_arn",
					SourceProfileName: "assume_role_with_credential_source",
					SourceProfile: &sharedConfig{
						RoleARN:          "assume_role_with_credential_source_role_arn",
						CredentialSource: credSourceEc2Metadata,
					},
				},
			},
		},
		{
			Filenames: []string{testConfigFilename},
			Profile:   "with_sts_regional",
			Expected: sharedConfig{
				STSRegionalEndpoint: endpoints.RegionalSTSEndpoint,
			},
		},
		{
			Filenames: []string{testConfigFilename},
			Profile:   "with_s3_us_east_1_regional",
			Expected: sharedConfig{
				S3UsEast1RegionalEndpoint: endpoints.RegionalS3UsEast1Endpoint,
			},
		},
	}

	for i, c := range cases {
		t.Run(strconv.Itoa(i)+"_"+c.Profile, func(t *testing.T) {
			cfg, err := loadSharedConfig(c.Profile, c.Filenames, true)
			if c.Err != nil {
				if err == nil {
					t.Fatalf("expect error, got none")
				}
				if e, a := c.Err.Error(), err.Error(); !strings.Contains(a, e) {
					t.Errorf("expect %v, to be in %v", e, a)
				}
				return
			}

			if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}
			if e, a := c.Expected, cfg; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestLoadSharedConfigFromFile(t *testing.T) {
	filename := testConfigFilename
	f, err := ini.OpenFile(filename)
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
			Profile: "partial_assume_role",
			Expected: sharedConfig{
				RoleARN: "partial_assume_role_role_arn",
			},
		},
		{
			Profile: "assume_role",
			Expected: sharedConfig{
				RoleARN:           "assume_role_role_arn",
				SourceProfileName: "complete_creds",
			},
		},
		{
			Profile: "assume_role_w_mfa",
			Expected: sharedConfig{
				RoleARN:           "assume_role_role_arn",
				SourceProfileName: "complete_creds",
				MFASerial:         "0123456789",
			},
		},
		{
			Profile: "does_not_exists",
			Err:     SharedConfigProfileNotExistsError{Profile: "does_not_exists"},
		},
		{
			Profile: "valid_arn_region",
			Expected: sharedConfig{
				S3UseARNRegion: true,
			},
		},
	}

	for i, c := range cases {
		t.Run(strconv.Itoa(i)+"_"+c.Profile, func(t *testing.T) {
			cfg := sharedConfig{}

			err := cfg.setFromIniFile(c.Profile, iniFile, true)
			if c.Err != nil {
				if err == nil {
					t.Fatalf("expect error, got none")
				}
				if e, a := c.Err.Error(), err.Error(); !strings.Contains(a, e) {
					t.Errorf("expect %v, to be in %v", e, a)
				}
				return
			}

			if err != nil {
				t.Errorf("expect no error, got %v", err)
			}
			if e, a := c.Expected, cfg; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
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
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			files, err := loadSharedConfigIniFiles(c.Filenames)
			if err != nil {
				t.Fatalf("expect no error, got %v", err)
			}
			if e, a := len(c.Expected), len(files); e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			for i, expectedFile := range c.Expected {
				if e, a := expectedFile.Filename, files[i].Filename; e != a {
					t.Errorf("expect %v, got %v", e, a)
				}
			}
		})
	}
}
