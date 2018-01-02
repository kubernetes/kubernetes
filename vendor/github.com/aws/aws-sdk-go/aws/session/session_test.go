package session

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/service/s3"
)

func TestNewDefaultSession(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	s := New(&aws.Config{Region: aws.String("region")})

	assert.Equal(t, "region", *s.Config.Region)
	assert.Equal(t, http.DefaultClient, s.Config.HTTPClient)
	assert.NotNil(t, s.Config.Logger)
	assert.Equal(t, aws.LogOff, *s.Config.LogLevel)
}

func TestNew_WithCustomCreds(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	customCreds := credentials.NewStaticCredentials("AKID", "SECRET", "TOKEN")
	s := New(&aws.Config{Credentials: customCreds})

	assert.Equal(t, customCreds, s.Config.Credentials)
}

type mockLogger struct {
	*bytes.Buffer
}

func (w mockLogger) Log(args ...interface{}) {
	fmt.Fprintln(w, args...)
}

func TestNew_WithSessionLoadError(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_CONFIG_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_invalid_source_profile")

	logger := bytes.Buffer{}
	s := New(&aws.Config{Logger: &mockLogger{&logger}})

	assert.NotNil(t, s)

	svc := s3.New(s)
	_, err := svc.ListBuckets(&s3.ListBucketsInput{})

	assert.Error(t, err)
	assert.Contains(t, logger.String(), "ERROR: failed to create session with AWS_SDK_LOAD_CONFIG enabled")
	assert.Contains(t, err.Error(), SharedConfigAssumeRoleError{
		RoleARN: "assume_role_invalid_source_profile_role_arn",
	}.Error())
}

func TestSessionCopy(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_REGION", "orig_region")

	s := Session{
		Config:   defaults.Config(),
		Handlers: defaults.Handlers(),
	}

	newSess := s.Copy(&aws.Config{Region: aws.String("new_region")})

	assert.Equal(t, "orig_region", *s.Config.Region)
	assert.Equal(t, "new_region", *newSess.Config.Region)
}

func TestSessionClientConfig(t *testing.T) {
	s, err := NewSession(&aws.Config{Region: aws.String("orig_region")})
	assert.NoError(t, err)

	cfg := s.ClientConfig("s3", &aws.Config{Region: aws.String("us-west-2")})

	assert.Equal(t, "https://s3-us-west-2.amazonaws.com", cfg.Endpoint)
	assert.Equal(t, "us-west-2", cfg.SigningRegion)
	assert.Equal(t, "us-west-2", *cfg.Config.Region)
}

func TestNewSession_NoCredentials(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	s, err := NewSession()
	assert.NoError(t, err)

	assert.NotNil(t, s.Config.Credentials)
	assert.NotEqual(t, credentials.AnonymousCredentials, s.Config.Credentials)
}

func TestNewSessionWithOptions_OverrideProfile(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "other_profile")

	s, err := NewSessionWithOptions(Options{
		Profile: "full_profile",
	})
	assert.NoError(t, err)

	assert.Equal(t, "full_profile_region", *s.Config.Region)

	creds, err := s.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.Equal(t, "full_profile_akid", creds.AccessKeyID)
	assert.Equal(t, "full_profile_secret", creds.SecretAccessKey)
	assert.Empty(t, creds.SessionToken)
	assert.Contains(t, creds.ProviderName, "SharedConfigCredentials")
}

func TestNewSessionWithOptions_OverrideSharedConfigEnable(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "0")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "full_profile")

	s, err := NewSessionWithOptions(Options{
		SharedConfigState: SharedConfigEnable,
	})
	assert.NoError(t, err)

	assert.Equal(t, "full_profile_region", *s.Config.Region)

	creds, err := s.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.Equal(t, "full_profile_akid", creds.AccessKeyID)
	assert.Equal(t, "full_profile_secret", creds.SecretAccessKey)
	assert.Empty(t, creds.SessionToken)
	assert.Contains(t, creds.ProviderName, "SharedConfigCredentials")
}

func TestNewSessionWithOptions_OverrideSharedConfigDisable(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "full_profile")

	s, err := NewSessionWithOptions(Options{
		SharedConfigState: SharedConfigDisable,
	})
	assert.NoError(t, err)

	assert.Empty(t, *s.Config.Region)

	creds, err := s.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.Equal(t, "full_profile_akid", creds.AccessKeyID)
	assert.Equal(t, "full_profile_secret", creds.SecretAccessKey)
	assert.Empty(t, creds.SessionToken)
	assert.Contains(t, creds.ProviderName, "SharedConfigCredentials")
}

func TestNewSessionWithOptions_OverrideSharedConfigFiles(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "config_file_load_order")

	s, err := NewSessionWithOptions(Options{
		SharedConfigFiles: []string{testConfigOtherFilename},
	})
	assert.NoError(t, err)

	assert.Equal(t, "shared_config_other_region", *s.Config.Region)

	creds, err := s.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.Equal(t, "shared_config_other_akid", creds.AccessKeyID)
	assert.Equal(t, "shared_config_other_secret", creds.SecretAccessKey)
	assert.Empty(t, creds.SessionToken)
	assert.Contains(t, creds.ProviderName, "SharedConfigCredentials")
}

func TestNewSessionWithOptions_Overrides(t *testing.T) {
	cases := []struct {
		InEnvs    map[string]string
		InProfile string
		OutRegion string
		OutCreds  credentials.Value
	}{
		{
			InEnvs: map[string]string{
				"AWS_SDK_LOAD_CONFIG":         "0",
				"AWS_SHARED_CREDENTIALS_FILE": testConfigFilename,
				"AWS_PROFILE":                 "other_profile",
			},
			InProfile: "full_profile",
			OutRegion: "full_profile_region",
			OutCreds: credentials.Value{
				AccessKeyID:     "full_profile_akid",
				SecretAccessKey: "full_profile_secret",
				ProviderName:    "SharedConfigCredentials",
			},
		},
		{
			InEnvs: map[string]string{
				"AWS_SDK_LOAD_CONFIG":         "0",
				"AWS_SHARED_CREDENTIALS_FILE": testConfigFilename,
				"AWS_REGION":                  "env_region",
				"AWS_ACCESS_KEY":              "env_akid",
				"AWS_SECRET_ACCESS_KEY":       "env_secret",
				"AWS_PROFILE":                 "other_profile",
			},
			InProfile: "full_profile",
			OutRegion: "env_region",
			OutCreds: credentials.Value{
				AccessKeyID:     "env_akid",
				SecretAccessKey: "env_secret",
				ProviderName:    "EnvConfigCredentials",
			},
		},
		{
			InEnvs: map[string]string{
				"AWS_SDK_LOAD_CONFIG":         "0",
				"AWS_SHARED_CREDENTIALS_FILE": testConfigFilename,
				"AWS_CONFIG_FILE":             testConfigOtherFilename,
				"AWS_PROFILE":                 "shared_profile",
			},
			InProfile: "config_file_load_order",
			OutRegion: "shared_config_region",
			OutCreds: credentials.Value{
				AccessKeyID:     "shared_config_akid",
				SecretAccessKey: "shared_config_secret",
				ProviderName:    "SharedConfigCredentials",
			},
		},
	}

	for _, c := range cases {
		oldEnv := initSessionTestEnv()
		defer awstesting.PopEnv(oldEnv)

		for k, v := range c.InEnvs {
			os.Setenv(k, v)
		}

		s, err := NewSessionWithOptions(Options{
			Profile:           c.InProfile,
			SharedConfigState: SharedConfigEnable,
		})
		assert.NoError(t, err)

		creds, err := s.Config.Credentials.Get()
		assert.NoError(t, err)
		assert.Equal(t, c.OutRegion, *s.Config.Region)
		assert.Equal(t, c.OutCreds.AccessKeyID, creds.AccessKeyID)
		assert.Equal(t, c.OutCreds.SecretAccessKey, creds.SecretAccessKey)
		assert.Equal(t, c.OutCreds.SessionToken, creds.SessionToken)
		assert.Contains(t, creds.ProviderName, c.OutCreds.ProviderName)
	}
}

const assumeRoleRespMsg = `
<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleResult>
    <AssumedRoleUser>
      <Arn>arn:aws:sts::account_id:assumed-role/role/session_name</Arn>
      <AssumedRoleId>AKID:session_name</AssumedRoleId>
    </AssumedRoleUser>
    <Credentials>
      <AccessKeyId>AKID</AccessKeyId>
      <SecretAccessKey>SECRET</SecretAccessKey>
      <SessionToken>SESSION_TOKEN</SessionToken>
      <Expiration>%s</Expiration>
    </Credentials>
  </AssumeRoleResult>
  <ResponseMetadata>
    <RequestId>request-id</RequestId>
  </ResponseMetadata>
</AssumeRoleResponse>
`

func TestSesisonAssumeRole(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(fmt.Sprintf(assumeRoleRespMsg, time.Now().Add(15*time.Minute).Format("2006-01-02T15:04:05Z"))))
	}))

	s, err := NewSession(&aws.Config{Endpoint: aws.String(server.URL), DisableSSL: aws.Bool(true)})

	creds, err := s.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.Equal(t, "AKID", creds.AccessKeyID)
	assert.Equal(t, "SECRET", creds.SecretAccessKey)
	assert.Equal(t, "SESSION_TOKEN", creds.SessionToken)
	assert.Contains(t, creds.ProviderName, "AssumeRoleProvider")
}

func TestSessionAssumeRole_WithMFA(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, r.FormValue("SerialNumber"), "0123456789")
		assert.Equal(t, r.FormValue("TokenCode"), "tokencode")

		w.Write([]byte(fmt.Sprintf(assumeRoleRespMsg, time.Now().Add(15*time.Minute).Format("2006-01-02T15:04:05Z"))))
	}))

	customProviderCalled := false
	sess, err := NewSessionWithOptions(Options{
		Profile: "assume_role_w_mfa",
		Config: aws.Config{
			Region:     aws.String("us-east-1"),
			Endpoint:   aws.String(server.URL),
			DisableSSL: aws.Bool(true),
		},
		SharedConfigState: SharedConfigEnable,
		AssumeRoleTokenProvider: func() (string, error) {
			customProviderCalled = true

			return "tokencode", nil
		},
	})
	assert.NoError(t, err)

	creds, err := sess.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.True(t, customProviderCalled)

	assert.Equal(t, "AKID", creds.AccessKeyID)
	assert.Equal(t, "SECRET", creds.SecretAccessKey)
	assert.Equal(t, "SESSION_TOKEN", creds.SessionToken)
	assert.Contains(t, creds.ProviderName, "AssumeRoleProvider")
}

func TestSessionAssumeRole_WithMFA_NoTokenProvider(t *testing.T) {
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	_, err := NewSessionWithOptions(Options{
		Profile:           "assume_role_w_mfa",
		SharedConfigState: SharedConfigEnable,
	})
	assert.Equal(t, err, AssumeRoleTokenProviderNotSetError{})
}

func TestSessionAssumeRole_DisableSharedConfig(t *testing.T) {
	// Backwards compatibility with Shared config disabled
	// assume role should not be built into the config.
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "0")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	s, err := NewSession()
	assert.NoError(t, err)

	creds, err := s.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.Equal(t, "assume_role_w_creds_akid", creds.AccessKeyID)
	assert.Equal(t, "assume_role_w_creds_secret", creds.SecretAccessKey)
	assert.Contains(t, creds.ProviderName, "SharedConfigCredentials")
}

func TestSessionAssumeRole_InvalidSourceProfile(t *testing.T) {
	// Backwards compatibility with Shared config disabled
	// assume role should not be built into the config.
	oldEnv := initSessionTestEnv()
	defer awstesting.PopEnv(oldEnv)

	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_invalid_source_profile")

	s, err := NewSession()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "SharedConfigAssumeRoleError: failed to load assume role")
	assert.Nil(t, s)
}

func initSessionTestEnv() (oldEnv []string) {
	oldEnv = awstesting.StashEnv()
	os.Setenv("AWS_CONFIG_FILE", "file_not_exists")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", "file_not_exists")

	return oldEnv
}
