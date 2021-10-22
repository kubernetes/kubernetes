// +build go1.7

package session

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdktesting"
	"github.com/aws/aws-sdk-go/internal/shareddefaults"
	"github.com/aws/aws-sdk-go/service/sts"
)

func newEc2MetadataServer(key, secret string, closeAfterGetCreds bool) *httptest.Server {
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/latest/meta-data/iam/security-credentials/RoleName" {
				w.Write([]byte(fmt.Sprintf(ec2MetadataResponse, key, secret)))

				if closeAfterGetCreds {
					go server.Close()
				}
			} else if r.URL.Path == "/latest/meta-data/iam/security-credentials/" {
				w.Write([]byte("RoleName"))
			} else {
				w.Write([]byte(""))
			}
		}))

	return server
}

func setupCredentialsEndpoints(t *testing.T) (endpoints.Resolver, func()) {
	origECSEndpoint := shareddefaults.ECSContainerCredentialsURI

	ecsMetadataServer := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/ECS" {
				w.Write([]byte(ecsResponse))
			} else {
				w.Write([]byte(""))
			}
		}))
	shareddefaults.ECSContainerCredentialsURI = ecsMetadataServer.URL

	ec2MetadataServer := newEc2MetadataServer("ec2_key", "ec2_secret", false)

	stsServer := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Write([]byte(fmt.Sprintf(
				assumeRoleRespMsg,
				time.Now().
					Add(15*time.Minute).
					Format("2006-01-02T15:04:05Z"))))
		}))

	resolver := endpoints.ResolverFunc(
		func(service, region string, opts ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
			switch service {
			case "ec2metadata":
				return endpoints.ResolvedEndpoint{
					URL: ec2MetadataServer.URL,
				}, nil
			case "sts":
				return endpoints.ResolvedEndpoint{
					URL: stsServer.URL,
				}, nil
			default:
				return endpoints.ResolvedEndpoint{},
					fmt.Errorf("unknown service endpoint, %s", service)
			}
		})

	return resolver, func() {
		shareddefaults.ECSContainerCredentialsURI = origECSEndpoint
		ecsMetadataServer.Close()
		ec2MetadataServer.Close()
		stsServer.Close()
	}
}

func TestSharedConfigCredentialSource(t *testing.T) {
	const configFileForWindows = "testdata/credential_source_config_for_windows"
	const configFile = "testdata/credential_source_config"

	cases := []struct {
		name                   string
		profile                string
		sessOptProfile         string
		sessOptEC2IMDSEndpoint string
		expectedError          error
		expectedAccessKey      string
		expectedSecretKey      string
		expectedChain          []string
		init                   func()
		dependentOnOS          bool
	}{
		{
			name:          "credential source and source profile",
			profile:       "invalid_source_and_credential_source",
			expectedError: ErrSharedConfigSourceCollision,
			init: func() {
				os.Setenv("AWS_ACCESS_KEY", "access_key")
				os.Setenv("AWS_SECRET_KEY", "secret_key")
			},
		},
		{
			name:              "env var credential source",
			sessOptProfile:    "env_var_credential_source",
			expectedAccessKey: "AKID",
			expectedSecretKey: "SECRET",
			expectedChain: []string{
				"assume_role_w_creds_role_arn_env",
			},
			init: func() {
				os.Setenv("AWS_ACCESS_KEY", "access_key")
				os.Setenv("AWS_SECRET_KEY", "secret_key")
			},
		},
		{
			name:    "ec2metadata credential source",
			profile: "ec2metadata",
			expectedChain: []string{
				"assume_role_w_creds_role_arn_ec2",
			},
			expectedAccessKey: "AKID",
			expectedSecretKey: "SECRET",
		},
		{
			name:              "ec2metadata custom EC2 IMDS endpoint, env var",
			profile:           "not-exists-profile",
			expectedAccessKey: "ec2_custom_key",
			expectedSecretKey: "ec2_custom_secret",
			init: func() {
				altServer := newEc2MetadataServer("ec2_custom_key", "ec2_custom_secret", true)
				os.Setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", altServer.URL)
			},
		},
		{
			name:              "ecs container credential source",
			profile:           "ecscontainer",
			expectedAccessKey: "AKID",
			expectedSecretKey: "SECRET",
			expectedChain: []string{
				"assume_role_w_creds_role_arn_ecs",
			},
			init: func() {
				os.Setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/ECS")
			},
		},
		{
			name:              "chained assume role with env creds",
			profile:           "chained_assume_role",
			expectedAccessKey: "AKID",
			expectedSecretKey: "SECRET",
			expectedChain: []string{
				"assume_role_w_creds_role_arn_chain",
				"assume_role_w_creds_role_arn_ec2",
			},
		},
		{
			name:              "credential process with no ARN set",
			profile:           "cred_proc_no_arn_set",
			dependentOnOS:     true,
			expectedAccessKey: "cred_proc_akid",
			expectedSecretKey: "cred_proc_secret",
		},
		{
			name:              "credential process with ARN set",
			profile:           "cred_proc_arn_set",
			dependentOnOS:     true,
			expectedAccessKey: "AKID",
			expectedSecretKey: "SECRET",
			expectedChain: []string{
				"assume_role_w_creds_proc_role_arn",
			},
		},
		{
			name:              "chained assume role with credential process",
			profile:           "chained_cred_proc",
			dependentOnOS:     true,
			expectedAccessKey: "AKID",
			expectedSecretKey: "SECRET",
			expectedChain: []string{
				"assume_role_w_creds_proc_source_prof",
			},
		},
	}

	for i, c := range cases {
		t.Run(strconv.Itoa(i)+"_"+c.name, func(t *testing.T) {
			restoreEnvFn := sdktesting.StashEnv()
			defer restoreEnvFn()

			if c.dependentOnOS && runtime.GOOS == "windows" {
				os.Setenv("AWS_CONFIG_FILE", configFileForWindows)
			} else {
				os.Setenv("AWS_CONFIG_FILE", configFile)
			}

			os.Setenv("AWS_REGION", "us-east-1")
			os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
			if len(c.profile) != 0 {
				os.Setenv("AWS_PROFILE", c.profile)
			}

			endpointResolver, cleanupFn := setupCredentialsEndpoints(t)
			defer cleanupFn()

			if c.init != nil {
				c.init()
			}

			var credChain []string
			handlers := defaults.Handlers()
			handlers.Sign.PushBack(func(r *request.Request) {
				if r.Config.Credentials == credentials.AnonymousCredentials {
					return
				}
				params := r.Params.(*sts.AssumeRoleInput)
				credChain = append(credChain, *params.RoleArn)
			})

			sess, err := NewSessionWithOptions(Options{
				Profile: c.sessOptProfile,
				Config: aws.Config{
					Logger:           t,
					EndpointResolver: endpointResolver,
				},
				Handlers:        handlers,
				EC2IMDSEndpoint: c.sessOptEC2IMDSEndpoint,
			})
			if e, a := c.expectedError, err; e != a {
				t.Fatalf("expected %v, but received %v", e, a)
			}

			if c.expectedError != nil {
				return
			}

			creds, err := sess.Config.Credentials.Get()
			if err != nil {
				t.Fatalf("expected no error, but received %v", err)
			}

			if e, a := c.expectedChain, credChain; !reflect.DeepEqual(e, a) {
				t.Errorf("expected %v, but received %v", e, a)
			}

			if e, a := c.expectedAccessKey, creds.AccessKeyID; e != a {
				t.Errorf("expected %v, but received %v", e, a)
			}

			if e, a := c.expectedSecretKey, creds.SecretAccessKey; e != a {
				t.Errorf("expected %v, but received %v", e, a)
			}
		})
	}
}

const ecsResponse = `{
	  "Code": "Success",
	  "Type": "AWS-HMAC",
	  "AccessKeyId" : "ecs-access-key",
	  "SecretAccessKey" : "ecs-secret-key",
	  "Token" : "token",
	  "Expiration" : "2100-01-01T00:00:00Z",
	  "LastUpdated" : "2009-11-23T0:00:00Z"
	}`

const ec2MetadataResponse = `{
	  "Code": "Success",
	  "Type": "AWS-HMAC",
	  "AccessKeyId" : "%s",
	  "SecretAccessKey" : "%s",
	  "Token" : "token",
	  "Expiration" : "2100-01-01T00:00:00Z",
	  "LastUpdated" : "2009-11-23T0:00:00Z"
	}`

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

func TestSessionAssumeRole(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(fmt.Sprintf(
			assumeRoleRespMsg,
			time.Now().Add(15*time.Minute).Format("2006-01-02T15:04:05Z"))))
	}))
	defer server.Close()

	s, err := NewSession(&aws.Config{
		Endpoint:   aws.String(server.URL),
		DisableSSL: aws.Bool(true),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	creds, err := s.Config.Credentials.Get()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := "AKID", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SECRET", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SESSION_TOKEN", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "AssumeRoleProvider", creds.ProviderName; !strings.Contains(a, e) {
		t.Errorf("expect %v, to be in %v", e, a)
	}
}

func TestSessionAssumeRole_WithMFA(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if e, a := r.FormValue("SerialNumber"), "0123456789"; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := r.FormValue("TokenCode"), "tokencode"; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := "900", r.FormValue("DurationSeconds"); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}

		w.Write([]byte(fmt.Sprintf(
			assumeRoleRespMsg,
			time.Now().Add(15*time.Minute).Format("2006-01-02T15:04:05Z"))))
	}))
	defer server.Close()

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
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	creds, err := sess.Config.Credentials.Get()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if !customProviderCalled {
		t.Errorf("expect true")
	}

	if e, a := "AKID", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SECRET", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SESSION_TOKEN", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "AssumeRoleProvider", creds.ProviderName; !strings.Contains(a, e) {
		t.Errorf("expect %v, to be in %v", e, a)
	}
}

func TestSessionAssumeRole_WithMFA_NoTokenProvider(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	_, err := NewSessionWithOptions(Options{
		Profile:           "assume_role_w_mfa",
		SharedConfigState: SharedConfigEnable,
	})
	if e, a := (AssumeRoleTokenProviderNotSetError{}), err; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestSessionAssumeRole_DisableSharedConfig(t *testing.T) {
	// Backwards compatibility with Shared config disabled
	// assume role should not be built into the config.
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_SDK_LOAD_CONFIG", "0")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	s, err := NewSession(&aws.Config{
		CredentialsChainVerboseErrors: aws.Bool(true),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	creds, err := s.Config.Credentials.Get()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if e, a := "assume_role_w_creds_akid", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "assume_role_w_creds_secret", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SharedConfigCredentials", creds.ProviderName; !strings.Contains(a, e) {
		t.Errorf("expect %v, to be in %v", e, a)
	}
}

func TestSessionAssumeRole_InvalidSourceProfile(t *testing.T) {
	// Backwards compatibility with Shared config disabled
	// assume role should not be built into the config.
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_invalid_source_profile")

	s, err := NewSession()
	if err == nil {
		t.Fatalf("expect error, got none")
	}

	expectMsg := "SharedConfigAssumeRoleError: failed to load assume role"
	if e, a := expectMsg, err.Error(); !strings.Contains(a, e) {
		t.Errorf("expect %v, to be in %v", e, a)
	}
	if s != nil {
		t.Errorf("expect nil, %v", err)
	}
}

func TestSessionAssumeRole_ExtendedDuration(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	cases := []struct {
		profile          string
		optionDuration   time.Duration
		expectedDuration string
	}{
		{
			profile:          "assume_role_w_creds",
			expectedDuration: "900",
		},
		{
			profile:          "assume_role_w_creds",
			optionDuration:   30 * time.Minute,
			expectedDuration: "1800",
		},
		{
			profile:          "assume_role_w_creds_w_duration",
			expectedDuration: "1800",
		},
		{
			profile:          "assume_role_w_creds_w_invalid_duration",
			expectedDuration: "900",
		},
	}

	for _, tt := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if e, a := tt.expectedDuration, r.FormValue("DurationSeconds"); e != a {
				t.Errorf("expect %v, got %v", e, a)
			}

			w.Write([]byte(fmt.Sprintf(
				assumeRoleRespMsg,
				time.Now().Add(15*time.Minute).Format("2006-01-02T15:04:05Z"))))
		}))

		os.Setenv("AWS_REGION", "us-east-1")
		os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
		os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
		os.Setenv("AWS_PROFILE", "assume_role_w_creds")

		opts := Options{
			Profile: tt.profile,
			Config: aws.Config{
				Endpoint:   aws.String(server.URL),
				DisableSSL: aws.Bool(true),
			},
			SharedConfigState: SharedConfigEnable,
		}
		if tt.optionDuration != 0 {
			opts.AssumeRoleDuration = tt.optionDuration
		}

		s, err := NewSessionWithOptions(opts)
		if err != nil {
			server.Close()
			t.Fatalf("expect no error, got %v", err)
		}

		creds, err := s.Config.Credentials.Get()
		if err != nil {
			server.Close()
			t.Fatalf("expect no error, got %v", err)
		}

		if e, a := "AKID", creds.AccessKeyID; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := "SECRET", creds.SecretAccessKey; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := "SESSION_TOKEN", creds.SessionToken; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := "AssumeRoleProvider", creds.ProviderName; !strings.Contains(a, e) {
			t.Errorf("expect %v, to be in %v", e, a)
		}

		server.Close()
	}
}

func TestSessionAssumeRole_WithMFA_ExtendedDuration(t *testing.T) {
	restoreEnvFn := initSessionTestEnv()
	defer restoreEnvFn()

	os.Setenv("AWS_REGION", "us-east-1")
	os.Setenv("AWS_SDK_LOAD_CONFIG", "1")
	os.Setenv("AWS_SHARED_CREDENTIALS_FILE", testConfigFilename)
	os.Setenv("AWS_PROFILE", "assume_role_w_creds")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if e, a := "0123456789", r.FormValue("SerialNumber"); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := "tokencode", r.FormValue("TokenCode"); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if e, a := "1800", r.FormValue("DurationSeconds"); e != a {
			t.Errorf("expect %v, got %v", e, a)
		}

		w.Write([]byte(fmt.Sprintf(
			assumeRoleRespMsg,
			time.Now().Add(30*time.Minute).Format("2006-01-02T15:04:05Z"))))
	}))
	defer server.Close()

	customProviderCalled := false
	sess, err := NewSessionWithOptions(Options{
		Profile: "assume_role_w_mfa",
		Config: aws.Config{
			Region:     aws.String("us-east-1"),
			Endpoint:   aws.String(server.URL),
			DisableSSL: aws.Bool(true),
		},
		SharedConfigState:  SharedConfigEnable,
		AssumeRoleDuration: 30 * time.Minute,
		AssumeRoleTokenProvider: func() (string, error) {
			customProviderCalled = true

			return "tokencode", nil
		},
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	creds, err := sess.Config.Credentials.Get()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
	if !customProviderCalled {
		t.Errorf("expect true")
	}

	if e, a := "AKID", creds.AccessKeyID; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SECRET", creds.SecretAccessKey; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "SESSION_TOKEN", creds.SessionToken; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "AssumeRoleProvider", creds.ProviderName; !strings.Contains(a, e) {
		t.Errorf("expect %v, to be in %v", e, a)
	}
}
