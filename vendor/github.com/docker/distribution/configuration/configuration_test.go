package configuration

import (
	"bytes"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"

	. "gopkg.in/check.v1"
	"gopkg.in/yaml.v2"
)

// Hook up gocheck into the "go test" runner
func Test(t *testing.T) { TestingT(t) }

// configStruct is a canonical example configuration, which should map to configYamlV0_1
var configStruct = Configuration{
	Version: "0.1",
	Log: struct {
		AccessLog struct {
			Disabled bool `yaml:"disabled,omitempty"`
		} `yaml:"accesslog,omitempty"`
		Level     Loglevel               `yaml:"level"`
		Formatter string                 `yaml:"formatter,omitempty"`
		Fields    map[string]interface{} `yaml:"fields,omitempty"`
		Hooks     []LogHook              `yaml:"hooks,omitempty"`
	}{
		Fields: map[string]interface{}{"environment": "test"},
	},
	Loglevel: "info",
	Storage: Storage{
		"s3": Parameters{
			"region":        "us-east-1",
			"bucket":        "my-bucket",
			"rootdirectory": "/registry",
			"encrypt":       true,
			"secure":        false,
			"accesskey":     "SAMPLEACCESSKEY",
			"secretkey":     "SUPERSECRET",
			"host":          nil,
			"port":          42,
		},
	},
	Auth: Auth{
		"silly": Parameters{
			"realm":   "silly",
			"service": "silly",
		},
	},
	Reporting: Reporting{
		Bugsnag: BugsnagReporting{
			APIKey: "BugsnagApiKey",
		},
	},
	Notifications: Notifications{
		Endpoints: []Endpoint{
			{
				Name: "endpoint-1",
				URL:  "http://example.com",
				Headers: http.Header{
					"Authorization": []string{"Bearer <example>"},
				},
				IgnoredMediaTypes: []string{"application/octet-stream"},
			},
		},
	},
	HTTP: struct {
		Addr         string `yaml:"addr,omitempty"`
		Net          string `yaml:"net,omitempty"`
		Host         string `yaml:"host,omitempty"`
		Prefix       string `yaml:"prefix,omitempty"`
		Secret       string `yaml:"secret,omitempty"`
		RelativeURLs bool   `yaml:"relativeurls,omitempty"`
		TLS          struct {
			Certificate string   `yaml:"certificate,omitempty"`
			Key         string   `yaml:"key,omitempty"`
			ClientCAs   []string `yaml:"clientcas,omitempty"`
			LetsEncrypt struct {
				CacheFile string `yaml:"cachefile,omitempty"`
				Email     string `yaml:"email,omitempty"`
			} `yaml:"letsencrypt,omitempty"`
		} `yaml:"tls,omitempty"`
		Headers http.Header `yaml:"headers,omitempty"`
		Debug   struct {
			Addr string `yaml:"addr,omitempty"`
		} `yaml:"debug,omitempty"`
		HTTP2 struct {
			Disabled bool `yaml:"disabled,omitempty"`
		} `yaml:"http2,omitempty"`
	}{
		TLS: struct {
			Certificate string   `yaml:"certificate,omitempty"`
			Key         string   `yaml:"key,omitempty"`
			ClientCAs   []string `yaml:"clientcas,omitempty"`
			LetsEncrypt struct {
				CacheFile string `yaml:"cachefile,omitempty"`
				Email     string `yaml:"email,omitempty"`
			} `yaml:"letsencrypt,omitempty"`
		}{
			ClientCAs: []string{"/path/to/ca.pem"},
		},
		Headers: http.Header{
			"X-Content-Type-Options": []string{"nosniff"},
		},
		HTTP2: struct {
			Disabled bool `yaml:"disabled,omitempty"`
		}{
			Disabled: false,
		},
	},
}

// configYamlV0_1 is a Version 0.1 yaml document representing configStruct
var configYamlV0_1 = `
version: 0.1
log:
  fields:
    environment: test
loglevel: info
storage:
  s3:
    region: us-east-1
    bucket: my-bucket
    rootdirectory: /registry
    encrypt: true
    secure: false
    accesskey: SAMPLEACCESSKEY
    secretkey: SUPERSECRET
    host: ~
    port: 42
auth:
  silly:
    realm: silly
    service: silly
notifications:
  endpoints:
    - name: endpoint-1
      url:  http://example.com
      headers:
        Authorization: [Bearer <example>]
      ignoredmediatypes:
        - application/octet-stream
reporting:
  bugsnag:
    apikey: BugsnagApiKey
http:
  clientcas:
    - /path/to/ca.pem
  headers:
    X-Content-Type-Options: [nosniff]
`

// inmemoryConfigYamlV0_1 is a Version 0.1 yaml document specifying an inmemory
// storage driver with no parameters
var inmemoryConfigYamlV0_1 = `
version: 0.1
loglevel: info
storage: inmemory
auth:
  silly:
    realm: silly
    service: silly
notifications:
  endpoints:
    - name: endpoint-1
      url:  http://example.com
      headers:
        Authorization: [Bearer <example>]
      ignoredmediatypes:
        - application/octet-stream
http:
  headers:
    X-Content-Type-Options: [nosniff]
`

type ConfigSuite struct {
	expectedConfig *Configuration
}

var _ = Suite(new(ConfigSuite))

func (suite *ConfigSuite) SetUpTest(c *C) {
	os.Clearenv()
	suite.expectedConfig = copyConfig(configStruct)
}

// TestMarshalRoundtrip validates that configStruct can be marshaled and
// unmarshaled without changing any parameters
func (suite *ConfigSuite) TestMarshalRoundtrip(c *C) {
	configBytes, err := yaml.Marshal(suite.expectedConfig)
	c.Assert(err, IsNil)
	config, err := Parse(bytes.NewReader(configBytes))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseSimple validates that configYamlV0_1 can be parsed into a struct
// matching configStruct
func (suite *ConfigSuite) TestParseSimple(c *C) {
	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseInmemory validates that configuration yaml with storage provided as
// a string can be parsed into a Configuration struct with no storage parameters
func (suite *ConfigSuite) TestParseInmemory(c *C) {
	suite.expectedConfig.Storage = Storage{"inmemory": Parameters{}}
	suite.expectedConfig.Reporting = Reporting{}
	suite.expectedConfig.Log.Fields = nil

	config, err := Parse(bytes.NewReader([]byte(inmemoryConfigYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseIncomplete validates that an incomplete yaml configuration cannot
// be parsed without providing environment variables to fill in the missing
// components.
func (suite *ConfigSuite) TestParseIncomplete(c *C) {
	incompleteConfigYaml := "version: 0.1"
	_, err := Parse(bytes.NewReader([]byte(incompleteConfigYaml)))
	c.Assert(err, NotNil)

	suite.expectedConfig.Log.Fields = nil
	suite.expectedConfig.Storage = Storage{"filesystem": Parameters{"rootdirectory": "/tmp/testroot"}}
	suite.expectedConfig.Auth = Auth{"silly": Parameters{"realm": "silly"}}
	suite.expectedConfig.Reporting = Reporting{}
	suite.expectedConfig.Notifications = Notifications{}
	suite.expectedConfig.HTTP.Headers = nil

	// Note: this also tests that REGISTRY_STORAGE and
	// REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY can be used together
	os.Setenv("REGISTRY_STORAGE", "filesystem")
	os.Setenv("REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY", "/tmp/testroot")
	os.Setenv("REGISTRY_AUTH", "silly")
	os.Setenv("REGISTRY_AUTH_SILLY_REALM", "silly")

	config, err := Parse(bytes.NewReader([]byte(incompleteConfigYaml)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseWithSameEnvStorage validates that providing environment variables
// that match the given storage type will only include environment-defined
// parameters and remove yaml-defined parameters
func (suite *ConfigSuite) TestParseWithSameEnvStorage(c *C) {
	suite.expectedConfig.Storage = Storage{"s3": Parameters{"region": "us-east-1"}}

	os.Setenv("REGISTRY_STORAGE", "s3")
	os.Setenv("REGISTRY_STORAGE_S3_REGION", "us-east-1")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseWithDifferentEnvStorageParams validates that providing environment variables that change
// and add to the given storage parameters will change and add parameters to the parsed
// Configuration struct
func (suite *ConfigSuite) TestParseWithDifferentEnvStorageParams(c *C) {
	suite.expectedConfig.Storage.setParameter("region", "us-west-1")
	suite.expectedConfig.Storage.setParameter("secure", true)
	suite.expectedConfig.Storage.setParameter("newparam", "some Value")

	os.Setenv("REGISTRY_STORAGE_S3_REGION", "us-west-1")
	os.Setenv("REGISTRY_STORAGE_S3_SECURE", "true")
	os.Setenv("REGISTRY_STORAGE_S3_NEWPARAM", "some Value")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseWithDifferentEnvStorageType validates that providing an environment variable that
// changes the storage type will be reflected in the parsed Configuration struct
func (suite *ConfigSuite) TestParseWithDifferentEnvStorageType(c *C) {
	suite.expectedConfig.Storage = Storage{"inmemory": Parameters{}}

	os.Setenv("REGISTRY_STORAGE", "inmemory")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseWithDifferentEnvStorageTypeAndParams validates that providing an environment variable
// that changes the storage type will be reflected in the parsed Configuration struct and that
// environment storage parameters will also be included
func (suite *ConfigSuite) TestParseWithDifferentEnvStorageTypeAndParams(c *C) {
	suite.expectedConfig.Storage = Storage{"filesystem": Parameters{}}
	suite.expectedConfig.Storage.setParameter("rootdirectory", "/tmp/testroot")

	os.Setenv("REGISTRY_STORAGE", "filesystem")
	os.Setenv("REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY", "/tmp/testroot")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseWithSameEnvLoglevel validates that providing an environment variable defining the log
// level to the same as the one provided in the yaml will not change the parsed Configuration struct
func (suite *ConfigSuite) TestParseWithSameEnvLoglevel(c *C) {
	os.Setenv("REGISTRY_LOGLEVEL", "info")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseWithDifferentEnvLoglevel validates that providing an environment variable defining the
// log level will override the value provided in the yaml document
func (suite *ConfigSuite) TestParseWithDifferentEnvLoglevel(c *C) {
	suite.expectedConfig.Loglevel = "error"

	os.Setenv("REGISTRY_LOGLEVEL", "error")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseInvalidLoglevel validates that the parser will fail to parse a
// configuration if the loglevel is malformed
func (suite *ConfigSuite) TestParseInvalidLoglevel(c *C) {
	invalidConfigYaml := "version: 0.1\nloglevel: derp\nstorage: inmemory"
	_, err := Parse(bytes.NewReader([]byte(invalidConfigYaml)))
	c.Assert(err, NotNil)

	os.Setenv("REGISTRY_LOGLEVEL", "derp")

	_, err = Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, NotNil)

}

// TestParseWithDifferentEnvReporting validates that environment variables
// properly override reporting parameters
func (suite *ConfigSuite) TestParseWithDifferentEnvReporting(c *C) {
	suite.expectedConfig.Reporting.Bugsnag.APIKey = "anotherBugsnagApiKey"
	suite.expectedConfig.Reporting.Bugsnag.Endpoint = "localhost:8080"
	suite.expectedConfig.Reporting.NewRelic.LicenseKey = "NewRelicLicenseKey"
	suite.expectedConfig.Reporting.NewRelic.Name = "some NewRelic NAME"

	os.Setenv("REGISTRY_REPORTING_BUGSNAG_APIKEY", "anotherBugsnagApiKey")
	os.Setenv("REGISTRY_REPORTING_BUGSNAG_ENDPOINT", "localhost:8080")
	os.Setenv("REGISTRY_REPORTING_NEWRELIC_LICENSEKEY", "NewRelicLicenseKey")
	os.Setenv("REGISTRY_REPORTING_NEWRELIC_NAME", "some NewRelic NAME")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseInvalidVersion validates that the parser will fail to parse a newer configuration
// version than the CurrentVersion
func (suite *ConfigSuite) TestParseInvalidVersion(c *C) {
	suite.expectedConfig.Version = MajorMinorVersion(CurrentVersion.Major(), CurrentVersion.Minor()+1)
	configBytes, err := yaml.Marshal(suite.expectedConfig)
	c.Assert(err, IsNil)
	_, err = Parse(bytes.NewReader(configBytes))
	c.Assert(err, NotNil)
}

// TestParseExtraneousVars validates that environment variables referring to
// nonexistent variables don't cause side effects.
func (suite *ConfigSuite) TestParseExtraneousVars(c *C) {
	suite.expectedConfig.Reporting.Bugsnag.Endpoint = "localhost:8080"

	// A valid environment variable
	os.Setenv("REGISTRY_REPORTING_BUGSNAG_ENDPOINT", "localhost:8080")

	// Environment variables which shouldn't set config items
	os.Setenv("registry_REPORTING_NEWRELIC_LICENSEKEY", "NewRelicLicenseKey")
	os.Setenv("REPORTING_NEWRELIC_NAME", "some NewRelic NAME")
	os.Setenv("REGISTRY_DUCKS", "quack")
	os.Setenv("REGISTRY_REPORTING_ASDF", "ghjk")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseEnvVarImplicitMaps validates that environment variables can set
// values in maps that don't already exist.
func (suite *ConfigSuite) TestParseEnvVarImplicitMaps(c *C) {
	readonly := make(map[string]interface{})
	readonly["enabled"] = true

	maintenance := make(map[string]interface{})
	maintenance["readonly"] = readonly

	suite.expectedConfig.Storage["maintenance"] = maintenance

	os.Setenv("REGISTRY_STORAGE_MAINTENANCE_READONLY_ENABLED", "true")

	config, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
	c.Assert(config, DeepEquals, suite.expectedConfig)
}

// TestParseEnvWrongTypeMap validates that incorrectly attempting to unmarshal a
// string over existing map fails.
func (suite *ConfigSuite) TestParseEnvWrongTypeMap(c *C) {
	os.Setenv("REGISTRY_STORAGE_S3", "somestring")

	_, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, NotNil)
}

// TestParseEnvWrongTypeStruct validates that incorrectly attempting to
// unmarshal a string into a struct fails.
func (suite *ConfigSuite) TestParseEnvWrongTypeStruct(c *C) {
	os.Setenv("REGISTRY_STORAGE_LOG", "somestring")

	_, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, NotNil)
}

// TestParseEnvWrongTypeSlice validates that incorrectly attempting to
// unmarshal a string into a slice fails.
func (suite *ConfigSuite) TestParseEnvWrongTypeSlice(c *C) {
	os.Setenv("REGISTRY_LOG_HOOKS", "somestring")

	_, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, NotNil)
}

// TestParseEnvMany tests several environment variable overrides.
// The result is not checked - the goal of this test is to detect panics
// from misuse of reflection.
func (suite *ConfigSuite) TestParseEnvMany(c *C) {
	os.Setenv("REGISTRY_VERSION", "0.1")
	os.Setenv("REGISTRY_LOG_LEVEL", "debug")
	os.Setenv("REGISTRY_LOG_FORMATTER", "json")
	os.Setenv("REGISTRY_LOG_HOOKS", "json")
	os.Setenv("REGISTRY_LOG_FIELDS", "abc: xyz")
	os.Setenv("REGISTRY_LOG_HOOKS", "- type: asdf")
	os.Setenv("REGISTRY_LOGLEVEL", "debug")
	os.Setenv("REGISTRY_STORAGE", "s3")
	os.Setenv("REGISTRY_AUTH_PARAMS", "param1: value1")
	os.Setenv("REGISTRY_AUTH_PARAMS_VALUE2", "value2")
	os.Setenv("REGISTRY_AUTH_PARAMS_VALUE2", "value2")

	_, err := Parse(bytes.NewReader([]byte(configYamlV0_1)))
	c.Assert(err, IsNil)
}

func checkStructs(c *C, t reflect.Type, structsChecked map[string]struct{}) {
	for t.Kind() == reflect.Ptr || t.Kind() == reflect.Map || t.Kind() == reflect.Slice {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return
	}
	if _, present := structsChecked[t.String()]; present {
		// Already checked this type
		return
	}

	structsChecked[t.String()] = struct{}{}

	byUpperCase := make(map[string]int)
	for i := 0; i < t.NumField(); i++ {
		sf := t.Field(i)

		// Check that the yaml tag does not contain an _.
		yamlTag := sf.Tag.Get("yaml")
		if strings.Contains(yamlTag, "_") {
			c.Fatalf("yaml field name includes _ character: %s", yamlTag)
		}
		upper := strings.ToUpper(sf.Name)
		if _, present := byUpperCase[upper]; present {
			c.Fatalf("field name collision in configuration object: %s", sf.Name)
		}
		byUpperCase[upper] = i

		checkStructs(c, sf.Type, structsChecked)
	}
}

// TestValidateConfigStruct makes sure that the config struct has no members
// with yaml tags that would be ambiguous to the environment variable parser.
func (suite *ConfigSuite) TestValidateConfigStruct(c *C) {
	structsChecked := make(map[string]struct{})
	checkStructs(c, reflect.TypeOf(Configuration{}), structsChecked)
}

func copyConfig(config Configuration) *Configuration {
	configCopy := new(Configuration)

	configCopy.Version = MajorMinorVersion(config.Version.Major(), config.Version.Minor())
	configCopy.Loglevel = config.Loglevel
	configCopy.Log = config.Log
	configCopy.Log.Fields = make(map[string]interface{}, len(config.Log.Fields))
	for k, v := range config.Log.Fields {
		configCopy.Log.Fields[k] = v
	}

	configCopy.Storage = Storage{config.Storage.Type(): Parameters{}}
	for k, v := range config.Storage.Parameters() {
		configCopy.Storage.setParameter(k, v)
	}
	configCopy.Reporting = Reporting{
		Bugsnag:  BugsnagReporting{config.Reporting.Bugsnag.APIKey, config.Reporting.Bugsnag.ReleaseStage, config.Reporting.Bugsnag.Endpoint},
		NewRelic: NewRelicReporting{config.Reporting.NewRelic.LicenseKey, config.Reporting.NewRelic.Name, config.Reporting.NewRelic.Verbose},
	}

	configCopy.Auth = Auth{config.Auth.Type(): Parameters{}}
	for k, v := range config.Auth.Parameters() {
		configCopy.Auth.setParameter(k, v)
	}

	configCopy.Notifications = Notifications{Endpoints: []Endpoint{}}
	for _, v := range config.Notifications.Endpoints {
		configCopy.Notifications.Endpoints = append(configCopy.Notifications.Endpoints, v)
	}

	configCopy.HTTP.Headers = make(http.Header)
	for k, v := range config.HTTP.Headers {
		configCopy.HTTP.Headers[k] = v
	}

	return configCopy
}
