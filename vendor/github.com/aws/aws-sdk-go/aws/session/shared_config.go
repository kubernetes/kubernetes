package session

import (
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/internal/ini"
)

const (
	// Static Credentials group
	accessKeyIDKey  = `aws_access_key_id`     // group required
	secretAccessKey = `aws_secret_access_key` // group required
	sessionTokenKey = `aws_session_token`     // optional

	// Assume Role Credentials group
	roleArnKey             = `role_arn`          // group required
	sourceProfileKey       = `source_profile`    // group required (or credential_source)
	credentialSourceKey    = `credential_source` // group required (or source_profile)
	externalIDKey          = `external_id`       // optional
	mfaSerialKey           = `mfa_serial`        // optional
	roleSessionNameKey     = `role_session_name` // optional
	roleDurationSecondsKey = "duration_seconds"  // optional

	// AWS Single Sign-On (AWS SSO) group
	ssoAccountIDKey = "sso_account_id"
	ssoRegionKey    = "sso_region"
	ssoRoleNameKey  = "sso_role_name"
	ssoStartURL     = "sso_start_url"

	// CSM options
	csmEnabledKey  = `csm_enabled`
	csmHostKey     = `csm_host`
	csmPortKey     = `csm_port`
	csmClientIDKey = `csm_client_id`

	// Additional Config fields
	regionKey = `region`

	// custom CA Bundle filename
	customCABundleKey = `ca_bundle`

	// endpoint discovery group
	enableEndpointDiscoveryKey = `endpoint_discovery_enabled` // optional

	// External Credential Process
	credentialProcessKey = `credential_process` // optional

	// Web Identity Token File
	webIdentityTokenFileKey = `web_identity_token_file` // optional

	// Additional config fields for regional or legacy endpoints
	stsRegionalEndpointSharedKey = `sts_regional_endpoints`

	// Additional config fields for regional or legacy endpoints
	s3UsEast1RegionalSharedKey = `s3_us_east_1_regional_endpoint`

	// DefaultSharedConfigProfile is the default profile to be used when
	// loading configuration from the config files if another profile name
	// is not provided.
	DefaultSharedConfigProfile = `default`

	// S3 ARN Region Usage
	s3UseARNRegionKey = "s3_use_arn_region"
)

// sharedConfig represents the configuration fields of the SDK config files.
type sharedConfig struct {
	Profile string

	// Credentials values from the config file. Both aws_access_key_id and
	// aws_secret_access_key must be provided together in the same file to be
	// considered valid. The values will be ignored if not a complete group.
	// aws_session_token is an optional field that can be provided if both of
	// the other two fields are also provided.
	//
	//	aws_access_key_id
	//	aws_secret_access_key
	//	aws_session_token
	Creds credentials.Value

	CredentialSource     string
	CredentialProcess    string
	WebIdentityTokenFile string

	SSOAccountID string
	SSORegion    string
	SSORoleName  string
	SSOStartURL  string

	RoleARN            string
	RoleSessionName    string
	ExternalID         string
	MFASerial          string
	AssumeRoleDuration *time.Duration

	SourceProfileName string
	SourceProfile     *sharedConfig

	// Region is the region the SDK should use for looking up AWS service
	// endpoints and signing requests.
	//
	//	region
	Region string

	// CustomCABundle is the file path to a PEM file the SDK will read and
	// use to configure the HTTP transport with additional CA certs that are
	// not present in the platforms default CA store.
	//
	// This value will be ignored if the file does not exist.
	//
	//  ca_bundle
	CustomCABundle string

	// EnableEndpointDiscovery can be enabled in the shared config by setting
	// endpoint_discovery_enabled to true
	//
	//	endpoint_discovery_enabled = true
	EnableEndpointDiscovery *bool

	// CSM Options
	CSMEnabled  *bool
	CSMHost     string
	CSMPort     string
	CSMClientID string

	// Specifies the Regional Endpoint flag for the SDK to resolve the endpoint for a service
	//
	// sts_regional_endpoints = regional
	// This can take value as `LegacySTSEndpoint` or `RegionalSTSEndpoint`
	STSRegionalEndpoint endpoints.STSRegionalEndpoint

	// Specifies the Regional Endpoint flag for the SDK to resolve the endpoint for a service
	//
	// s3_us_east_1_regional_endpoint = regional
	// This can take value as `LegacyS3UsEast1Endpoint` or `RegionalS3UsEast1Endpoint`
	S3UsEast1RegionalEndpoint endpoints.S3UsEast1RegionalEndpoint

	// Specifies if the S3 service should allow ARNs to direct the region
	// the client's requests are sent to.
	//
	// s3_use_arn_region=true
	S3UseARNRegion bool
}

type sharedConfigFile struct {
	Filename string
	IniData  ini.Sections
}

// loadSharedConfig retrieves the configuration from the list of files using
// the profile provided. The order the files are listed will determine
// precedence. Values in subsequent files will overwrite values defined in
// earlier files.
//
// For example, given two files A and B. Both define credentials. If the order
// of the files are A then B, B's credential values will be used instead of
// A's.
//
// See sharedConfig.setFromFile for information how the config files
// will be loaded.
func loadSharedConfig(profile string, filenames []string, exOpts bool) (sharedConfig, error) {
	if len(profile) == 0 {
		profile = DefaultSharedConfigProfile
	}

	files, err := loadSharedConfigIniFiles(filenames)
	if err != nil {
		return sharedConfig{}, err
	}

	cfg := sharedConfig{}
	profiles := map[string]struct{}{}
	if err = cfg.setFromIniFiles(profiles, profile, files, exOpts); err != nil {
		return sharedConfig{}, err
	}

	return cfg, nil
}

func loadSharedConfigIniFiles(filenames []string) ([]sharedConfigFile, error) {
	files := make([]sharedConfigFile, 0, len(filenames))

	for _, filename := range filenames {
		sections, err := ini.OpenFile(filename)
		if aerr, ok := err.(awserr.Error); ok && aerr.Code() == ini.ErrCodeUnableToReadFile {
			// Skip files which can't be opened and read for whatever reason
			continue
		} else if err != nil {
			return nil, SharedConfigLoadError{Filename: filename, Err: err}
		}

		files = append(files, sharedConfigFile{
			Filename: filename, IniData: sections,
		})
	}

	return files, nil
}

func (cfg *sharedConfig) setFromIniFiles(profiles map[string]struct{}, profile string, files []sharedConfigFile, exOpts bool) error {
	cfg.Profile = profile

	// Trim files from the list that don't exist.
	var skippedFiles int
	var profileNotFoundErr error
	for _, f := range files {
		if err := cfg.setFromIniFile(profile, f, exOpts); err != nil {
			if _, ok := err.(SharedConfigProfileNotExistsError); ok {
				// Ignore profiles not defined in individual files.
				profileNotFoundErr = err
				skippedFiles++
				continue
			}
			return err
		}
	}
	if skippedFiles == len(files) {
		// If all files were skipped because the profile is not found, return
		// the original profile not found error.
		return profileNotFoundErr
	}

	if _, ok := profiles[profile]; ok {
		// if this is the second instance of the profile the Assume Role
		// options must be cleared because they are only valid for the
		// first reference of a profile. The self linked instance of the
		// profile only have credential provider options.
		cfg.clearAssumeRoleOptions()
	} else {
		// First time a profile has been seen, It must either be a assume role
		// credentials, or SSO. Assert if the credential type requires a role ARN,
		// the ARN is also set, or validate that the SSO configuration is complete.
		if err := cfg.validateCredentialsConfig(profile); err != nil {
			return err
		}
	}
	profiles[profile] = struct{}{}

	if err := cfg.validateCredentialType(); err != nil {
		return err
	}

	// Link source profiles for assume roles
	if len(cfg.SourceProfileName) != 0 {
		// Linked profile via source_profile ignore credential provider
		// options, the source profile must provide the credentials.
		cfg.clearCredentialOptions()

		srcCfg := &sharedConfig{}
		err := srcCfg.setFromIniFiles(profiles, cfg.SourceProfileName, files, exOpts)
		if err != nil {
			// SourceProfile that doesn't exist is an error in configuration.
			if _, ok := err.(SharedConfigProfileNotExistsError); ok {
				err = SharedConfigAssumeRoleError{
					RoleARN:       cfg.RoleARN,
					SourceProfile: cfg.SourceProfileName,
				}
			}
			return err
		}

		if !srcCfg.hasCredentials() {
			return SharedConfigAssumeRoleError{
				RoleARN:       cfg.RoleARN,
				SourceProfile: cfg.SourceProfileName,
			}
		}

		cfg.SourceProfile = srcCfg
	}

	return nil
}

// setFromFile loads the configuration from the file using the profile
// provided. A sharedConfig pointer type value is used so that multiple config
// file loadings can be chained.
//
// Only loads complete logically grouped values, and will not set fields in cfg
// for incomplete grouped values in the config. Such as credentials. For
// example if a config file only includes aws_access_key_id but no
// aws_secret_access_key the aws_access_key_id will be ignored.
func (cfg *sharedConfig) setFromIniFile(profile string, file sharedConfigFile, exOpts bool) error {
	section, ok := file.IniData.GetSection(profile)
	if !ok {
		// Fallback to to alternate profile name: profile <name>
		section, ok = file.IniData.GetSection(fmt.Sprintf("profile %s", profile))
		if !ok {
			return SharedConfigProfileNotExistsError{Profile: profile, Err: nil}
		}
	}

	if exOpts {
		// Assume Role Parameters
		updateString(&cfg.RoleARN, section, roleArnKey)
		updateString(&cfg.ExternalID, section, externalIDKey)
		updateString(&cfg.MFASerial, section, mfaSerialKey)
		updateString(&cfg.RoleSessionName, section, roleSessionNameKey)
		updateString(&cfg.SourceProfileName, section, sourceProfileKey)
		updateString(&cfg.CredentialSource, section, credentialSourceKey)
		updateString(&cfg.Region, section, regionKey)
		updateString(&cfg.CustomCABundle, section, customCABundleKey)

		if section.Has(roleDurationSecondsKey) {
			d := time.Duration(section.Int(roleDurationSecondsKey)) * time.Second
			cfg.AssumeRoleDuration = &d
		}

		if v := section.String(stsRegionalEndpointSharedKey); len(v) != 0 {
			sre, err := endpoints.GetSTSRegionalEndpoint(v)
			if err != nil {
				return fmt.Errorf("failed to load %s from shared config, %s, %v",
					stsRegionalEndpointSharedKey, file.Filename, err)
			}
			cfg.STSRegionalEndpoint = sre
		}

		if v := section.String(s3UsEast1RegionalSharedKey); len(v) != 0 {
			sre, err := endpoints.GetS3UsEast1RegionalEndpoint(v)
			if err != nil {
				return fmt.Errorf("failed to load %s from shared config, %s, %v",
					s3UsEast1RegionalSharedKey, file.Filename, err)
			}
			cfg.S3UsEast1RegionalEndpoint = sre
		}

		// AWS Single Sign-On (AWS SSO)
		updateString(&cfg.SSOAccountID, section, ssoAccountIDKey)
		updateString(&cfg.SSORegion, section, ssoRegionKey)
		updateString(&cfg.SSORoleName, section, ssoRoleNameKey)
		updateString(&cfg.SSOStartURL, section, ssoStartURL)
	}

	updateString(&cfg.CredentialProcess, section, credentialProcessKey)
	updateString(&cfg.WebIdentityTokenFile, section, webIdentityTokenFileKey)

	// Shared Credentials
	creds := credentials.Value{
		AccessKeyID:     section.String(accessKeyIDKey),
		SecretAccessKey: section.String(secretAccessKey),
		SessionToken:    section.String(sessionTokenKey),
		ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", file.Filename),
	}
	if creds.HasKeys() {
		cfg.Creds = creds
	}

	// Endpoint discovery
	updateBoolPtr(&cfg.EnableEndpointDiscovery, section, enableEndpointDiscoveryKey)

	// CSM options
	updateBoolPtr(&cfg.CSMEnabled, section, csmEnabledKey)
	updateString(&cfg.CSMHost, section, csmHostKey)
	updateString(&cfg.CSMPort, section, csmPortKey)
	updateString(&cfg.CSMClientID, section, csmClientIDKey)

	updateBool(&cfg.S3UseARNRegion, section, s3UseARNRegionKey)

	return nil
}

func (cfg *sharedConfig) validateCredentialsConfig(profile string) error {
	if err := cfg.validateCredentialsRequireARN(profile); err != nil {
		return err
	}

	return nil
}

func (cfg *sharedConfig) validateCredentialsRequireARN(profile string) error {
	var credSource string

	switch {
	case len(cfg.SourceProfileName) != 0:
		credSource = sourceProfileKey
	case len(cfg.CredentialSource) != 0:
		credSource = credentialSourceKey
	case len(cfg.WebIdentityTokenFile) != 0:
		credSource = webIdentityTokenFileKey
	}

	if len(credSource) != 0 && len(cfg.RoleARN) == 0 {
		return CredentialRequiresARNError{
			Type:    credSource,
			Profile: profile,
		}
	}

	return nil
}

func (cfg *sharedConfig) validateCredentialType() error {
	// Only one or no credential type can be defined.
	if !oneOrNone(
		len(cfg.SourceProfileName) != 0,
		len(cfg.CredentialSource) != 0,
		len(cfg.CredentialProcess) != 0,
		len(cfg.WebIdentityTokenFile) != 0,
		cfg.hasSSOConfiguration(),
	) {
		return ErrSharedConfigSourceCollision
	}

	return nil
}

func (cfg *sharedConfig) validateSSOConfiguration() error {
	if !cfg.hasSSOConfiguration() {
		return nil
	}

	var missing []string
	if len(cfg.SSOAccountID) == 0 {
		missing = append(missing, ssoAccountIDKey)
	}

	if len(cfg.SSORegion) == 0 {
		missing = append(missing, ssoRegionKey)
	}

	if len(cfg.SSORoleName) == 0 {
		missing = append(missing, ssoRoleNameKey)
	}

	if len(cfg.SSOStartURL) == 0 {
		missing = append(missing, ssoStartURL)
	}

	if len(missing) > 0 {
		return fmt.Errorf("profile %q is configured to use SSO but is missing required configuration: %s",
			cfg.Profile, strings.Join(missing, ", "))
	}

	return nil
}

func (cfg *sharedConfig) hasCredentials() bool {
	switch {
	case len(cfg.SourceProfileName) != 0:
	case len(cfg.CredentialSource) != 0:
	case len(cfg.CredentialProcess) != 0:
	case len(cfg.WebIdentityTokenFile) != 0:
	case cfg.hasSSOConfiguration():
	case cfg.Creds.HasKeys():
	default:
		return false
	}

	return true
}

func (cfg *sharedConfig) clearCredentialOptions() {
	cfg.CredentialSource = ""
	cfg.CredentialProcess = ""
	cfg.WebIdentityTokenFile = ""
	cfg.Creds = credentials.Value{}
}

func (cfg *sharedConfig) clearAssumeRoleOptions() {
	cfg.RoleARN = ""
	cfg.ExternalID = ""
	cfg.MFASerial = ""
	cfg.RoleSessionName = ""
	cfg.SourceProfileName = ""
}

func (cfg *sharedConfig) hasSSOConfiguration() bool {
	switch {
	case len(cfg.SSOAccountID) != 0:
	case len(cfg.SSORegion) != 0:
	case len(cfg.SSORoleName) != 0:
	case len(cfg.SSOStartURL) != 0:
	default:
		return false
	}
	return true
}

func oneOrNone(bs ...bool) bool {
	var count int

	for _, b := range bs {
		if b {
			count++
			if count > 1 {
				return false
			}
		}
	}

	return true
}

// updateString will only update the dst with the value in the section key, key
// is present in the section.
func updateString(dst *string, section ini.Section, key string) {
	if !section.Has(key) {
		return
	}
	*dst = section.String(key)
}

// updateBool will only update the dst with the value in the section key, key
// is present in the section.
func updateBool(dst *bool, section ini.Section, key string) {
	if !section.Has(key) {
		return
	}
	*dst = section.Bool(key)
}

// updateBoolPtr will only update the dst with the value in the section key,
// key is present in the section.
func updateBoolPtr(dst **bool, section ini.Section, key string) {
	if !section.Has(key) {
		return
	}
	*dst = new(bool)
	**dst = section.Bool(key)
}

// SharedConfigLoadError is an error for the shared config file failed to load.
type SharedConfigLoadError struct {
	Filename string
	Err      error
}

// Code is the short id of the error.
func (e SharedConfigLoadError) Code() string {
	return "SharedConfigLoadError"
}

// Message is the description of the error
func (e SharedConfigLoadError) Message() string {
	return fmt.Sprintf("failed to load config file, %s", e.Filename)
}

// OrigErr is the underlying error that caused the failure.
func (e SharedConfigLoadError) OrigErr() error {
	return e.Err
}

// Error satisfies the error interface.
func (e SharedConfigLoadError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", e.Err)
}

// SharedConfigProfileNotExistsError is an error for the shared config when
// the profile was not find in the config file.
type SharedConfigProfileNotExistsError struct {
	Profile string
	Err     error
}

// Code is the short id of the error.
func (e SharedConfigProfileNotExistsError) Code() string {
	return "SharedConfigProfileNotExistsError"
}

// Message is the description of the error
func (e SharedConfigProfileNotExistsError) Message() string {
	return fmt.Sprintf("failed to get profile, %s", e.Profile)
}

// OrigErr is the underlying error that caused the failure.
func (e SharedConfigProfileNotExistsError) OrigErr() error {
	return e.Err
}

// Error satisfies the error interface.
func (e SharedConfigProfileNotExistsError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", e.Err)
}

// SharedConfigAssumeRoleError is an error for the shared config when the
// profile contains assume role information, but that information is invalid
// or not complete.
type SharedConfigAssumeRoleError struct {
	RoleARN       string
	SourceProfile string
}

// Code is the short id of the error.
func (e SharedConfigAssumeRoleError) Code() string {
	return "SharedConfigAssumeRoleError"
}

// Message is the description of the error
func (e SharedConfigAssumeRoleError) Message() string {
	return fmt.Sprintf(
		"failed to load assume role for %s, source profile %s has no shared credentials",
		e.RoleARN, e.SourceProfile,
	)
}

// OrigErr is the underlying error that caused the failure.
func (e SharedConfigAssumeRoleError) OrigErr() error {
	return nil
}

// Error satisfies the error interface.
func (e SharedConfigAssumeRoleError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", nil)
}

// CredentialRequiresARNError provides the error for shared config credentials
// that are incorrectly configured in the shared config or credentials file.
type CredentialRequiresARNError struct {
	// type of credentials that were configured.
	Type string

	// Profile name the credentials were in.
	Profile string
}

// Code is the short id of the error.
func (e CredentialRequiresARNError) Code() string {
	return "CredentialRequiresARNError"
}

// Message is the description of the error
func (e CredentialRequiresARNError) Message() string {
	return fmt.Sprintf(
		"credential type %s requires role_arn, profile %s",
		e.Type, e.Profile,
	)
}

// OrigErr is the underlying error that caused the failure.
func (e CredentialRequiresARNError) OrigErr() error {
	return nil
}

// Error satisfies the error interface.
func (e CredentialRequiresARNError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", nil)
}
