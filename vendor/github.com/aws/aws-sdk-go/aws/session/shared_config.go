package session

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"

	"github.com/aws/aws-sdk-go/internal/ini"
)

const (
	// Static Credentials group
	accessKeyIDKey  = `aws_access_key_id`     // group required
	secretAccessKey = `aws_secret_access_key` // group required
	sessionTokenKey = `aws_session_token`     // optional

	// Assume Role Credentials group
	roleArnKey          = `role_arn`          // group required
	sourceProfileKey    = `source_profile`    // group required (or credential_source)
	credentialSourceKey = `credential_source` // group required (or source_profile)
	externalIDKey       = `external_id`       // optional
	mfaSerialKey        = `mfa_serial`        // optional
	roleSessionNameKey  = `role_session_name` // optional

	// Additional Config fields
	regionKey = `region`

	// endpoint discovery group
	enableEndpointDiscoveryKey = `endpoint_discovery_enabled` // optional
	// External Credential Process
	credentialProcessKey = `credential_process`

	// DefaultSharedConfigProfile is the default profile to be used when
	// loading configuration from the config files if another profile name
	// is not provided.
	DefaultSharedConfigProfile = `default`
)

type assumeRoleConfig struct {
	RoleARN          string
	SourceProfile    string
	CredentialSource string
	ExternalID       string
	MFASerial        string
	RoleSessionName  string
}

// sharedConfig represents the configuration fields of the SDK config files.
type sharedConfig struct {
	// Credentials values from the config file. Both aws_access_key_id
	// and aws_secret_access_key must be provided together in the same file
	// to be considered valid. The values will be ignored if not a complete group.
	// aws_session_token is an optional field that can be provided if both of the
	// other two fields are also provided.
	//
	//	aws_access_key_id
	//	aws_secret_access_key
	//	aws_session_token
	Creds credentials.Value

	AssumeRole       assumeRoleConfig
	AssumeRoleSource *sharedConfig

	// An external process to request credentials
	CredentialProcess string

	// Region is the region the SDK should use for looking up AWS service endpoints
	// and signing requests.
	//
	//	region
	Region string

	// EnableEndpointDiscovery can be enabled in the shared config by setting
	// endpoint_discovery_enabled to true
	//
	//	endpoint_discovery_enabled = true
	EnableEndpointDiscovery *bool
}

type sharedConfigFile struct {
	Filename string
	IniData  ini.Sections
}

// loadSharedConfig retrieves the configuration from the list of files
// using the profile provided. The order the files are listed will determine
// precedence. Values in subsequent files will overwrite values defined in
// earlier files.
//
// For example, given two files A and B. Both define credentials. If the order
// of the files are A then B, B's credential values will be used instead of A's.
//
// See sharedConfig.setFromFile for information how the config files
// will be loaded.
func loadSharedConfig(profile string, filenames []string) (sharedConfig, error) {
	if len(profile) == 0 {
		profile = DefaultSharedConfigProfile
	}

	files, err := loadSharedConfigIniFiles(filenames)
	if err != nil {
		return sharedConfig{}, err
	}

	cfg := sharedConfig{}
	if err = cfg.setFromIniFiles(profile, files); err != nil {
		return sharedConfig{}, err
	}

	if len(cfg.AssumeRole.SourceProfile) > 0 {
		if err := cfg.setAssumeRoleSource(profile, files); err != nil {
			return sharedConfig{}, err
		}
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

func (cfg *sharedConfig) setAssumeRoleSource(origProfile string, files []sharedConfigFile) error {
	var assumeRoleSrc sharedConfig

	if len(cfg.AssumeRole.CredentialSource) > 0 {
		// setAssumeRoleSource is only called when source_profile is found.
		// If both source_profile and credential_source are set, then
		// ErrSharedConfigSourceCollision will be returned
		return ErrSharedConfigSourceCollision
	}

	// Multiple level assume role chains are not support
	if cfg.AssumeRole.SourceProfile == origProfile {
		assumeRoleSrc = *cfg
		assumeRoleSrc.AssumeRole = assumeRoleConfig{}
	} else {
		err := assumeRoleSrc.setFromIniFiles(cfg.AssumeRole.SourceProfile, files)
		if err != nil {
			return err
		}
	}

	if len(assumeRoleSrc.Creds.AccessKeyID) == 0 {
		return SharedConfigAssumeRoleError{RoleARN: cfg.AssumeRole.RoleARN}
	}

	cfg.AssumeRoleSource = &assumeRoleSrc

	return nil
}

func (cfg *sharedConfig) setFromIniFiles(profile string, files []sharedConfigFile) error {
	// Trim files from the list that don't exist.
	for _, f := range files {
		if err := cfg.setFromIniFile(profile, f); err != nil {
			if _, ok := err.(SharedConfigProfileNotExistsError); ok {
				// Ignore proviles missings
				continue
			}
			return err
		}
	}

	return nil
}

// setFromFile loads the configuration from the file using
// the profile provided. A sharedConfig pointer type value is used so that
// multiple config file loadings can be chained.
//
// Only loads complete logically grouped values, and will not set fields in cfg
// for incomplete grouped values in the config. Such as credentials. For example
// if a config file only includes aws_access_key_id but no aws_secret_access_key
// the aws_access_key_id will be ignored.
func (cfg *sharedConfig) setFromIniFile(profile string, file sharedConfigFile) error {
	section, ok := file.IniData.GetSection(profile)
	if !ok {
		// Fallback to to alternate profile name: profile <name>
		section, ok = file.IniData.GetSection(fmt.Sprintf("profile %s", profile))
		if !ok {
			return SharedConfigProfileNotExistsError{Profile: profile, Err: nil}
		}
	}

	// Shared Credentials
	akid := section.String(accessKeyIDKey)
	secret := section.String(secretAccessKey)
	if len(akid) > 0 && len(secret) > 0 {
		cfg.Creds = credentials.Value{
			AccessKeyID:     akid,
			SecretAccessKey: secret,
			SessionToken:    section.String(sessionTokenKey),
			ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", file.Filename),
		}
	}

	// Assume Role
	roleArn := section.String(roleArnKey)
	srcProfile := section.String(sourceProfileKey)
	credentialSource := section.String(credentialSourceKey)
	hasSource := len(srcProfile) > 0 || len(credentialSource) > 0
	if len(roleArn) > 0 && hasSource {
		cfg.AssumeRole = assumeRoleConfig{
			RoleARN:          roleArn,
			SourceProfile:    srcProfile,
			CredentialSource: credentialSource,
			ExternalID:       section.String(externalIDKey),
			MFASerial:        section.String(mfaSerialKey),
			RoleSessionName:  section.String(roleSessionNameKey),
		}
	}

	// `credential_process`
	if credProc := section.String(credentialProcessKey); len(credProc) > 0 {
		cfg.CredentialProcess = credProc
	}

	// Region
	if v := section.String(regionKey); len(v) > 0 {
		cfg.Region = v
	}

	// Endpoint discovery
	if section.Has(enableEndpointDiscoveryKey) {
		v := section.Bool(enableEndpointDiscoveryKey)
		cfg.EnableEndpointDiscovery = &v
	}

	return nil
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
	RoleARN string
}

// Code is the short id of the error.
func (e SharedConfigAssumeRoleError) Code() string {
	return "SharedConfigAssumeRoleError"
}

// Message is the description of the error
func (e SharedConfigAssumeRoleError) Message() string {
	return fmt.Sprintf("failed to load assume role for %s, source profile has no shared credentials",
		e.RoleARN)
}

// OrigErr is the underlying error that caused the failure.
func (e SharedConfigAssumeRoleError) OrigErr() error {
	return nil
}

// Error satisfies the error interface.
func (e SharedConfigAssumeRoleError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", nil)
}
