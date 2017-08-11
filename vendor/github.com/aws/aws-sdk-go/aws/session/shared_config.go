package session

import (
	"fmt"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/go-ini/ini"
)

const (
	// Static Credentials group
	accessKeyIDKey  = `aws_access_key_id`     // group required
	secretAccessKey = `aws_secret_access_key` // group required
	sessionTokenKey = `aws_session_token`     // optional

	// Assume Role Credentials group
	roleArnKey         = `role_arn`          // group required
	sourceProfileKey   = `source_profile`    // group required
	externalIDKey      = `external_id`       // optional
	mfaSerialKey       = `mfa_serial`        // optional
	roleSessionNameKey = `role_session_name` // optional

	// Additional Config fields
	regionKey = `region`

	// DefaultSharedConfigProfile is the default profile to be used when
	// loading configuration from the config files if another profile name
	// is not provided.
	DefaultSharedConfigProfile = `default`
)

type assumeRoleConfig struct {
	RoleARN         string
	SourceProfile   string
	ExternalID      string
	MFASerial       string
	RoleSessionName string
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

	// Region is the region the SDK should use for looking up AWS service endpoints
	// and signing requests.
	//
	//	region
	Region string
}

type sharedConfigFile struct {
	Filename string
	IniData  *ini.File
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
		b, err := ioutil.ReadFile(filename)
		if err != nil {
			// Skip files which can't be opened and read for whatever reason
			continue
		}

		f, err := ini.Load(b)
		if err != nil {
			return nil, SharedConfigLoadError{Filename: filename, Err: err}
		}

		files = append(files, sharedConfigFile{
			Filename: filename, IniData: f,
		})
	}

	return files, nil
}

func (cfg *sharedConfig) setAssumeRoleSource(origProfile string, files []sharedConfigFile) error {
	var assumeRoleSrc sharedConfig

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
	section, err := file.IniData.GetSection(profile)
	if err != nil {
		// Fallback to to alternate profile name: profile <name>
		section, err = file.IniData.GetSection(fmt.Sprintf("profile %s", profile))
		if err != nil {
			return SharedConfigProfileNotExistsError{Profile: profile, Err: err}
		}
	}

	// Shared Credentials
	akid := section.Key(accessKeyIDKey).String()
	secret := section.Key(secretAccessKey).String()
	if len(akid) > 0 && len(secret) > 0 {
		cfg.Creds = credentials.Value{
			AccessKeyID:     akid,
			SecretAccessKey: secret,
			SessionToken:    section.Key(sessionTokenKey).String(),
			ProviderName:    fmt.Sprintf("SharedConfigCredentials: %s", file.Filename),
		}
	}

	// Assume Role
	roleArn := section.Key(roleArnKey).String()
	srcProfile := section.Key(sourceProfileKey).String()
	if len(roleArn) > 0 && len(srcProfile) > 0 {
		cfg.AssumeRole = assumeRoleConfig{
			RoleARN:         roleArn,
			SourceProfile:   srcProfile,
			ExternalID:      section.Key(externalIDKey).String(),
			MFASerial:       section.Key(mfaSerialKey).String(),
			RoleSessionName: section.Key(roleSessionNameKey).String(),
		}
	}

	// Region
	if v := section.Key(regionKey).String(); len(v) > 0 {
		cfg.Region = v
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
