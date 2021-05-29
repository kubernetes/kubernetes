package session

import (
	"fmt"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/processcreds"
	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/shareddefaults"
)

func resolveCredentials(cfg *aws.Config,
	envCfg envConfig, sharedCfg sharedConfig,
	handlers request.Handlers,
	sessOpts Options,
) (*credentials.Credentials, error) {

	switch {
	case len(sessOpts.Profile) != 0:
		// User explicitly provided an Profile in the session's configuration
		// so load that profile from shared config first.
		// Github(aws/aws-sdk-go#2727)
		return resolveCredsFromProfile(cfg, envCfg, sharedCfg, handlers, sessOpts)

	case envCfg.Creds.HasKeys():
		// Environment credentials
		return credentials.NewStaticCredentialsFromCreds(envCfg.Creds), nil

	case len(envCfg.WebIdentityTokenFilePath) != 0:
		// Web identity token from environment, RoleARN required to also be
		// set.
		return assumeWebIdentity(cfg, handlers,
			envCfg.WebIdentityTokenFilePath,
			envCfg.RoleARN,
			envCfg.RoleSessionName,
		)

	default:
		// Fallback to the "default" credential resolution chain.
		return resolveCredsFromProfile(cfg, envCfg, sharedCfg, handlers, sessOpts)
	}
}

// WebIdentityEmptyRoleARNErr will occur if 'AWS_WEB_IDENTITY_TOKEN_FILE' was set but
// 'AWS_ROLE_ARN' was not set.
var WebIdentityEmptyRoleARNErr = awserr.New(stscreds.ErrCodeWebIdentity, "role ARN is not set", nil)

// WebIdentityEmptyTokenFilePathErr will occur if 'AWS_ROLE_ARN' was set but
// 'AWS_WEB_IDENTITY_TOKEN_FILE' was not set.
var WebIdentityEmptyTokenFilePathErr = awserr.New(stscreds.ErrCodeWebIdentity, "token file path is not set", nil)

func assumeWebIdentity(cfg *aws.Config, handlers request.Handlers,
	filepath string,
	roleARN, sessionName string,
) (*credentials.Credentials, error) {

	if len(filepath) == 0 {
		return nil, WebIdentityEmptyTokenFilePathErr
	}

	if len(roleARN) == 0 {
		return nil, WebIdentityEmptyRoleARNErr
	}

	creds := stscreds.NewWebIdentityCredentials(
		&Session{
			Config:   cfg,
			Handlers: handlers.Copy(),
		},
		roleARN,
		sessionName,
		filepath,
	)

	return creds, nil
}

func resolveCredsFromProfile(cfg *aws.Config,
	envCfg envConfig, sharedCfg sharedConfig,
	handlers request.Handlers,
	sessOpts Options,
) (creds *credentials.Credentials, err error) {

	switch {
	case sharedCfg.SourceProfile != nil:
		// Assume IAM role with credentials source from a different profile.
		creds, err = resolveCredsFromProfile(cfg, envCfg,
			*sharedCfg.SourceProfile, handlers, sessOpts,
		)

	case sharedCfg.Creds.HasKeys():
		// Static Credentials from Shared Config/Credentials file.
		creds = credentials.NewStaticCredentialsFromCreds(
			sharedCfg.Creds,
		)

	case len(sharedCfg.CredentialProcess) != 0:
		// Get credentials from CredentialProcess
		creds = processcreds.NewCredentials(sharedCfg.CredentialProcess)

	case len(sharedCfg.CredentialSource) != 0:
		creds, err = resolveCredsFromSource(cfg, envCfg,
			sharedCfg, handlers, sessOpts,
		)

	case len(sharedCfg.WebIdentityTokenFile) != 0:
		// Credentials from Assume Web Identity token require an IAM Role, and
		// that roll will be assumed. May be wrapped with another assume role
		// via SourceProfile.
		return assumeWebIdentity(cfg, handlers,
			sharedCfg.WebIdentityTokenFile,
			sharedCfg.RoleARN,
			sharedCfg.RoleSessionName,
		)

	default:
		// Fallback to default credentials provider, include mock errors for
		// the credential chain so user can identify why credentials failed to
		// be retrieved.
		creds = credentials.NewCredentials(&credentials.ChainProvider{
			VerboseErrors: aws.BoolValue(cfg.CredentialsChainVerboseErrors),
			Providers: []credentials.Provider{
				&credProviderError{
					Err: awserr.New("EnvAccessKeyNotFound",
						"failed to find credentials in the environment.", nil),
				},
				&credProviderError{
					Err: awserr.New("SharedCredsLoad",
						fmt.Sprintf("failed to load profile, %s.", envCfg.Profile), nil),
				},
				defaults.RemoteCredProvider(*cfg, handlers),
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(sharedCfg.RoleARN) > 0 {
		cfgCp := *cfg
		cfgCp.Credentials = creds
		return credsFromAssumeRole(cfgCp, handlers, sharedCfg, sessOpts)
	}

	return creds, nil
}

// valid credential source values
const (
	credSourceEc2Metadata  = "Ec2InstanceMetadata"
	credSourceEnvironment  = "Environment"
	credSourceECSContainer = "EcsContainer"
)

func resolveCredsFromSource(cfg *aws.Config,
	envCfg envConfig, sharedCfg sharedConfig,
	handlers request.Handlers,
	sessOpts Options,
) (creds *credentials.Credentials, err error) {

	switch sharedCfg.CredentialSource {
	case credSourceEc2Metadata:
		p := defaults.RemoteCredProvider(*cfg, handlers)
		creds = credentials.NewCredentials(p)

	case credSourceEnvironment:
		creds = credentials.NewStaticCredentialsFromCreds(envCfg.Creds)

	case credSourceECSContainer:
		if len(os.Getenv(shareddefaults.ECSCredsProviderEnvVar)) == 0 {
			return nil, ErrSharedConfigECSContainerEnvVarEmpty
		}

		p := defaults.RemoteCredProvider(*cfg, handlers)
		creds = credentials.NewCredentials(p)

	default:
		return nil, ErrSharedConfigInvalidCredSource
	}

	return creds, nil
}

func credsFromAssumeRole(cfg aws.Config,
	handlers request.Handlers,
	sharedCfg sharedConfig,
	sessOpts Options,
) (*credentials.Credentials, error) {

	if len(sharedCfg.MFASerial) != 0 && sessOpts.AssumeRoleTokenProvider == nil {
		// AssumeRole Token provider is required if doing Assume Role
		// with MFA.
		return nil, AssumeRoleTokenProviderNotSetError{}
	}

	return stscreds.NewCredentials(
		&Session{
			Config:   &cfg,
			Handlers: handlers.Copy(),
		},
		sharedCfg.RoleARN,
		func(opt *stscreds.AssumeRoleProvider) {
			opt.RoleSessionName = sharedCfg.RoleSessionName

			if sessOpts.AssumeRoleDuration == 0 &&
				sharedCfg.AssumeRoleDuration != nil &&
				*sharedCfg.AssumeRoleDuration/time.Minute > 15 {
				opt.Duration = *sharedCfg.AssumeRoleDuration
			} else if sessOpts.AssumeRoleDuration != 0 {
				opt.Duration = sessOpts.AssumeRoleDuration
			}

			// Assume role with external ID
			if len(sharedCfg.ExternalID) > 0 {
				opt.ExternalID = aws.String(sharedCfg.ExternalID)
			}

			// Assume role with MFA
			if len(sharedCfg.MFASerial) > 0 {
				opt.SerialNumber = aws.String(sharedCfg.MFASerial)
				opt.TokenProvider = sessOpts.AssumeRoleTokenProvider
			}
		},
	), nil
}

// AssumeRoleTokenProviderNotSetError is an error returned when creating a
// session when the MFAToken option is not set when shared config is configured
// load assume a role with an MFA token.
type AssumeRoleTokenProviderNotSetError struct{}

// Code is the short id of the error.
func (e AssumeRoleTokenProviderNotSetError) Code() string {
	return "AssumeRoleTokenProviderNotSetError"
}

// Message is the description of the error
func (e AssumeRoleTokenProviderNotSetError) Message() string {
	return fmt.Sprintf("assume role with MFA enabled, but AssumeRoleTokenProvider session option not set.")
}

// OrigErr is the underlying error that caused the failure.
func (e AssumeRoleTokenProviderNotSetError) OrigErr() error {
	return nil
}

// Error satisfies the error interface.
func (e AssumeRoleTokenProviderNotSetError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", nil)
}

type credProviderError struct {
	Err error
}

func (c credProviderError) Retrieve() (credentials.Value, error) {
	return credentials.Value{}, c.Err
}
func (c credProviderError) IsExpired() bool {
	return true
}
