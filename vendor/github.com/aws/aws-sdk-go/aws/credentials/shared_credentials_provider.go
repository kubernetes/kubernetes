package credentials

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/go-ini/ini"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

// SharedCredsProviderName provides a name of SharedCreds provider
const SharedCredsProviderName = "SharedCredentialsProvider"

var (
	// ErrSharedCredentialsHomeNotFound is emitted when the user directory cannot be found.
	//
	// @readonly
	ErrSharedCredentialsHomeNotFound = awserr.New("UserHomeNotFound", "user home directory not found.", nil)
)

// A SharedCredentialsProvider retrieves credentials from the current user's home
// directory, and keeps track if those credentials are expired.
//
// Profile ini file example: $HOME/.aws/credentials
type SharedCredentialsProvider struct {
	// Path to the shared credentials file.
	//
	// If empty will look for "AWS_SHARED_CREDENTIALS_FILE" env variable. If the
	// env value is empty will default to current user's home directory.
	// Linux/OSX: "$HOME/.aws/credentials"
	// Windows:   "%USERPROFILE%\.aws\credentials"
	Filename string

	// AWS Profile to extract credentials from the shared credentials file. If empty
	// will default to environment variable "AWS_PROFILE" or "default" if
	// environment variable is also not set.
	Profile string

	// retrieved states if the credentials have been successfully retrieved.
	retrieved bool
}

// NewSharedCredentials returns a pointer to a new Credentials object
// wrapping the Profile file provider.
func NewSharedCredentials(filename, profile string) *Credentials {
	return NewCredentials(&SharedCredentialsProvider{
		Filename: filename,
		Profile:  profile,
	})
}

// Retrieve reads and extracts the shared credentials from the current
// users home directory.
func (p *SharedCredentialsProvider) Retrieve() (Value, error) {
	p.retrieved = false

	filename, err := p.filename()
	if err != nil {
		return Value{ProviderName: SharedCredsProviderName}, err
	}

	creds, err := loadProfile(filename, p.profile())
	if err != nil {
		return Value{ProviderName: SharedCredsProviderName}, err
	}

	p.retrieved = true
	return creds, nil
}

// IsExpired returns if the shared credentials have expired.
func (p *SharedCredentialsProvider) IsExpired() bool {
	return !p.retrieved
}

// loadProfiles loads from the file pointed to by shared credentials filename for profile.
// The credentials retrieved from the profile will be returned or error. Error will be
// returned if it fails to read from the file, or the data is invalid.
func loadProfile(filename, profile string) (Value, error) {
	config, err := ini.Load(filename)
	if err != nil {
		return Value{ProviderName: SharedCredsProviderName}, awserr.New("SharedCredsLoad", "failed to load shared credentials file", err)
	}
	iniProfile, err := config.GetSection(profile)
	if err != nil {
		return Value{ProviderName: SharedCredsProviderName}, awserr.New("SharedCredsLoad", "failed to get profile", err)
	}

	id, err := iniProfile.GetKey("aws_access_key_id")
	if err != nil {
		return Value{ProviderName: SharedCredsProviderName}, awserr.New("SharedCredsAccessKey",
			fmt.Sprintf("shared credentials %s in %s did not contain aws_access_key_id", profile, filename),
			err)
	}

	secret, err := iniProfile.GetKey("aws_secret_access_key")
	if err != nil {
		return Value{ProviderName: SharedCredsProviderName}, awserr.New("SharedCredsSecret",
			fmt.Sprintf("shared credentials %s in %s did not contain aws_secret_access_key", profile, filename),
			nil)
	}

	// Default to empty string if not found
	token := iniProfile.Key("aws_session_token")

	return Value{
		AccessKeyID:     id.String(),
		SecretAccessKey: secret.String(),
		SessionToken:    token.String(),
		ProviderName:    SharedCredsProviderName,
	}, nil
}

// filename returns the filename to use to read AWS shared credentials.
//
// Will return an error if the user's home directory path cannot be found.
func (p *SharedCredentialsProvider) filename() (string, error) {
	if p.Filename == "" {
		if p.Filename = os.Getenv("AWS_SHARED_CREDENTIALS_FILE"); p.Filename != "" {
			return p.Filename, nil
		}

		homeDir := os.Getenv("HOME") // *nix
		if homeDir == "" {           // Windows
			homeDir = os.Getenv("USERPROFILE")
		}
		if homeDir == "" {
			return "", ErrSharedCredentialsHomeNotFound
		}

		p.Filename = filepath.Join(homeDir, ".aws", "credentials")
	}

	return p.Filename, nil
}

// profile returns the AWS shared credentials profile.  If empty will read
// environment variable "AWS_PROFILE". If that is not set profile will
// return "default".
func (p *SharedCredentialsProvider) profile() string {
	if p.Profile == "" {
		p.Profile = os.Getenv("AWS_PROFILE")
	}
	if p.Profile == "" {
		p.Profile = "default"
	}

	return p.Profile
}
