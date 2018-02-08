package defaults

import (
	"github.com/aws/aws-sdk-go/internal/shareddefaults"
)

// SharedCredentialsFilename returns the SDK's default file path
// for the shared credentials file.
//
// Builds the shared config file path based on the OS's platform.
//
//   - Linux/Unix: $HOME/.aws/credentials
//   - Windows: %USERPROFILE%\.aws\credentials
func SharedCredentialsFilename() string {
	return shareddefaults.SharedCredentialsFilename()
}

// SharedConfigFilename returns the SDK's default file path for
// the shared config file.
//
// Builds the shared config file path based on the OS's platform.
//
//   - Linux/Unix: $HOME/.aws/config
//   - Windows: %USERPROFILE%\.aws\config
func SharedConfigFilename() string {
	return shareddefaults.SharedConfigFilename()
}
