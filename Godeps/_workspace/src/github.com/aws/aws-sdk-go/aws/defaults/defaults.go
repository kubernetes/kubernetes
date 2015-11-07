package defaults

import (
	"net/http"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
)

// DefaultChainCredentials is a Credentials which will find the first available
// credentials Value from the list of Providers.
//
// This should be used in the default case. Once the type of credentials are
// known switching to the specific Credentials will be more efficient.
var DefaultChainCredentials = credentials.NewChainCredentials(
	[]credentials.Provider{
		&credentials.EnvProvider{},
		&credentials.SharedCredentialsProvider{Filename: "", Profile: ""},
		&ec2rolecreds.EC2RoleProvider{ExpiryWindow: 5 * time.Minute},
	})

// DefaultConfig is the default all service configuration will be based off of.
// By default, all clients use this structure for initialization options unless
// a custom configuration object is passed in.
//
// You may modify this global structure to change all default configuration
// in the SDK. Note that configuration options are copied by value, so any
// modifications must happen before constructing a client.
var DefaultConfig = aws.NewConfig().
	WithCredentials(DefaultChainCredentials).
	WithRegion(os.Getenv("AWS_REGION")).
	WithHTTPClient(http.DefaultClient).
	WithMaxRetries(aws.DefaultRetries).
	WithLogger(aws.NewDefaultLogger()).
	WithLogLevel(aws.LogOff).
	WithSleepDelay(time.Sleep)
