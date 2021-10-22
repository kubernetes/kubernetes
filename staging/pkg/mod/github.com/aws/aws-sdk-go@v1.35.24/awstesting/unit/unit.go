// Package unit performs initialization and validation for unit tests
package unit

import (
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
)

// Session is a shared session for unit tests to use.
var Session = session.Must(session.NewSession(&aws.Config{
	Credentials: credentials.NewStaticCredentials("AKID", "SECRET", "SESSION"),
	Region:      aws.String("mock-region"),
	SleepDelay:  func(time.Duration) {},
}))
