//Package codecommit provides gucumber integration tests support.
package codecommit

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/codecommit"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@codecommit", func() {
		World["client"] = codecommit.New(smoke.Session)
	})
}
