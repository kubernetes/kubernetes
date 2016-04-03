//Package cloudwatch provides gucumber integration tests support.
package cloudwatch

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudwatch"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@cloudwatch", func() {
		World["client"] = cloudwatch.New(smoke.Session)
	})
}
