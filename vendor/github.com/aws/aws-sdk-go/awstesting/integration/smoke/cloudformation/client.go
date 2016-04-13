//Package cloudformation provides gucumber integration tests support.
package cloudformation

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudformation"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@cloudformation", func() {
		World["client"] = cloudformation.New(smoke.Session)
	})
}
