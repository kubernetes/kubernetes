//Package machinelearning provides gucumber integration tests support.
package machinelearning

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/machinelearning"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@machinelearning", func() {
		World["client"] = machinelearning.New(smoke.Session)
	})
}
