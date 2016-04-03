//Package cloudsearch provides gucumber integration tests support.
package cloudsearch

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/cloudsearch"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@cloudsearch", func() {
		World["client"] = cloudsearch.New(smoke.Session)
	})
}
