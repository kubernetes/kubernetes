//Package apigateway provides gucumber integration tests support.
package apigateway

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/apigateway"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@apigateway", func() {
		World["client"] = apigateway.New(smoke.Session)
	})
}
