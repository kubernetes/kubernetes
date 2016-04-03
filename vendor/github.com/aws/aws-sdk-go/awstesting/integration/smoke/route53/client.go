//Package route53 provides gucumber integration tests support.
package route53

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/route53"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@route53", func() {
		World["client"] = route53.New(smoke.Session)
	})
}
