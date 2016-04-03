//Package es provides gucumber integration tests support.
package es

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/elasticsearchservice"
	. "github.com/lsegal/gucumber"
)

func init() {
	Before("@es", func() {
		World["client"] = elasticsearchservice.New(smoke.Session)
	})
}
