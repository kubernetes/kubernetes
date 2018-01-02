// +build integration

//Package elastictranscoder provides gucumber integration tests support.
package elastictranscoder

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/elastictranscoder"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@elastictranscoder", func() {
		gucumber.World["client"] = elastictranscoder.New(smoke.Session)
	})
}
