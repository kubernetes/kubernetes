// +build integration

//Package machinelearning provides gucumber integration tests support.
package machinelearning

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/machinelearning"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@machinelearning", func() {
		gucumber.World["client"] = machinelearning.New(smoke.Session)
	})
}
