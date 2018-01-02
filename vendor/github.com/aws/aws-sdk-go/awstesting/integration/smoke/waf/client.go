// +build integration

//Package waf provides gucumber integration tests support.
package waf

import (
	"github.com/aws/aws-sdk-go/awstesting/integration/smoke"
	"github.com/aws/aws-sdk-go/service/waf"
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@waf", func() {
		gucumber.World["client"] = waf.New(smoke.Session)
	})
}
