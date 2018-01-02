// +build integration

//Package performance provides gucumber integration tests support.
package performance

import (
	"github.com/gucumber/gucumber"
)

func init() {
	gucumber.Before("@performance", func() {
	})
}
