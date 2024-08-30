//go:build windows

package hcnshim

import (
	"github.com/Microsoft/hcnshim/internal/hns"
)

type HNSGlobals = hns.HNSGlobals
type HNSVersion = hns.HNSVersion

var (
	HNSVersion1803 = hns.HNSVersion1803
)

func GetHNSGlobals() (*HNSGlobals, error) {
	return hns.GetHNSGlobals()
}
