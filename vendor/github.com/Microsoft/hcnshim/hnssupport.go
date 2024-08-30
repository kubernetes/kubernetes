//go:build windows

package hcnshim

import (
	"github.com/Microsoft/hcnshim/internal/hns"
)

type HNSSupportedFeatures = hns.HNSSupportedFeatures

type HNSAclFeatures = hns.HNSAclFeatures

func GetHNSSupportedFeatures() HNSSupportedFeatures {
	return hns.GetHNSSupportedFeatures()
}
