package imagepolicy

import (
	imagepolicyapiv1 "github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1"
)

// RequestsResolution returns true if you should attempt to resolve image pull specs
func RequestsResolution(imageResolutionType imagepolicyapiv1.ImageResolutionType) bool {
	switch imageResolutionType {
	case imagepolicyapiv1.RequiredRewrite, imagepolicyapiv1.Required, imagepolicyapiv1.AttemptRewrite, imagepolicyapiv1.Attempt:
		return true
	}
	return false
}

// FailOnResolutionFailure returns true if you should fail when resolution fails
func FailOnResolutionFailure(imageResolutionType imagepolicyapiv1.ImageResolutionType) bool {
	switch imageResolutionType {
	case imagepolicyapiv1.RequiredRewrite, imagepolicyapiv1.Required:
		return true
	}
	return false
}

// RewriteImagePullSpec returns true if you should rewrite image pull specs when resolution succeeds
func RewriteImagePullSpec(imageResolutionType imagepolicyapiv1.ImageResolutionType) bool {
	switch imageResolutionType {
	case imagepolicyapiv1.RequiredRewrite, imagepolicyapiv1.AttemptRewrite:
		return true
	}
	return false
}
