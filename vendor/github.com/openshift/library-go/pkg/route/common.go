package route

import (
	"context"

	authorizationv1 "k8s.io/api/authorization/v1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SubjectAccessReviewCreator is an interface for performing subject access reviews
type SubjectAccessReviewCreator interface {
	Create(ctx context.Context, sar *authorizationv1.SubjectAccessReview, opts metav1.CreateOptions) (*authorizationv1.SubjectAccessReview, error)
}

// RouteValidationOptions used to tweak how/what fields are validated. These
// options are propagated by the apiserver.
type RouteValidationOptions struct {

	// AllowExternalCertificates option is set when the RouteExternalCertificate
	// feature gate is enabled.
	AllowExternalCertificates bool
}
