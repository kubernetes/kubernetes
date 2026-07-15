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
