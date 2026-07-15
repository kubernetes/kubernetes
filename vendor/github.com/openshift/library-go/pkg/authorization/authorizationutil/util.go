package authorizationutil

import (
	"context"
	"errors"

	authorizationv1 "k8s.io/api/authorization/v1"
	kerrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	authorizationclient "k8s.io/client-go/kubernetes/typed/authorization/v1"
)

// AddUserToSAR adds the requisite user information to a SubjectAccessReview.
// It returns the modified SubjectAccessReview.
func AddUserToSAR(user user.Info, sar *authorizationv1.SubjectAccessReview) *authorizationv1.SubjectAccessReview {
	sar.Spec.User = user.GetName()
	// reminiscent of the bad old days of C.  Copies copy the min number of elements of both source and dest
	sar.Spec.Groups = make([]string, len(user.GetGroups()))
	copy(sar.Spec.Groups, user.GetGroups())
	sar.Spec.Extra = map[string]authorizationv1.ExtraValue{}

	for k, v := range user.GetExtra() {
		sar.Spec.Extra[k] = authorizationv1.ExtraValue(v)
	}

	return sar
}

// Authorize verifies that a given user is permitted to carry out a given
// action.  If this cannot be determined, or if the user is not permitted, an
// error is returned.
func Authorize(sarClient authorizationclient.SubjectAccessReviewInterface, user user.Info, resourceAttributes *authorizationv1.ResourceAttributes) error {
	sar := AddUserToSAR(user, &authorizationv1.SubjectAccessReview{
		Spec: authorizationv1.SubjectAccessReviewSpec{
			ResourceAttributes: resourceAttributes,
		},
	})

	resp, err := sarClient.Create(context.TODO(), sar, metav1.CreateOptions{})
	if err == nil && resp != nil && resp.Status.Allowed {
		return nil
	}

	if err == nil {
		err = errors.New(resp.Status.Reason)
	}
	return kerrors.NewForbidden(schema.GroupResource{Group: resourceAttributes.Group, Resource: resourceAttributes.Resource}, resourceAttributes.Name, err)
}
