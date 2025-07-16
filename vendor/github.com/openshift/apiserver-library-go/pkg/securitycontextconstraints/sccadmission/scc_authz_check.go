package sccadmission

import (
	"context"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"

	"github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccmatching"
)

const (
	allowedForUser = "user"
	allowedForSA   = "serviceaccount"
	allowedForNone = "none"
)

type sccAuthorizationChecker struct {
	authz              authorizer.Authorizer
	userInfo           user.Info
	namespace          string
	serviceAccountName string
}

func newSCCAuthorizerChecker(authz authorizer.Authorizer, attr admission.Attributes, serviceAccountName string) *sccAuthorizationChecker {
	return &sccAuthorizationChecker{
		authz:              authz,
		userInfo:           attr.GetUserInfo(),
		namespace:          attr.GetNamespace(),
		serviceAccountName: serviceAccountName,
	}
}

func (c *sccAuthorizationChecker) allowedForType(ctx context.Context, provider sccmatching.SecurityContextConstraintsProvider) string {
	sccName := provider.GetSCCName()
	sccUsers := provider.GetSCCUsers()
	sccGroups := provider.GetSCCGroups()

	if len(c.serviceAccountName) != 0 {
		saUserInfo := serviceaccount.UserInfo(c.namespace, c.serviceAccountName, "")

		if sccmatching.ConstraintAppliesTo(
			ctx,
			sccName, sccUsers, sccGroups,
			saUserInfo, c.namespace, c.authz,
		) {
			return allowedForSA
		}
	}

	if sccmatching.ConstraintAppliesTo(
		ctx,
		sccName, sccUsers, sccGroups,
		c.userInfo, c.namespace, c.authz,
	) {
		return allowedForUser
	}

	return allowedForNone
}
