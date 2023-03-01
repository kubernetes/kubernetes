package selfsubjectaccessreview

import (
	"reflect"
	"sort"

	"k8s.io/apiserver/pkg/authentication/user"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	authorizationscope "github.com/openshift/apiserver-library-go/pkg/authorization/scope"
)

func userWithRequiredScopes(userToCheck user.Info) user.Info {
	userExtra := userToCheck.GetExtra()
	if userExtra == nil || !scopesNeedUserFull(userExtra[authorizationv1.ScopesKey]) {
		return userToCheck
	}

	userExtraCopy := make(map[string][]string)
	for k, v := range userExtra {
		userExtraCopy[k] = v
	}
	userExtraCopy[authorizationv1.ScopesKey] = append(userExtraCopy[authorizationv1.ScopesKey], authorizationscope.UserFull)

	userWithFullScope := &user.DefaultInfo{
		Name:   userToCheck.GetName(),
		UID:    userToCheck.GetUID(),
		Groups: userToCheck.GetGroups(),
		Extra:  userExtraCopy,
	}

	return userWithFullScope
}

// a self-SAR request must be authorized as if it has either the full user's permissions
// or the permissions of the user's role set on the request (if applicable) in order
// to be able to perform the access review
func scopesNeedUserFull(scopes []string) bool {
	if len(scopes) == 0 {
		return false
	}

	sort.Strings(scopes)
	switch {
	case
		// all scope slices used here must be sorted
		reflect.DeepEqual(scopes, []string{authorizationscope.UserAccessCheck}),
		reflect.DeepEqual(scopes, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo}),
		reflect.DeepEqual(scopes, []string{authorizationscope.UserAccessCheck, authorizationscope.UserListAllProjects}),
		reflect.DeepEqual(scopes, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, authorizationscope.UserListAllProjects}):
		return true
	}

	return false
}
