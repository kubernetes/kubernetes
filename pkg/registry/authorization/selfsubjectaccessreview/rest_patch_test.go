package selfsubjectaccessreview

import (
	"testing"

	authorizationscope "github.com/openshift/apiserver-library-go/pkg/authorization/scope"
)

func TestScopesNeedUserFull(t *testing.T) {
	roleScope := "role:testrole:testns"
	tests := []struct {
		want   bool
		scopes []string
	}{
		{true, []string{authorizationscope.UserAccessCheck}},
		{true, []string{authorizationscope.UserInfo, authorizationscope.UserAccessCheck}},
		{true, []string{authorizationscope.UserListAllProjects, authorizationscope.UserAccessCheck}},
		{true, []string{authorizationscope.UserListAllProjects, authorizationscope.UserInfo, authorizationscope.UserAccessCheck}},
		{false, nil},
		{false, []string{}},
		{false, []string{authorizationscope.UserInfo}},
		{false, []string{authorizationscope.UserListAllProjects}},
		{false, []string{authorizationscope.UserFull}},
		{false, []string{roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserAccessCheck, roleScope}},
		{false, []string{authorizationscope.UserInfo, authorizationscope.UserListAllProjects}},
		{false, []string{authorizationscope.UserInfo, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserInfo, roleScope}},
		{false, []string{authorizationscope.UserListAllProjects, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserListAllProjects, roleScope}},
		{false, []string{authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserListAllProjects, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserListAllProjects, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserInfo, authorizationscope.UserListAllProjects, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserInfo, authorizationscope.UserListAllProjects, roleScope}},
		{false, []string{authorizationscope.UserInfo, authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserListAllProjects, authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, authorizationscope.UserListAllProjects, authorizationscope.UserFull}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, authorizationscope.UserListAllProjects, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserListAllProjects, authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserInfo, authorizationscope.UserListAllProjects, authorizationscope.UserFull, roleScope}},
		{false, []string{authorizationscope.UserAccessCheck, authorizationscope.UserInfo, authorizationscope.UserListAllProjects, authorizationscope.UserFull, roleScope}},
	}

	for _, tt := range tests {
		if got := scopesNeedUserFull(tt.scopes); got != tt.want {
			t.Errorf("scopes %v; got %v; want %v", tt.scopes, got, tt.want)
		}
	}
}
