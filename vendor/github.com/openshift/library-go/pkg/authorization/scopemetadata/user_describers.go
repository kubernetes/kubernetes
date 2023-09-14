package scopemetadata

import (
	"fmt"
)

// these must agree with the scope authorizer, but it's an API we cannot realistically change
const (
	scopesAllNamespaces = "*"

	userIndicator        = "user:"
	clusterRoleIndicator = "role:"

	UserInfo        = userIndicator + "info"
	UserAccessCheck = userIndicator + "check-access"

	// UserListScopedProjects gives explicit permission to see the projects that this token can see.
	UserListScopedProjects = userIndicator + "list-scoped-projects"

	// UserListAllProjects gives explicit permission to see the projects a user can see.  This is often used to prime secondary ACL systems
	// unrelated to openshift and to display projects for selection in a secondary UI.
	UserListAllProjects = userIndicator + "list-projects"

	// UserFull includes all permissions of the user
	userFull = userIndicator + "full"
)

// user:<scope name>
type UserEvaluator struct{}

func (UserEvaluator) Handles(scope string) bool {
	return UserEvaluatorHandles(scope)
}

func (e UserEvaluator) Validate(scope string) error {
	if e.Handles(scope) {
		return nil
	}

	return fmt.Errorf("unrecognized scope: %v", scope)
}

var defaultSupportedScopesMap = map[string]string{
	UserInfo:               "Read-only access to your user information (including username, identities, and group membership)",
	UserAccessCheck:        `Read-only access to view your privileges (for example, "can I create builds?")`,
	UserListScopedProjects: `Read-only access to list your projects viewable with this token and view their metadata (display name, description, etc.)`,
	UserListAllProjects:    `Read-only access to list your projects and view their metadata (display name, description, etc.)`,
	userFull:               `Full read/write access with all of your permissions`,
}

func (UserEvaluator) Describe(scope string) (string, string, error) {
	switch scope {
	case UserInfo, UserAccessCheck, UserListScopedProjects, UserListAllProjects:
		return defaultSupportedScopesMap[scope], "", nil
	case userFull:
		return defaultSupportedScopesMap[scope], `Includes any access you have to escalating resources like secrets`, nil
	default:
		return "", "", fmt.Errorf("unrecognized scope: %v", scope)
	}
}

func UserEvaluatorHandles(scope string) bool {
	switch scope {
	case userFull, UserInfo, UserAccessCheck, UserListScopedProjects, UserListAllProjects:
		return true
	}
	return false
}
