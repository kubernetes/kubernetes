package roles

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListAssignmentsOptsBuilder allows extensions to add additional parameters to
// the ListAssignments request.
type ListAssignmentsOptsBuilder interface {
	ToRolesListAssignmentsQuery() (string, error)
}

// ListAssignmentsOpts allows you to query the ListAssignments method.
// Specify one of or a combination of GroupId, RoleId, ScopeDomainId, ScopeProjectId,
// and/or UserId to search for roles assigned to corresponding entities.
// Effective lists effective assignments at the user, project, and domain level,
// allowing for the effects of group membership.
type ListAssignmentsOpts struct {
	GroupId        string `q:"group.id"`
	RoleId         string `q:"role.id"`
	ScopeDomainId  string `q:"scope.domain.id"`
	ScopeProjectId string `q:"scope.project.id"`
	UserId         string `q:"user.id"`
	Effective      bool   `q:"effective"`
}

// ToRolesListAssignmentsQuery formats a ListAssignmentsOpts into a query string.
func (opts ListAssignmentsOpts) ToRolesListAssignmentsQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// ListAssignments enumerates the roles assigned to a specified resource.
func ListAssignments(client *gophercloud.ServiceClient, opts ListAssignmentsOptsBuilder) pagination.Pager {
	url := listAssignmentsURL(client)
	query, err := opts.ToRolesListAssignmentsQuery()
	if err != nil {
		return pagination.Pager{Err: err}
	}
	url += query
	createPage := func(r pagination.PageResult) pagination.Page {
		return RoleAssignmentsPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, url, createPage)
}
