// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/domains"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/roles"
)

func TestRolesList(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	listOpts := roles.ListOpts{
		DomainID: "default",
	}

	allPages, err := roles.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list roles: %v", err)
	}

	allRoles, err := roles.ExtractRoles(allPages)
	if err != nil {
		t.Fatalf("Unable to extract roles: %v", err)
	}

	for _, role := range allRoles {
		tools.PrintResource(t, role)
	}
}

func TestRolesGet(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	role, err := FindRole(t, client)
	if err != nil {
		t.Fatalf("Unable to find a role: %v", err)
	}

	p, err := roles.Get(client, role.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to get role: %v", err)
	}

	tools.PrintResource(t, p)
}

func TestRoleCRUD(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	createOpts := roles.CreateOpts{
		Name:     "testrole",
		DomainID: "default",
		Extra: map[string]interface{}{
			"description": "test role description",
		},
	}

	// Create Role in the default domain
	role, err := CreateRole(t, client, &createOpts)
	if err != nil {
		t.Fatalf("Unable to create role: %v", err)
	}
	defer DeleteRole(t, client, role.ID)

	tools.PrintResource(t, role)
	tools.PrintResource(t, role.Extra)

	updateOpts := roles.UpdateOpts{
		Extra: map[string]interface{}{
			"description": "updated test role description",
		},
	}

	newRole, err := roles.Update(client, role.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update role: %v", err)
	}

	tools.PrintResource(t, newRole)
	tools.PrintResource(t, newRole.Extra)
}

func TestRoleAssignToUserOnProject(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an indentity client: %v", err)
	}

	project, err := CreateProject(t, client, nil)
	if err != nil {
		t.Fatal("Unable to create a project")
	}
	defer DeleteProject(t, client, project.ID)

	role, err := FindRole(t, client)
	if err != nil {
		t.Fatalf("Unable to get a role: %v", err)
	}

	user, err := CreateUser(t, client, nil)
	if err != nil {
		t.Fatalf("Unable to create user: %v", err)
	}
	defer DeleteUser(t, client, user.ID)

	t.Logf("Attempting to assign a role %s to a user %s on a project %s", role.Name, user.Name, project.Name)
	err = roles.Assign(client, role.ID, roles.AssignOpts{
		UserID:    user.ID,
		ProjectID: project.ID,
	}).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to assign a role to a user on a project: %v", err)
	}
	t.Logf("Successfully assigned a role %s to a user %s on a project %s", role.Name, user.Name, project.Name)
	defer UnassignRole(t, client, role.ID, &roles.UnassignOpts{
		UserID:    user.ID,
		ProjectID: project.ID,
	})

	allPages, err := roles.ListAssignments(client, roles.ListAssignmentsOpts{
		RoleID:         role.ID,
		ScopeProjectID: project.ID,
		UserID:         user.ID,
	}).AllPages()
	if err != nil {
		t.Fatalf("Unable to list role assignments: %v", err)
	}

	allRoleAssignments, err := roles.ExtractRoleAssignments(allPages)
	if err != nil {
		t.Fatalf("Unable to extract role assignments: %v", err)
	}

	t.Logf("Role assignments of user %s on project %s:", user.Name, project.Name)
	for _, roleAssignment := range allRoleAssignments {
		tools.PrintResource(t, roleAssignment)
	}
}

func TestRoleAssignToUserOnDomain(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an indentity client: %v", err)
	}

	domain, err := CreateDomain(t, client, &domains.CreateOpts{
		Enabled: gophercloud.Disabled,
	})
	if err != nil {
		t.Fatal("Unable to create a domain")
	}
	defer DeleteDomain(t, client, domain.ID)

	role, err := FindRole(t, client)
	if err != nil {
		t.Fatalf("Unable to get a role: %v", err)
	}

	user, err := CreateUser(t, client, nil)
	if err != nil {
		t.Fatalf("Unable to create user: %v", err)
	}
	defer DeleteUser(t, client, user.ID)

	t.Logf("Attempting to assign a role %s to a user %s on a domain %s", role.Name, user.Name, domain.Name)
	err = roles.Assign(client, role.ID, roles.AssignOpts{
		UserID:   user.ID,
		DomainID: domain.ID,
	}).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to assign a role to a user on a domain: %v", err)
	}
	t.Logf("Successfully assigned a role %s to a user %s on a domain %s", role.Name, user.Name, domain.Name)
	defer UnassignRole(t, client, role.ID, &roles.UnassignOpts{
		UserID:   user.ID,
		DomainID: domain.ID,
	})

	allPages, err := roles.ListAssignments(client, roles.ListAssignmentsOpts{
		RoleID:        role.ID,
		ScopeDomainID: domain.ID,
		UserID:        user.ID,
	}).AllPages()
	if err != nil {
		t.Fatalf("Unable to list role assignments: %v", err)
	}

	allRoleAssignments, err := roles.ExtractRoleAssignments(allPages)
	if err != nil {
		t.Fatalf("Unable to extract role assignments: %v", err)
	}

	t.Logf("Role assignments of user %s on domain %s:", user.Name, domain.Name)
	for _, roleAssignment := range allRoleAssignments {
		tools.PrintResource(t, roleAssignment)
	}
}

func TestRoleAssignToGroupOnDomain(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an indentity client: %v", err)
	}

	domain, err := CreateDomain(t, client, &domains.CreateOpts{
		Enabled: gophercloud.Disabled,
	})
	if err != nil {
		t.Fatal("Unable to create a domain")
	}
	defer DeleteDomain(t, client, domain.ID)

	role, err := FindRole(t, client)
	if err != nil {
		t.Fatalf("Unable to get a role: %v", err)
	}

	group, err := CreateGroup(t, client, nil)
	if err != nil {
		t.Fatalf("Unable to create group: %v", err)
	}
	defer DeleteGroup(t, client, group.ID)

	t.Logf("Attempting to assign a role %s to a group %s on a domain %s", role.Name, group.Name, domain.Name)
	err = roles.Assign(client, role.ID, roles.AssignOpts{
		GroupID:  group.ID,
		DomainID: domain.ID,
	}).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to assign a role to a group on a domain: %v", err)
	}
	t.Logf("Successfully assigned a role %s to a group %s on a domain %s", role.Name, group.Name, domain.Name)
	defer UnassignRole(t, client, role.ID, &roles.UnassignOpts{
		GroupID:  group.ID,
		DomainID: domain.ID,
	})

	allPages, err := roles.ListAssignments(client, roles.ListAssignmentsOpts{
		RoleID:        role.ID,
		ScopeDomainID: domain.ID,
		GroupID:       group.ID,
	}).AllPages()
	if err != nil {
		t.Fatalf("Unable to list role assignments: %v", err)
	}

	allRoleAssignments, err := roles.ExtractRoleAssignments(allPages)
	if err != nil {
		t.Fatalf("Unable to extract role assignments: %v", err)
	}

	t.Logf("Role assignments of group %s on domain %s:", group.Name, domain.Name)
	for _, roleAssignment := range allRoleAssignments {
		tools.PrintResource(t, roleAssignment)
	}
}

func TestRoleAssignToGroupOnProject(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an indentity client: %v", err)
	}

	project, err := CreateProject(t, client, nil)
	if err != nil {
		t.Fatal("Unable to create a project")
	}
	defer DeleteProject(t, client, project.ID)

	role, err := FindRole(t, client)
	if err != nil {
		t.Fatalf("Unable to get a role: %v", err)
	}

	group, err := CreateGroup(t, client, nil)
	if err != nil {
		t.Fatalf("Unable to create group: %v", err)
	}
	defer DeleteGroup(t, client, group.ID)

	t.Logf("Attempting to assign a role %s to a group %s on a project %s", role.Name, group.Name, project.Name)
	err = roles.Assign(client, role.ID, roles.AssignOpts{
		GroupID:   group.ID,
		ProjectID: project.ID,
	}).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to assign a role to a group on a project: %v", err)
	}
	t.Logf("Successfully assigned a role %s to a group %s on a project %s", role.Name, group.Name, project.Name)
	defer UnassignRole(t, client, role.ID, &roles.UnassignOpts{
		GroupID:   group.ID,
		ProjectID: project.ID,
	})

	allPages, err := roles.ListAssignments(client, roles.ListAssignmentsOpts{
		RoleID:         role.ID,
		ScopeProjectID: project.ID,
		GroupID:        group.ID,
	}).AllPages()
	if err != nil {
		t.Fatalf("Unable to list role assignments: %v", err)
	}

	allRoleAssignments, err := roles.ExtractRoleAssignments(allPages)
	if err != nil {
		t.Fatalf("Unable to extract role assignments: %v", err)
	}

	t.Logf("Role assignments of group %s on project %s:", group.Name, project.Name)
	for _, roleAssignment := range allRoleAssignments {
		tools.PrintResource(t, roleAssignment)
	}
}
