package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/domains"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/groups"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/projects"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/regions"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/roles"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/services"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/users"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateProject will create a project with a random name.
// It takes an optional createOpts parameter since creating a project
// has so many options. An error will be returned if the project was
// unable to be created.
func CreateProject(t *testing.T, client *gophercloud.ServiceClient, c *projects.CreateOpts) (*projects.Project, error) {
	name := tools.RandomString("ACPTTEST", 8)
	description := tools.RandomString("ACPTTEST-DESC", 8)
	t.Logf("Attempting to create project: %s", name)

	var createOpts projects.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = projects.CreateOpts{}
	}

	createOpts.Name = name
	createOpts.Description = description

	project, err := projects.Create(client, createOpts).Extract()
	if err != nil {
		return project, err
	}

	t.Logf("Successfully created project %s with ID %s", name, project.ID)

	th.AssertEquals(t, project.Name, name)
	th.AssertEquals(t, project.Description, description)

	return project, nil
}

// CreateUser will create a user with a random name.
// It takes an optional createOpts parameter since creating a user
// has so many options. An error will be returned if the user was
// unable to be created.
func CreateUser(t *testing.T, client *gophercloud.ServiceClient, c *users.CreateOpts) (*users.User, error) {
	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create user: %s", name)

	var createOpts users.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = users.CreateOpts{}
	}

	createOpts.Name = name

	user, err := users.Create(client, createOpts).Extract()
	if err != nil {
		return user, err
	}

	t.Logf("Successfully created user %s with ID %s", name, user.ID)

	th.AssertEquals(t, user.Name, name)

	return user, nil
}

// CreateGroup will create a group with a random name.
// It takes an optional createOpts parameter since creating a group
// has so many options. An error will be returned if the group was
// unable to be created.
func CreateGroup(t *testing.T, client *gophercloud.ServiceClient, c *groups.CreateOpts) (*groups.Group, error) {
	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create group: %s", name)

	var createOpts groups.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = groups.CreateOpts{}
	}

	createOpts.Name = name

	group, err := groups.Create(client, createOpts).Extract()
	if err != nil {
		return group, err
	}

	t.Logf("Successfully created group %s with ID %s", name, group.ID)

	th.AssertEquals(t, group.Name, name)

	return group, nil
}

// CreateDomain will create a domain with a random name.
// It takes an optional createOpts parameter since creating a domain
// has many options. An error will be returned if the domain was
// unable to be created.
func CreateDomain(t *testing.T, client *gophercloud.ServiceClient, c *domains.CreateOpts) (*domains.Domain, error) {
	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create domain: %s", name)

	var createOpts domains.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = domains.CreateOpts{}
	}

	createOpts.Name = name

	domain, err := domains.Create(client, createOpts).Extract()
	if err != nil {
		return domain, err
	}

	t.Logf("Successfully created domain %s with ID %s", name, domain.ID)

	th.AssertEquals(t, domain.Name, name)

	return domain, nil
}

// CreateRole will create a role with a random name.
// It takes an optional createOpts parameter since creating a role
// has so many options. An error will be returned if the role was
// unable to be created.
func CreateRole(t *testing.T, client *gophercloud.ServiceClient, c *roles.CreateOpts) (*roles.Role, error) {
	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create role: %s", name)

	var createOpts roles.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = roles.CreateOpts{}
	}

	createOpts.Name = name

	role, err := roles.Create(client, createOpts).Extract()
	if err != nil {
		return role, err
	}

	t.Logf("Successfully created role %s with ID %s", name, role.ID)

	th.AssertEquals(t, role.Name, name)

	return role, nil
}

// CreateRegion will create a region with a random name.
// It takes an optional createOpts parameter since creating a region
// has so many options. An error will be returned if the region was
// unable to be created.
func CreateRegion(t *testing.T, client *gophercloud.ServiceClient, c *regions.CreateOpts) (*regions.Region, error) {
	id := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create region: %s", id)

	var createOpts regions.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = regions.CreateOpts{}
	}

	createOpts.ID = id

	region, err := regions.Create(client, createOpts).Extract()
	if err != nil {
		return region, err
	}

	t.Logf("Successfully created region %s", id)

	th.AssertEquals(t, region.ID, id)

	return region, nil
}

// CreateService will create a service with a random name.
// It takes an optional createOpts parameter since creating a service
// has so many options. An error will be returned if the service was
// unable to be created.
func CreateService(t *testing.T, client *gophercloud.ServiceClient, c *services.CreateOpts) (*services.Service, error) {
	name := tools.RandomString("ACPTTEST", 8)
	t.Logf("Attempting to create service: %s", name)

	var createOpts services.CreateOpts
	if c != nil {
		createOpts = *c
	} else {
		createOpts = services.CreateOpts{}
	}

	createOpts.Extra["name"] = name

	service, err := services.Create(client, createOpts).Extract()
	if err != nil {
		return service, err
	}

	t.Logf("Successfully created service %s", service.ID)

	th.AssertEquals(t, service.Extra["name"], name)

	return service, nil
}

// DeleteProject will delete a project by ID. A fatal error will occur if
// the project ID failed to be deleted. This works best when using it as
// a deferred function.
func DeleteProject(t *testing.T, client *gophercloud.ServiceClient, projectID string) {
	err := projects.Delete(client, projectID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete project %s: %v", projectID, err)
	}

	t.Logf("Deleted project: %s", projectID)
}

// DeleteUser will delete a user by ID. A fatal error will occur if
// the user failed to be deleted. This works best when using it as
// a deferred function.
func DeleteUser(t *testing.T, client *gophercloud.ServiceClient, userID string) {
	err := users.Delete(client, userID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete user with ID %s: %v", userID, err)
	}

	t.Logf("Deleted user with ID: %s", userID)
}

// DeleteGroup will delete a group by ID. A fatal error will occur if
// the group failed to be deleted. This works best when using it as
// a deferred function.
func DeleteGroup(t *testing.T, client *gophercloud.ServiceClient, groupID string) {
	err := groups.Delete(client, groupID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete group %s: %v", groupID, err)
	}

	t.Logf("Deleted group: %s", groupID)
}

// DeleteDomain will delete a domain by ID. A fatal error will occur if
// the project ID failed to be deleted. This works best when using it as
// a deferred function.
func DeleteDomain(t *testing.T, client *gophercloud.ServiceClient, domainID string) {
	err := domains.Delete(client, domainID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete domain %s: %v", domainID, err)
	}

	t.Logf("Deleted domain: %s", domainID)
}

// DeleteRole will delete a role by ID. A fatal error will occur if
// the role failed to be deleted. This works best when using it as
// a deferred function.
func DeleteRole(t *testing.T, client *gophercloud.ServiceClient, roleID string) {
	err := roles.Delete(client, roleID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete role %s: %v", roleID, err)
	}

	t.Logf("Deleted role: %s", roleID)
}

// DeleteRegion will delete a reg by ID. A fatal error will occur if
// the region failed to be deleted. This works best when using it as
// a deferred function.
func DeleteRegion(t *testing.T, client *gophercloud.ServiceClient, regionID string) {
	err := regions.Delete(client, regionID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete region %s: %v", regionID, err)
	}

	t.Logf("Deleted region: %s", regionID)
}

// DeleteService will delete a reg by ID. A fatal error will occur if
// the service failed to be deleted. This works best when using it as
// a deferred function.
func DeleteService(t *testing.T, client *gophercloud.ServiceClient, serviceID string) {
	err := services.Delete(client, serviceID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete service %s: %v", serviceID, err)
	}

	t.Logf("Deleted service: %s", serviceID)
}

// UnassignRole will delete a role assigned to a user/group on a project/domain
// A fatal error will occur if it fails to delete the assignment.
// This works best when using it as a deferred function.
func UnassignRole(t *testing.T, client *gophercloud.ServiceClient, roleID string, opts *roles.UnassignOpts) {
	err := roles.Unassign(client, roleID, *opts).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to unassign a role %v on context %+v: %v", roleID, *opts, err)
	}
	t.Logf("Unassigned the role %v on context %+v", roleID, *opts)
}

// FindRole finds all roles that the current authenticated client has access
// to and returns the first one found. An error will be returned if the lookup
// was unsuccessful.
func FindRole(t *testing.T, client *gophercloud.ServiceClient) (*roles.Role, error) {
	t.Log("Attempting to find a role")
	var role *roles.Role

	allPages, err := roles.List(client, nil).AllPages()
	if err != nil {
		return nil, err
	}

	allRoles, err := roles.ExtractRoles(allPages)
	if err != nil {
		return nil, err
	}

	for _, r := range allRoles {
		role = &r
		break
	}

	t.Logf("Successfully found a role %s with ID %s", role.Name, role.ID)

	return role, nil
}
