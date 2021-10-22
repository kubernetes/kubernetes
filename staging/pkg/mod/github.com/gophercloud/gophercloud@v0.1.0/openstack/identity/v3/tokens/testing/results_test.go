package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/testhelper"
)

func TestExtractToken(t *testing.T) {
	result := getGetResult(t)

	token, err := result.ExtractToken()
	testhelper.AssertNoErr(t, err)

	testhelper.CheckDeepEquals(t, &ExpectedToken, token)
}

func TestExtractCatalog(t *testing.T) {
	result := getGetResult(t)

	catalog, err := result.ExtractServiceCatalog()
	testhelper.AssertNoErr(t, err)

	testhelper.CheckDeepEquals(t, &ExpectedServiceCatalog, catalog)
}

func TestExtractUser(t *testing.T) {
	result := getGetResult(t)

	user, err := result.ExtractUser()
	testhelper.AssertNoErr(t, err)

	testhelper.CheckDeepEquals(t, &ExpectedUser, user)
}

func TestExtractRoles(t *testing.T) {
	result := getGetResult(t)

	roles, err := result.ExtractRoles()
	testhelper.AssertNoErr(t, err)

	testhelper.CheckDeepEquals(t, ExpectedRoles, roles)
}

func TestExtractProject(t *testing.T) {
	result := getGetResult(t)

	project, err := result.ExtractProject()
	testhelper.AssertNoErr(t, err)

	testhelper.CheckDeepEquals(t, &ExpectedProject, project)
}
