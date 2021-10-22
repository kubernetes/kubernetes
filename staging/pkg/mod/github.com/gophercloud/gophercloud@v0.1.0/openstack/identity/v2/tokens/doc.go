/*
Package tokens provides information and interaction with the token API
resource for the OpenStack Identity service.

For more information, see:
http://developer.openstack.org/api-ref-identity-v2.html#identity-auth-v2

Example to Create an Unscoped Token from a Password

	authOpts := gophercloud.AuthOptions{
		Username: "user",
		Password: "pass"
	}

	token, err := tokens.Create(identityClient, authOpts).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token from a Tenant ID and Password

	authOpts := gophercloud.AuthOptions{
		Username: "user",
		Password: "password",
		TenantID: "fc394f2ab2df4114bde39905f800dc57"
	}

	token, err := tokens.Create(identityClient, authOpts).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token from a Tenant Name and Password

	authOpts := gophercloud.AuthOptions{
		Username:   "user",
		Password:   "password",
		TenantName: "tenantname"
	}

	token, err := tokens.Create(identityClient, authOpts).ExtractToken()
	if err != nil {
		panic(err)
	}
*/
package tokens
