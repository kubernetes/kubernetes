/*
Package tokens provides information and interaction with the token API
resource for the OpenStack Identity service.

For more information, see:
http://developer.openstack.org/api-ref-identity-v3.html#tokens-v3

Example to Create a Token From a Username and Password

	authOptions := tokens.AuthOptions{
		UserID:   "username",
		Password: "password",
	}

	token, err := tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token From a Username, Password, and Domain

	authOptions := tokens.AuthOptions{
		UserID:   "username",
		Password: "password",
		DomainID: "default",
	}

	token, err := tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

	authOptions = tokens.AuthOptions{
		UserID:     "username",
		Password:   "password",
		DomainName: "default",
	}

	token, err = tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token From a Token

	authOptions := tokens.AuthOptions{
		TokenID: "token_id",
	}

	token, err := tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token from a Username and Password with Project ID Scope

	scope := tokens.Scope{
		ProjectID: "0fe36e73809d46aeae6705c39077b1b3",
	}

	authOptions := tokens.AuthOptions{
		Scope:    &scope,
		UserID:   "username",
		Password: "password",
	}

	token, err = tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token from a Username and Password with Domain ID Scope

	scope := tokens.Scope{
		DomainID: "default",
	}

	authOptions := tokens.AuthOptions{
		Scope:    &scope,
		UserID:   "username",
		Password: "password",
	}

	token, err = tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

Example to Create a Token from a Username and Password with Project Name Scope

	scope := tokens.Scope{
		ProjectName: "project_name",
		DomainID:    "default",
	}

	authOptions := tokens.AuthOptions{
		Scope:    &scope,
		UserID:   "username",
		Password: "password",
	}

	token, err = tokens.Create(identityClient, authOptions).ExtractToken()
	if err != nil {
		panic(err)
	}

*/
package tokens
