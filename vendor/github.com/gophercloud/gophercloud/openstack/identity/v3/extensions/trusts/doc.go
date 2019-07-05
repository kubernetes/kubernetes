/*
Package trusts enables management of OpenStack Identity Trusts.

Example to Create a Token with Username, Password, and Trust ID

	var trustToken struct {
		tokens.Token
		trusts.TokenExt
	}

	authOptions := tokens.AuthOptions{
		UserID:   "username",
		Password: "password",
	}

	createOpts := trusts.AuthOptsExt{
		AuthOptionsBuilder: authOptions,
		TrustID:            "de0945a",
	}

	err := tokens.Create(identityClient, createOpts).ExtractInto(&trustToken)
	if err != nil {
		panic(err)
	}
*/
package trusts
