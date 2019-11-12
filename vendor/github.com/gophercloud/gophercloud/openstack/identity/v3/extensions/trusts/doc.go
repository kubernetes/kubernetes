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

Example to Create a Trust

    expiresAt := time.Date(2019, 12, 1, 14, 0, 0, 999999999, time.UTC)
    createOpts := trusts.CreateOpts{
        ExpiresAt:         &expiresAt,
        Impersonation:     true,
        AllowRedelegation: true,
        ProjectID:         "9b71012f5a4a4aef9193f1995fe159b2",
        Roles: []trusts.Role{
            {
                Name: "member",
            },
        },
        TrusteeUserID: "ecb37e88cc86431c99d0332208cb6fbf",
        TrustorUserID: "959ed913a32c4ec88c041c98e61cbbc3",
	}

    trust, err := trusts.Create(identityClient, createOpts).Extract()
    if err != nil {
        panic(err)
    }

    fmt.Printf("Trust: %+v\n", trust)

Example to Delete a Trust

    trustID := "3422b7c113894f5d90665e1a79655e23"
    err := trusts.Delete(identityClient, trustID).ExtractErr()
    if err != nil {
        panic(err)
    }
*/
package trusts
