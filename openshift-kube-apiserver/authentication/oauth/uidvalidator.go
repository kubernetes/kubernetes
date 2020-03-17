package oauth

import (
	"fmt"

	oauthv1 "github.com/openshift/api/oauth/v1"
	userv1 "github.com/openshift/api/user/v1"
)

const errInvalidUIDStr = "user.UID (%s) does not match token.userUID (%s)"

func NewUIDValidator() OAuthTokenValidator {
	return OAuthTokenValidatorFunc(
		func(token *oauthv1.OAuthAccessToken, user *userv1.User) error {
			if string(user.UID) != token.UserUID {
				return fmt.Errorf(errInvalidUIDStr, user.UID, token.UserUID)
			}
			return nil
		},
	)
}
