package oauth

import (
	"errors"
	"time"

	oauthv1 "github.com/openshift/api/oauth/v1"
	userv1 "github.com/openshift/api/user/v1"
)

var errExpired = errors.New("token is expired")

func NewExpirationValidator() OAuthTokenValidator {
	return OAuthTokenValidatorFunc(
		func(token *oauthv1.OAuthAccessToken, _ *userv1.User) error {
			if token.ExpiresIn > 0 {
				if expire(token).Before(time.Now()) {
					return errExpired
				}
			}
			if token.DeletionTimestamp != nil {
				return errExpired
			}
			return nil
		},
	)
}

func expire(token *oauthv1.OAuthAccessToken) time.Time {
	return token.CreationTimestamp.Add(time.Duration(token.ExpiresIn) * time.Second)
}
