package oauth

import (
	"context"
	"errors"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kauthenticator "k8s.io/apiserver/pkg/authentication/authenticator"
	kuser "k8s.io/apiserver/pkg/authentication/user"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	oauthclient "github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1"
	userclient "github.com/openshift/client-go/user/clientset/versioned/typed/user/v1"
)

var errLookup = errors.New("token lookup failed")

type tokenAuthenticator struct {
	tokens       oauthclient.OAuthAccessTokenInterface
	users        userclient.UserInterface
	groupMapper  UserToGroupMapper
	validators   OAuthTokenValidator
	implicitAuds kauthenticator.Audiences
}

func NewTokenAuthenticator(tokens oauthclient.OAuthAccessTokenInterface, users userclient.UserInterface, groupMapper UserToGroupMapper, implicitAuds kauthenticator.Audiences, validators ...OAuthTokenValidator) kauthenticator.Token {
	return &tokenAuthenticator{
		tokens:       tokens,
		users:        users,
		groupMapper:  groupMapper,
		validators:   OAuthTokenValidators(validators),
		implicitAuds: implicitAuds,
	}
}

func (a *tokenAuthenticator) AuthenticateToken(ctx context.Context, name string) (*kauthenticator.Response, bool, error) {
	token, err := a.tokens.Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, false, errLookup // mask the error so we do not leak token data in logs
	}

	user, err := a.users.Get(token.UserName, metav1.GetOptions{})
	if err != nil {
		return nil, false, err
	}

	if err := a.validators.Validate(token, user); err != nil {
		return nil, false, err
	}

	groups, err := a.groupMapper.GroupsFor(user.Name)
	if err != nil {
		return nil, false, err
	}
	groupNames := make([]string, 0, len(groups))
	for _, group := range groups {
		groupNames = append(groupNames, group.Name)
	}

	tokenAudiences := a.implicitAuds
	requestedAudiences, ok := kauthenticator.AudiencesFrom(ctx)
	if !ok {
		// default to apiserver audiences
		requestedAudiences = a.implicitAuds
	}

	auds := kauthenticator.Audiences(tokenAudiences).Intersect(requestedAudiences)
	if len(auds) == 0 && len(a.implicitAuds) != 0 {
		return nil, false, fmt.Errorf("token audiences %q is invalid for the target audiences %q", tokenAudiences, requestedAudiences)
	}

	return &kauthenticator.Response{
		User: &kuser.DefaultInfo{
			Name:   user.Name,
			UID:    string(user.UID),
			Groups: groupNames,
			Extra: map[string][]string{
				authorizationv1.ScopesKey: token.Scopes,
			},
		},
		Audiences: auds,
	}, true, nil
}
