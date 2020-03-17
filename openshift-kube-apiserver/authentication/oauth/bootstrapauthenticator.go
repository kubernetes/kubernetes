package oauth

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	kauthenticator "k8s.io/apiserver/pkg/authentication/authenticator"
	kuser "k8s.io/apiserver/pkg/authentication/user"

	authorizationv1 "github.com/openshift/api/authorization/v1"
	userv1 "github.com/openshift/api/user/v1"
	oauthclient "github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1"
	bootstrap "github.com/openshift/library-go/pkg/authentication/bootstrapauthenticator"
)

const ClusterAdminGroup = "system:cluster-admins"

type bootstrapAuthenticator struct {
	tokens            oauthclient.OAuthAccessTokenInterface
	getter            bootstrap.BootstrapUserDataGetter
	validator         OAuthTokenValidator
	implicitAudiences kauthenticator.Audiences
}

func NewBootstrapAuthenticator(tokens oauthclient.OAuthAccessTokenInterface, getter bootstrap.BootstrapUserDataGetter, implicitAudiences kauthenticator.Audiences, validators ...OAuthTokenValidator) kauthenticator.Token {
	return &bootstrapAuthenticator{
		tokens:            tokens,
		getter:            getter,
		validator:         OAuthTokenValidators(validators),
		implicitAudiences: implicitAudiences,
	}
}

func (a *bootstrapAuthenticator) AuthenticateToken(ctx context.Context, name string) (*kauthenticator.Response, bool, error) {
	token, err := a.tokens.Get(name, metav1.GetOptions{})
	if err != nil {
		return nil, false, errLookup // mask the error so we do not leak token data in logs
	}

	if token.UserName != bootstrap.BootstrapUser {
		return nil, false, nil
	}

	data, ok, err := a.getter.Get()
	if err != nil || !ok {
		return nil, ok, err
	}

	// this allows us to reuse existing validators
	// since the uid is based on the secret, if the secret changes, all
	// tokens issued for the bootstrap user before that change stop working
	fakeUser := &userv1.User{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(data.UID),
		},
	}

	if err := a.validator.Validate(token, fakeUser); err != nil {
		return nil, false, err
	}

	tokenAudiences := a.implicitAudiences
	requestedAudiences, ok := kauthenticator.AudiencesFrom(ctx)
	if !ok {
		// default to apiserver audiences
		requestedAudiences = a.implicitAudiences
	}

	auds := kauthenticator.Audiences(tokenAudiences).Intersect(requestedAudiences)
	if len(auds) == 0 && len(a.implicitAudiences) != 0 {
		return nil, false, fmt.Errorf("token audiences %q is invalid for the target audiences %q", tokenAudiences, requestedAudiences)
	}

	// we explicitly do not set UID as we do not want to leak any derivative of the password
	return &kauthenticator.Response{
		Audiences: auds,
		User: &kuser.DefaultInfo{
			Name: bootstrap.BootstrapUser,
			// we cannot use SystemPrivilegedGroup because it cannot be properly scoped.
			// see openshift/origin#18922 and how loopback connections are handled upstream via AuthorizeClientBearerToken.
			// api aggregation with delegated authorization makes this impossible to control, see WithAlwaysAllowGroups.
			// an openshift specific cluster role binding binds ClusterAdminGroup to the cluster role cluster-admin.
			// thus this group is authorized to do everything via RBAC.
			// this does make the bootstrap user susceptible to anything that causes the RBAC authorizer to fail.
			// this is a safe trade-off because scopes must always be evaluated before RBAC for them to work at all.
			// a failure in that logic means scopes are broken instead of a specific failure related to the bootstrap user.
			// if this becomes a problem in the future, we could generate a custom extra value based on the secret content
			// and store it in BootstrapUserData, similar to how UID is calculated.  this extra value would then be wired
			// to a custom authorizer that allows all actions.  the problem with such an approach is that since we do not
			// allow remote authorizers in OpenShift, the BootstrapUserDataGetter logic would have to be shared between the
			// the kube api server and osin instead of being an implementation detail hidden inside of osin.  currently the
			// only shared code is the value of the BootstrapUser constant (since it is special cased in validation).
			Groups: []string{ClusterAdminGroup},
			Extra: map[string][]string{
				// this user still needs scopes because it can be used in OAuth flows (unlike cert based users)
				authorizationv1.ScopesKey: token.Scopes,
			},
		},
	}, true, nil
}
