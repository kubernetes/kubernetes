package oauth

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	mathrand "math/rand"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	oauthv1 "github.com/openshift/api/oauth/v1"
	userv1 "github.com/openshift/api/user/v1"
	oauthfake "github.com/openshift/client-go/oauth/clientset/versioned/fake"
	userfake "github.com/openshift/client-go/user/clientset/versioned/fake"
)

func TestAuthenticateTokenExpired(t *testing.T) {
	token1, token1Hash := generateOAuthTokenPair()
	token2, token2Hash := generateOAuthTokenPair()
	fakeOAuthClient := oauthfake.NewSimpleClientset(
		// expired token that had a lifetime of 10 minutes
		&oauthv1.OAuthAccessToken{
			ObjectMeta: metav1.ObjectMeta{Name: token1Hash, CreationTimestamp: metav1.Time{Time: time.Now().Add(-1 * time.Hour)}},
			ExpiresIn:  600,
			UserName:   "foo",
		},
		// non-expired token that has a lifetime of 10 minutes, but has a non-nil deletion timestamp
		&oauthv1.OAuthAccessToken{
			ObjectMeta: metav1.ObjectMeta{Name: token2Hash, CreationTimestamp: metav1.Time{Time: time.Now()}, DeletionTimestamp: &metav1.Time{}},
			ExpiresIn:  600,
			UserName:   "foo",
		},
	)
	fakeUserClient := userfake.NewSimpleClientset(&userv1.User{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "bar"}})

	tokenAuthenticator := NewTokenAuthenticator(fakeOAuthClient.OauthV1().OAuthAccessTokens(), fakeUserClient.UserV1().Users(), NoopGroupMapper{}, nil, NewExpirationValidator())

	for _, tokenName := range []string{token1, token2} {
		userInfo, found, err := tokenAuthenticator.AuthenticateToken(context.TODO(), tokenName)
		if found {
			t.Error("Found token, but it should be missing!")
		}
		if err != errExpired {
			t.Errorf("Unexpected error: %v", err)
		}
		if userInfo != nil {
			t.Errorf("Unexpected user: %v", userInfo)
		}
	}
}

func TestAuthenticateTokenValidated(t *testing.T) {
	token, tokenHash := generateOAuthTokenPair()
	fakeOAuthClient := oauthfake.NewSimpleClientset(
		&oauthv1.OAuthAccessToken{
			ObjectMeta: metav1.ObjectMeta{Name: tokenHash, CreationTimestamp: metav1.Time{Time: time.Now()}},
			ExpiresIn:  600, // 10 minutes
			UserName:   "foo",
			UserUID:    string("bar"),
		},
	)
	fakeUserClient := userfake.NewSimpleClientset(&userv1.User{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "bar"}})

	tokenAuthenticator := NewTokenAuthenticator(fakeOAuthClient.OauthV1().OAuthAccessTokens(), fakeUserClient.UserV1().Users(), NoopGroupMapper{}, nil, NewExpirationValidator(), NewUIDValidator())

	userInfo, found, err := tokenAuthenticator.AuthenticateToken(context.TODO(), token)
	if !found {
		t.Error("Did not find a token!")
	}
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if userInfo == nil {
		t.Error("Did not get a user!")
	}
}

// generateOAuthTokenPair returns two tokens to use with OpenShift OAuth-based authentication.
// The first token is a private token meant to be used as a Bearer token to send
// queries to the API, the second token is a hashed token meant to be stored in
// the database.
func generateOAuthTokenPair() (privToken, pubToken string) {
	const sha256Prefix = "sha256~"
	randomToken := []byte(fmt.Sprintf("nottoorandom%d", mathrand.Int()))
	hashed := sha256.Sum256([]byte(randomToken))
	return sha256Prefix + string(randomToken), sha256Prefix + base64.RawURLEncoding.EncodeToString(hashed[:])
}
