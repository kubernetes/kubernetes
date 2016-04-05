// package testing provides code useful for testing code that interacts with OIDC components
package testing

import "github.com/coreos/go-oidc/jose"

// FakeClient implements OIDCClient with canned responses.
type FakeClient struct {
	Err     error
	IDToken jose.JWT
}

func (f *FakeClient) RefreshToken(rt string) (jose.JWT, error) {
	return f.IDToken, f.Err
}
