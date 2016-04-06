// package testing provides code useful for testing code that interacts with OIDC components
package testing

import (
	"fmt"

	"github.com/coreos/go-oidc/jose"
)

// FakeClient implements OIDCClient with canned responses.
type FakeClient struct {
	Err     error
	IDToken jose.JWT

	ExpectRefreshToken string
}

func (f *FakeClient) RefreshToken(rt string) (jose.JWT, error) {
	if rt != f.ExpectRefreshToken {
		return jose.JWT{}, fmt.Errorf("Unexpected RT: expected %v, got %v", f.ExpectRefreshToken, rt)
	}
	return f.IDToken, f.Err
}
