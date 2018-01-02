package docker

import (
	"context"
	"net/url"
	"sort"
	"strings"

	"github.com/containerd/containerd/reference"
)

// repositoryScope returns a repository scope string such as "repository:foo/bar:pull"
// for "host/foo/bar:baz".
// When push is true, both pull and push are added to the scope.
func repositoryScope(refspec reference.Spec, push bool) (string, error) {
	u, err := url.Parse("dummy://" + refspec.Locator)
	if err != nil {
		return "", err
	}
	s := "repository:" + strings.TrimPrefix(u.Path, "/") + ":pull"
	if push {
		s += ",push"
	}
	return s, nil
}

// tokenScopesKey is used for the key for context.WithValue().
// value: []string (e.g. {"registry:foo/bar:pull"})
type tokenScopesKey struct{}

// contextWithRepositoryScope returns a context with tokenScopesKey{} and the repository scope value.
func contextWithRepositoryScope(ctx context.Context, refspec reference.Spec, push bool) (context.Context, error) {
	s, err := repositoryScope(refspec, push)
	if err != nil {
		return nil, err
	}
	return context.WithValue(ctx, tokenScopesKey{}, []string{s}), nil
}

// getTokenScopes returns deduplicated and sorted scopes from ctx.Value(tokenScopesKey{}) and params["scope"].
func getTokenScopes(ctx context.Context, params map[string]string) []string {
	var scopes []string
	if x := ctx.Value(tokenScopesKey{}); x != nil {
		scopes = append(scopes, x.([]string)...)
	}
	if scope, ok := params["scope"]; ok {
		for _, s := range scopes {
			// Note: this comparison is unaware of the scope grammar (https://docs.docker.com/registry/spec/auth/scope/)
			// So, "repository:foo/bar:pull,push" != "repository:foo/bar:push,pull", although semantically they are equal.
			if s == scope {
				// already appended
				goto Sort
			}
		}
		scopes = append(scopes, scope)
	}
Sort:
	sort.Strings(scopes)
	return scopes
}
