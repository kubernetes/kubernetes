package browsersafe

import (
	"context"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

func TestBrowserSafeAuthorizer(t *testing.T) {
	for name, tc := range map[string]struct {
		attributes authorizer.Attributes

		expectedVerb        string
		expectedSubresource string
		expectedReason      string
	}{
		"non-resource": {
			attributes:   authorizer.AttributesRecord{ResourceRequest: false, Verb: "GET"},
			expectedVerb: "GET",
		},

		"non-proxy": {
			attributes:          authorizer.AttributesRecord{ResourceRequest: true, Verb: "get", Resource: "pods", Subresource: "logs"},
			expectedVerb:        "get",
			expectedSubresource: "logs",
		},

		"unsafe proxy subresource": {
			attributes:          authorizer.AttributesRecord{ResourceRequest: true, Verb: "get", Resource: "pods", Subresource: "proxy"},
			expectedVerb:        "get",
			expectedSubresource: "unsafeproxy",
			expectedReason:      "proxy subresource changed to unsafeproxy",
		},
		"unsafe proxy verb": {
			attributes:     authorizer.AttributesRecord{ResourceRequest: true, Verb: "proxy", Resource: "nodes"},
			expectedVerb:   "unsafeproxy",
			expectedReason: "proxy verb changed to unsafeproxy",
		},
		"unsafe proxy verb anonymous": {
			attributes: authorizer.AttributesRecord{ResourceRequest: true, Verb: "proxy", Resource: "nodes",
				User: &user.DefaultInfo{Name: "system:anonymous", Groups: []string{"system:unauthenticated"}}},
			expectedVerb:   "unsafeproxy",
			expectedReason: "proxy verb changed to unsafeproxy",
		},

		"proxy subresource authenticated": {
			attributes: authorizer.AttributesRecord{ResourceRequest: true, Verb: "get", Resource: "pods", Subresource: "proxy",
				User: &user.DefaultInfo{Name: "bob", Groups: []string{"system:authenticated"}}},
			expectedVerb:        "get",
			expectedSubresource: "proxy",
		},
	} {
		delegateAuthorizer := &recordingAuthorizer{}
		safeAuthorizer := NewBrowserSafeAuthorizer(delegateAuthorizer, "system:authenticated")

		authorized, reason, err := safeAuthorizer.Authorize(context.TODO(), tc.attributes)
		if authorized == authorizer.DecisionAllow || reason != tc.expectedReason || err != nil {
			t.Errorf("%s: unexpected output: %v %s %v", name, authorized, reason, err)
			continue
		}

		if delegateAuthorizer.attributes.GetVerb() != tc.expectedVerb {
			t.Errorf("%s: expected verb %s, got %s", name, tc.expectedVerb, delegateAuthorizer.attributes.GetVerb())
		}
		if delegateAuthorizer.attributes.GetSubresource() != tc.expectedSubresource {
			t.Errorf("%s: expected verb %s, got %s", name, tc.expectedSubresource, delegateAuthorizer.attributes.GetSubresource())
		}
	}
}

type recordingAuthorizer struct {
	attributes authorizer.Attributes
}

func (t *recordingAuthorizer) Authorize(_ context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	t.attributes = a
	return authorizer.DecisionNoOpinion, "", nil
}
