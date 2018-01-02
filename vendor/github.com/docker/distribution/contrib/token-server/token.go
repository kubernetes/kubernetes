package main

import (
	"crypto"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"
	"time"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
	"github.com/docker/distribution/registry/auth/token"
	"github.com/docker/libtrust"
)

// ResolveScopeSpecifiers converts a list of scope specifiers from a token
// request's `scope` query parameters into a list of standard access objects.
func ResolveScopeSpecifiers(ctx context.Context, scopeSpecs []string) []auth.Access {
	requestedAccessSet := make(map[auth.Access]struct{}, 2*len(scopeSpecs))

	for _, scopeSpecifier := range scopeSpecs {
		// There should be 3 parts, separated by a `:` character.
		parts := strings.SplitN(scopeSpecifier, ":", 3)

		if len(parts) != 3 {
			context.GetLogger(ctx).Infof("ignoring unsupported scope format %s", scopeSpecifier)
			continue
		}

		resourceType, resourceName, actions := parts[0], parts[1], parts[2]

		resourceType, resourceClass := splitResourceClass(resourceType)
		if resourceType == "" {
			continue
		}

		// Actions should be a comma-separated list of actions.
		for _, action := range strings.Split(actions, ",") {
			requestedAccess := auth.Access{
				Resource: auth.Resource{
					Type:  resourceType,
					Class: resourceClass,
					Name:  resourceName,
				},
				Action: action,
			}

			// Add this access to the requested access set.
			requestedAccessSet[requestedAccess] = struct{}{}
		}
	}

	requestedAccessList := make([]auth.Access, 0, len(requestedAccessSet))
	for requestedAccess := range requestedAccessSet {
		requestedAccessList = append(requestedAccessList, requestedAccess)
	}

	return requestedAccessList
}

var typeRegexp = regexp.MustCompile(`^([a-z0-9]+)(\([a-z0-9]+\))?$`)

func splitResourceClass(t string) (string, string) {
	matches := typeRegexp.FindStringSubmatch(t)
	if len(matches) < 2 {
		return "", ""
	}
	if len(matches) == 2 || len(matches[2]) < 2 {
		return matches[1], ""
	}
	return matches[1], matches[2][1 : len(matches[2])-1]
}

// ResolveScopeList converts a scope list from a token request's
// `scope` parameter into a list of standard access objects.
func ResolveScopeList(ctx context.Context, scopeList string) []auth.Access {
	scopes := strings.Split(scopeList, " ")
	return ResolveScopeSpecifiers(ctx, scopes)
}

func scopeString(a auth.Access) string {
	if a.Class != "" {
		return fmt.Sprintf("%s(%s):%s:%s", a.Type, a.Class, a.Name, a.Action)
	}
	return fmt.Sprintf("%s:%s:%s", a.Type, a.Name, a.Action)
}

// ToScopeList converts a list of access to a
// scope list string
func ToScopeList(access []auth.Access) string {
	var s []string
	for _, a := range access {
		s = append(s, scopeString(a))
	}
	return strings.Join(s, ",")
}

// TokenIssuer represents an issuer capable of generating JWT tokens
type TokenIssuer struct {
	Issuer     string
	SigningKey libtrust.PrivateKey
	Expiration time.Duration
}

// CreateJWT creates and signs a JSON Web Token for the given subject and
// audience with the granted access.
func (issuer *TokenIssuer) CreateJWT(subject string, audience string, grantedAccessList []auth.Access) (string, error) {
	// Make a set of access entries to put in the token's claimset.
	resourceActionSets := make(map[auth.Resource]map[string]struct{}, len(grantedAccessList))
	for _, access := range grantedAccessList {
		actionSet, exists := resourceActionSets[access.Resource]
		if !exists {
			actionSet = map[string]struct{}{}
			resourceActionSets[access.Resource] = actionSet
		}
		actionSet[access.Action] = struct{}{}
	}

	accessEntries := make([]*token.ResourceActions, 0, len(resourceActionSets))
	for resource, actionSet := range resourceActionSets {
		actions := make([]string, 0, len(actionSet))
		for action := range actionSet {
			actions = append(actions, action)
		}

		accessEntries = append(accessEntries, &token.ResourceActions{
			Type:    resource.Type,
			Class:   resource.Class,
			Name:    resource.Name,
			Actions: actions,
		})
	}

	randomBytes := make([]byte, 15)
	_, err := io.ReadFull(rand.Reader, randomBytes)
	if err != nil {
		return "", err
	}
	randomID := base64.URLEncoding.EncodeToString(randomBytes)

	now := time.Now()

	signingHash := crypto.SHA256
	var alg string
	switch issuer.SigningKey.KeyType() {
	case "RSA":
		alg = "RS256"
	case "EC":
		alg = "ES256"
	default:
		panic(fmt.Errorf("unsupported signing key type %q", issuer.SigningKey.KeyType()))
	}

	joseHeader := token.Header{
		Type:       "JWT",
		SigningAlg: alg,
	}

	if x5c := issuer.SigningKey.GetExtendedField("x5c"); x5c != nil {
		joseHeader.X5c = x5c.([]string)
	} else {
		var jwkMessage json.RawMessage
		jwkMessage, err = issuer.SigningKey.PublicKey().MarshalJSON()
		if err != nil {
			return "", err
		}
		joseHeader.RawJWK = &jwkMessage
	}

	exp := issuer.Expiration
	if exp == 0 {
		exp = 5 * time.Minute
	}

	claimSet := token.ClaimSet{
		Issuer:     issuer.Issuer,
		Subject:    subject,
		Audience:   audience,
		Expiration: now.Add(exp).Unix(),
		NotBefore:  now.Unix(),
		IssuedAt:   now.Unix(),
		JWTID:      randomID,

		Access: accessEntries,
	}

	var (
		joseHeaderBytes []byte
		claimSetBytes   []byte
	)

	if joseHeaderBytes, err = json.Marshal(joseHeader); err != nil {
		return "", fmt.Errorf("unable to encode jose header: %s", err)
	}
	if claimSetBytes, err = json.Marshal(claimSet); err != nil {
		return "", fmt.Errorf("unable to encode claim set: %s", err)
	}

	encodedJoseHeader := joseBase64Encode(joseHeaderBytes)
	encodedClaimSet := joseBase64Encode(claimSetBytes)
	encodingToSign := fmt.Sprintf("%s.%s", encodedJoseHeader, encodedClaimSet)

	var signatureBytes []byte
	if signatureBytes, _, err = issuer.SigningKey.Sign(strings.NewReader(encodingToSign), signingHash); err != nil {
		return "", fmt.Errorf("unable to sign jwt payload: %s", err)
	}

	signature := joseBase64Encode(signatureBytes)

	return fmt.Sprintf("%s.%s", encodingToSign, signature), nil
}

func joseBase64Encode(data []byte) string {
	return strings.TrimRight(base64.URLEncoding.EncodeToString(data), "=")
}
