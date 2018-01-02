package trustgraph

import (
	"crypto/x509"
	"encoding/json"
	"io"
	"io/ioutil"
	"sort"
	"strings"
	"time"

	"github.com/docker/libtrust"
)

type jsonGrant struct {
	Subject    string `json:"subject"`
	Permission uint16 `json:"permission"`
	Grantee    string `json:"grantee"`
}

type jsonRevocation struct {
	Subject    string `json:"subject"`
	Revocation uint16 `json:"revocation"`
	Grantee    string `json:"grantee"`
}

type jsonStatement struct {
	Revocations []*jsonRevocation `json:"revocations"`
	Grants      []*jsonGrant      `json:"grants"`
	Expiration  time.Time         `json:"expiration"`
	IssuedAt    time.Time         `json:"issuedAt"`
}

func (g *jsonGrant) Grant(statement *Statement) *Grant {
	return &Grant{
		Subject:    g.Subject,
		Permission: g.Permission,
		Grantee:    g.Grantee,
		statement:  statement,
	}
}

// Statement represents a set of grants made from a verifiable
// authority.  A statement has an expiration associated with it
// set by the authority.
type Statement struct {
	jsonStatement

	signature *libtrust.JSONSignature
}

// IsExpired returns whether the statement has expired
func (s *Statement) IsExpired() bool {
	return s.Expiration.Before(time.Now().Add(-10 * time.Second))
}

// Bytes returns an indented json representation of the statement
// in a byte array.  This value can be written to a file or stream
// without alteration.
func (s *Statement) Bytes() ([]byte, error) {
	return s.signature.PrettySignature("signatures")
}

// LoadStatement loads and verifies a statement from an input stream.
func LoadStatement(r io.Reader, authority *x509.CertPool) (*Statement, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	js, err := libtrust.ParsePrettySignature(b, "signatures")
	if err != nil {
		return nil, err
	}
	payload, err := js.Payload()
	if err != nil {
		return nil, err
	}
	var statement Statement
	err = json.Unmarshal(payload, &statement.jsonStatement)
	if err != nil {
		return nil, err
	}

	if authority == nil {
		_, err = js.Verify()
		if err != nil {
			return nil, err
		}
	} else {
		_, err = js.VerifyChains(authority)
		if err != nil {
			return nil, err
		}
	}
	statement.signature = js

	return &statement, nil
}

// CreateStatements creates and signs a statement from a stream of grants
// and revocations in a JSON array.
func CreateStatement(grants, revocations io.Reader, expiration time.Duration, key libtrust.PrivateKey, chain []*x509.Certificate) (*Statement, error) {
	var statement Statement
	err := json.NewDecoder(grants).Decode(&statement.jsonStatement.Grants)
	if err != nil {
		return nil, err
	}
	err = json.NewDecoder(revocations).Decode(&statement.jsonStatement.Revocations)
	if err != nil {
		return nil, err
	}
	statement.jsonStatement.Expiration = time.Now().UTC().Add(expiration)
	statement.jsonStatement.IssuedAt = time.Now().UTC()

	b, err := json.MarshalIndent(&statement.jsonStatement, "", "   ")
	if err != nil {
		return nil, err
	}

	statement.signature, err = libtrust.NewJSONSignature(b)
	if err != nil {
		return nil, err
	}
	err = statement.signature.SignWithChain(key, chain)
	if err != nil {
		return nil, err
	}

	return &statement, nil
}

type statementList []*Statement

func (s statementList) Len() int {
	return len(s)
}

func (s statementList) Less(i, j int) bool {
	return s[i].IssuedAt.Before(s[j].IssuedAt)
}

func (s statementList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// CollapseStatements returns a single list of the valid statements as well as the
// time when the next grant will expire.
func CollapseStatements(statements []*Statement, useExpired bool) ([]*Grant, time.Time, error) {
	sorted := make(statementList, 0, len(statements))
	for _, statement := range statements {
		if useExpired || !statement.IsExpired() {
			sorted = append(sorted, statement)
		}
	}
	sort.Sort(sorted)

	var minExpired time.Time
	var grantCount int
	roots := map[string]*grantNode{}
	for i, statement := range sorted {
		if statement.Expiration.Before(minExpired) || i == 0 {
			minExpired = statement.Expiration
		}
		for _, grant := range statement.Grants {
			parts := strings.Split(grant.Grantee, "/")
			nodes := roots
			g := grant.Grant(statement)
			grantCount = grantCount + 1

			for _, part := range parts {
				node, nodeOk := nodes[part]
				if !nodeOk {
					node = newGrantNode()
					nodes[part] = node
				}
				node.grants = append(node.grants, g)
				nodes = node.children
			}
		}

		for _, revocation := range statement.Revocations {
			parts := strings.Split(revocation.Grantee, "/")
			nodes := roots

			var node *grantNode
			var nodeOk bool
			for _, part := range parts {
				node, nodeOk = nodes[part]
				if !nodeOk {
					break
				}
				nodes = node.children
			}
			if node != nil {
				for _, grant := range node.grants {
					if isSubName(grant.Subject, revocation.Subject) {
						grant.Permission = grant.Permission &^ revocation.Revocation
					}
				}
			}
		}
	}

	retGrants := make([]*Grant, 0, grantCount)
	for _, rootNodes := range roots {
		retGrants = append(retGrants, rootNodes.grants...)
	}

	return retGrants, minExpired, nil
}

// FilterStatements filters the statements to statements including the given grants.
func FilterStatements(grants []*Grant) ([]*Statement, error) {
	statements := map[*Statement]bool{}
	for _, grant := range grants {
		if grant.statement != nil {
			statements[grant.statement] = true
		}
	}
	retStatements := make([]*Statement, len(statements))
	var i int
	for statement := range statements {
		retStatements[i] = statement
		i++
	}
	return retStatements, nil
}
