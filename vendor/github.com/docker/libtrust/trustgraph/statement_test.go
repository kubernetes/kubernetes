package trustgraph

import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"testing"
	"time"

	"github.com/docker/libtrust"
	"github.com/docker/libtrust/testutil"
)

const testStatementExpiration = time.Hour * 5

func generateStatement(grants []*Grant, key libtrust.PrivateKey, chain []*x509.Certificate) (*Statement, error) {
	var statement Statement

	statement.Grants = make([]*jsonGrant, len(grants))
	for i, grant := range grants {
		statement.Grants[i] = &jsonGrant{
			Subject:    grant.Subject,
			Permission: grant.Permission,
			Grantee:    grant.Grantee,
		}
	}
	statement.IssuedAt = time.Now()
	statement.Expiration = time.Now().Add(testStatementExpiration)
	statement.Revocations = make([]*jsonRevocation, 0)

	marshalled, err := json.MarshalIndent(statement.jsonStatement, "", "   ")
	if err != nil {
		return nil, err
	}

	sig, err := libtrust.NewJSONSignature(marshalled)
	if err != nil {
		return nil, err
	}
	err = sig.SignWithChain(key, chain)
	if err != nil {
		return nil, err
	}
	statement.signature = sig

	return &statement, nil
}

func generateTrustChain(t *testing.T, chainLen int) (libtrust.PrivateKey, *x509.CertPool, []*x509.Certificate) {
	caKey, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating key: %s", err)
	}
	ca, err := testutil.GenerateTrustCA(caKey.CryptoPublicKey(), caKey.CryptoPrivateKey())
	if err != nil {
		t.Fatalf("Error generating ca: %s", err)
	}

	parent := ca
	parentKey := caKey
	chain := make([]*x509.Certificate, chainLen)
	for i := chainLen - 1; i > 0; i-- {
		intermediatekey, err := libtrust.GenerateECP256PrivateKey()
		if err != nil {
			t.Fatalf("Error generate key: %s", err)
		}
		chain[i], err = testutil.GenerateIntermediate(intermediatekey.CryptoPublicKey(), parentKey.CryptoPrivateKey(), parent)
		if err != nil {
			t.Fatalf("Error generating intermdiate certificate: %s", err)
		}
		parent = chain[i]
		parentKey = intermediatekey
	}
	trustKey, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generate key: %s", err)
	}
	chain[0], err = testutil.GenerateTrustCert(trustKey.CryptoPublicKey(), parentKey.CryptoPrivateKey(), parent)
	if err != nil {
		t.Fatalf("Error generate trust cert: %s", err)
	}

	caPool := x509.NewCertPool()
	caPool.AddCert(ca)

	return trustKey, caPool, chain
}

func TestLoadStatement(t *testing.T) {
	grantCount := 4
	grants, _ := createTestKeysAndGrants(grantCount)

	trustKey, caPool, chain := generateTrustChain(t, 6)

	statement, err := generateStatement(grants, trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}

	statementBytes, err := statement.Bytes()
	if err != nil {
		t.Fatalf("Error getting statement bytes: %s", err)
	}

	s2, err := LoadStatement(bytes.NewReader(statementBytes), caPool)
	if err != nil {
		t.Fatalf("Error loading statement: %s", err)
	}
	if len(s2.Grants) != grantCount {
		t.Fatalf("Unexpected grant length\n\tExpected: %d\n\tActual: %d", grantCount, len(s2.Grants))
	}

	pool := x509.NewCertPool()
	_, err = LoadStatement(bytes.NewReader(statementBytes), pool)
	if err == nil {
		t.Fatalf("No error thrown verifying without an authority")
	} else if _, ok := err.(x509.UnknownAuthorityError); !ok {
		t.Fatalf("Unexpected error verifying without authority: %s", err)
	}

	s2, err = LoadStatement(bytes.NewReader(statementBytes), nil)
	if err != nil {
		t.Fatalf("Error loading statement: %s", err)
	}
	if len(s2.Grants) != grantCount {
		t.Fatalf("Unexpected grant length\n\tExpected: %d\n\tActual: %d", grantCount, len(s2.Grants))
	}

	badData := make([]byte, len(statementBytes))
	copy(badData, statementBytes)
	badData[0] = '['
	_, err = LoadStatement(bytes.NewReader(badData), nil)
	if err == nil {
		t.Fatalf("No error thrown parsing bad json")
	}

	alteredData := make([]byte, len(statementBytes))
	copy(alteredData, statementBytes)
	alteredData[30] = '0'
	_, err = LoadStatement(bytes.NewReader(alteredData), nil)
	if err == nil {
		t.Fatalf("No error thrown from bad data")
	}
}

func TestCollapseGrants(t *testing.T) {
	grantCount := 8
	grants, keys := createTestKeysAndGrants(grantCount)
	linkGrants := make([]*Grant, 4)
	linkGrants[0] = &Grant{
		Subject:    "/user-3",
		Permission: 0x0f,
		Grantee:    "/user-2",
	}
	linkGrants[1] = &Grant{
		Subject:    "/user-3/sub-project",
		Permission: 0x0f,
		Grantee:    "/user-4",
	}
	linkGrants[2] = &Grant{
		Subject:    "/user-6",
		Permission: 0x0f,
		Grantee:    "/user-7",
	}
	linkGrants[3] = &Grant{
		Subject:    "/user-6/sub-project/specific-app",
		Permission: 0x0f,
		Grantee:    "/user-5",
	}
	trustKey, pool, chain := generateTrustChain(t, 3)

	statements := make([]*Statement, 3)
	var err error
	statements[0], err = generateStatement(grants[0:4], trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	statements[1], err = generateStatement(grants[4:], trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	statements[2], err = generateStatement(linkGrants, trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}

	statementsCopy := make([]*Statement, len(statements))
	for i, statement := range statements {
		b, err := statement.Bytes()
		if err != nil {
			t.Fatalf("Error getting statement bytes: %s", err)
		}
		verifiedStatement, err := LoadStatement(bytes.NewReader(b), pool)
		if err != nil {
			t.Fatalf("Error loading statement: %s", err)
		}
		// Force sort by reversing order
		statementsCopy[len(statementsCopy)-i-1] = verifiedStatement
	}
	statements = statementsCopy

	collapsedGrants, expiration, err := CollapseStatements(statements, false)
	if len(collapsedGrants) != 12 {
		t.Fatalf("Unexpected number of grants\n\tExpected: %d\n\tActual: %d", 12, len(collapsedGrants))
	}
	if expiration.After(time.Now().Add(time.Hour*5)) || expiration.Before(time.Now()) {
		t.Fatalf("Unexpected expiration time: %s", expiration.String())
	}
	g := NewMemoryGraph(collapsedGrants)

	testVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-1", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-2", 0x0f)
	testVerified(t, g, keys[2].PublicKey(), "user-key-3", "/user-3", 0x0f)
	testVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-4", 0x0f)
	testVerified(t, g, keys[4].PublicKey(), "user-key-5", "/user-5", 0x0f)
	testVerified(t, g, keys[5].PublicKey(), "user-key-6", "/user-6", 0x0f)
	testVerified(t, g, keys[6].PublicKey(), "user-key-7", "/user-7", 0x0f)
	testVerified(t, g, keys[7].PublicKey(), "user-key-8", "/user-8", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-3", 0x0f)
	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-3/sub-project/specific-app", 0x0f)
	testVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-3/sub-project", 0x0f)
	testVerified(t, g, keys[6].PublicKey(), "user-key-7", "/user-6", 0x0f)
	testVerified(t, g, keys[6].PublicKey(), "user-key-7", "/user-6/sub-project/specific-app", 0x0f)
	testVerified(t, g, keys[4].PublicKey(), "user-key-5", "/user-6/sub-project/specific-app", 0x0f)

	testNotVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-3", 0x0f)
	testNotVerified(t, g, keys[3].PublicKey(), "user-key-4", "/user-6/sub-project", 0x0f)
	testNotVerified(t, g, keys[4].PublicKey(), "user-key-5", "/user-6/sub-project", 0x0f)

	// Add revocation grant
	statements = append(statements, &Statement{
		jsonStatement{
			IssuedAt:   time.Now(),
			Expiration: time.Now().Add(testStatementExpiration),
			Grants:     []*jsonGrant{},
			Revocations: []*jsonRevocation{
				&jsonRevocation{
					Subject:    "/user-1",
					Revocation: 0x0f,
					Grantee:    keys[0].KeyID(),
				},
				&jsonRevocation{
					Subject:    "/user-2",
					Revocation: 0x08,
					Grantee:    keys[1].KeyID(),
				},
				&jsonRevocation{
					Subject:    "/user-6",
					Revocation: 0x0f,
					Grantee:    "/user-7",
				},
				&jsonRevocation{
					Subject:    "/user-9",
					Revocation: 0x0f,
					Grantee:    "/user-10",
				},
			},
		},
		nil,
	})

	collapsedGrants, expiration, err = CollapseStatements(statements, false)
	if len(collapsedGrants) != 12 {
		t.Fatalf("Unexpected number of grants\n\tExpected: %d\n\tActual: %d", 12, len(collapsedGrants))
	}
	if expiration.After(time.Now().Add(time.Hour*5)) || expiration.Before(time.Now()) {
		t.Fatalf("Unexpected expiration time: %s", expiration.String())
	}
	g = NewMemoryGraph(collapsedGrants)

	testNotVerified(t, g, keys[0].PublicKey(), "user-key-1", "/user-1", 0x0f)
	testNotVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-2", 0x0f)
	testNotVerified(t, g, keys[6].PublicKey(), "user-key-7", "/user-6/sub-project/specific-app", 0x0f)

	testVerified(t, g, keys[1].PublicKey(), "user-key-2", "/user-2", 0x07)
}

func TestFilterStatements(t *testing.T) {
	grantCount := 8
	grants, keys := createTestKeysAndGrants(grantCount)
	linkGrants := make([]*Grant, 3)
	linkGrants[0] = &Grant{
		Subject:    "/user-3",
		Permission: 0x0f,
		Grantee:    "/user-2",
	}
	linkGrants[1] = &Grant{
		Subject:    "/user-5",
		Permission: 0x0f,
		Grantee:    "/user-4",
	}
	linkGrants[2] = &Grant{
		Subject:    "/user-7",
		Permission: 0x0f,
		Grantee:    "/user-6",
	}

	trustKey, _, chain := generateTrustChain(t, 3)

	statements := make([]*Statement, 5)
	var err error
	statements[0], err = generateStatement(grants[0:2], trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	statements[1], err = generateStatement(grants[2:4], trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	statements[2], err = generateStatement(grants[4:6], trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	statements[3], err = generateStatement(grants[6:], trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	statements[4], err = generateStatement(linkGrants, trustKey, chain)
	if err != nil {
		t.Fatalf("Error generating statement: %s", err)
	}
	collapsed, _, err := CollapseStatements(statements, false)
	if err != nil {
		t.Fatalf("Error collapsing grants: %s", err)
	}

	// Filter 1, all 5 statements
	filter1, err := FilterStatements(collapsed)
	if err != nil {
		t.Fatalf("Error filtering statements: %s", err)
	}
	if len(filter1) != 5 {
		t.Fatalf("Wrong number of statements, expected %d, received %d", 5, len(filter1))
	}

	// Filter 2, one statement
	filter2, err := FilterStatements([]*Grant{collapsed[0]})
	if err != nil {
		t.Fatalf("Error filtering statements: %s", err)
	}
	if len(filter2) != 1 {
		t.Fatalf("Wrong number of statements, expected %d, received %d", 1, len(filter2))
	}

	// Filter 3, 2 statements, from graph lookup
	g := NewMemoryGraph(collapsed)
	lookupGrants, err := g.GetGrants(keys[1], "/user-3", 0x0f)
	if err != nil {
		t.Fatalf("Error looking up grants: %s", err)
	}
	if len(lookupGrants) != 1 {
		t.Fatalf("Wrong numberof grant chains returned from lookup, expected %d, received %d", 1, len(lookupGrants))
	}
	if len(lookupGrants[0]) != 2 {
		t.Fatalf("Wrong number of grants looked up, expected %d, received %d", 2, len(lookupGrants))
	}
	filter3, err := FilterStatements(lookupGrants[0])
	if err != nil {
		t.Fatalf("Error filtering statements: %s", err)
	}
	if len(filter3) != 2 {
		t.Fatalf("Wrong number of statements, expected %d, received %d", 2, len(filter3))
	}

}

func TestCreateStatement(t *testing.T) {
	grantJSON := bytes.NewReader([]byte(`[
   {
      "subject": "/user-2",
      "permission": 15,
      "grantee": "/user-1"
   },
   {
      "subject": "/user-7",
      "permission": 1,
      "grantee": "/user-9"
   },
   {
      "subject": "/user-3",
      "permission": 15,
      "grantee": "/user-2"
   }
]`))
	revocationJSON := bytes.NewReader([]byte(`[
   {
      "subject": "user-8",
      "revocation": 12,
      "grantee": "user-9"
   }
]`))

	trustKey, pool, chain := generateTrustChain(t, 3)

	statement, err := CreateStatement(grantJSON, revocationJSON, testStatementExpiration, trustKey, chain)
	if err != nil {
		t.Fatalf("Error creating statement: %s", err)
	}

	b, err := statement.Bytes()
	if err != nil {
		t.Fatalf("Error retrieving bytes: %s", err)
	}

	verified, err := LoadStatement(bytes.NewReader(b), pool)
	if err != nil {
		t.Fatalf("Error loading statement: %s", err)
	}

	if len(verified.Grants) != 3 {
		t.Errorf("Unexpected number of grants, expected %d, received %d", 3, len(verified.Grants))
	}

	if len(verified.Revocations) != 1 {
		t.Errorf("Unexpected number of revocations, expected %d, received %d", 1, len(verified.Revocations))
	}
}
