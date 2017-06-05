package handlers

import "testing"

var blobUploadStates = []blobUploadState{
	{
		Name:   "hello",
		UUID:   "abcd-1234-qwer-0987",
		Offset: 0,
	},
	{
		Name:   "hello-world",
		UUID:   "abcd-1234-qwer-0987",
		Offset: 0,
	},
	{
		Name:   "h3ll0_w0rld",
		UUID:   "abcd-1234-qwer-0987",
		Offset: 1337,
	},
	{
		Name:   "ABCDEFG",
		UUID:   "ABCD-1234-QWER-0987",
		Offset: 1234567890,
	},
	{
		Name:   "this-is-A-sort-of-Long-name-for-Testing",
		UUID:   "dead-1234-beef-0987",
		Offset: 8675309,
	},
}

var secrets = []string{
	"supersecret",
	"12345",
	"a",
	"SuperSecret",
	"Sup3r... S3cr3t!",
	"This is a reasonably long secret key that is used for the purpose of testing.",
	"\u2603+\u2744", // snowman+snowflake
}

// TestLayerUploadTokens constructs stateTokens from LayerUploadStates and
// validates that the tokens can be used to reconstruct the proper upload state.
func TestLayerUploadTokens(t *testing.T) {
	secret := hmacKey("supersecret")

	for _, testcase := range blobUploadStates {
		token, err := secret.packUploadState(testcase)
		if err != nil {
			t.Fatal(err)
		}

		lus, err := secret.unpackUploadState(token)
		if err != nil {
			t.Fatal(err)
		}

		assertBlobUploadStateEquals(t, testcase, lus)
	}
}

// TestHMACValidate ensures that any HMAC token providers are compatible if and
// only if they share the same secret.
func TestHMACValidation(t *testing.T) {
	for _, secret := range secrets {
		secret1 := hmacKey(secret)
		secret2 := hmacKey(secret)
		badSecret := hmacKey("DifferentSecret")

		for _, testcase := range blobUploadStates {
			token, err := secret1.packUploadState(testcase)
			if err != nil {
				t.Fatal(err)
			}

			lus, err := secret2.unpackUploadState(token)
			if err != nil {
				t.Fatal(err)
			}

			assertBlobUploadStateEquals(t, testcase, lus)

			_, err = badSecret.unpackUploadState(token)
			if err == nil {
				t.Fatalf("Expected token provider to fail at retrieving state from token: %s", token)
			}

			badToken, err := badSecret.packUploadState(lus)
			if err != nil {
				t.Fatal(err)
			}

			_, err = secret1.unpackUploadState(badToken)
			if err == nil {
				t.Fatalf("Expected token provider to fail at retrieving state from token: %s", badToken)
			}

			_, err = secret2.unpackUploadState(badToken)
			if err == nil {
				t.Fatalf("Expected token provider to fail at retrieving state from token: %s", badToken)
			}
		}
	}
}

func assertBlobUploadStateEquals(t *testing.T, expected blobUploadState, received blobUploadState) {
	if expected.Name != received.Name {
		t.Fatalf("Expected Name=%q, Received Name=%q", expected.Name, received.Name)
	}
	if expected.UUID != received.UUID {
		t.Fatalf("Expected UUID=%q, Received UUID=%q", expected.UUID, received.UUID)
	}
	if expected.Offset != received.Offset {
		t.Fatalf("Expected Offset=%d, Received Offset=%d", expected.Offset, received.Offset)
	}
}
