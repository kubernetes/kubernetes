package htpasswd

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
)

func TestBasicAccessController(t *testing.T) {
	testRealm := "The-Shire"
	testUsers := []string{"bilbo", "frodo", "MiShil", "DeokMan"}
	testPasswords := []string{"baggins", "baggins", "새주", "공주님"}
	testHtpasswdContent := `bilbo:{SHA}5siv5c0SHx681xU6GiSx9ZQryqs=
							frodo:$2y$05$926C3y10Quzn/LnqQH86VOEVh/18T6RnLaS.khre96jLNL/7e.K5W
							MiShil:$2y$05$0oHgwMehvoe8iAWS8I.7l.KoECXrwVaC16RPfaSCU5eVTFrATuMI2
							DeokMan:공주님`

	tempFile, err := ioutil.TempFile("", "htpasswd-test")
	if err != nil {
		t.Fatal("could not create temporary htpasswd file")
	}
	if _, err = tempFile.WriteString(testHtpasswdContent); err != nil {
		t.Fatal("could not write temporary htpasswd file")
	}

	options := map[string]interface{}{
		"realm": testRealm,
		"path":  tempFile.Name(),
	}
	ctx := context.Background()

	accessController, err := newAccessController(options)
	if err != nil {
		t.Fatal("error creating access controller")
	}

	tempFile.Close()

	var userNumber = 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := context.WithRequest(ctx, r)
		authCtx, err := accessController.Authorized(ctx)
		if err != nil {
			switch err := err.(type) {
			case auth.Challenge:
				err.SetHeaders(w)
				w.WriteHeader(http.StatusUnauthorized)
				return
			default:
				t.Fatalf("unexpected error authorizing request: %v", err)
			}
		}

		userInfo, ok := authCtx.Value(auth.UserKey).(auth.UserInfo)
		if !ok {
			t.Fatal("basic accessController did not set auth.user context")
		}

		if userInfo.Name != testUsers[userNumber] {
			t.Fatalf("expected user name %q, got %q", testUsers[userNumber], userInfo.Name)
		}

		w.WriteHeader(http.StatusNoContent)
	}))

	client := &http.Client{
		CheckRedirect: nil,
	}

	req, _ := http.NewRequest("GET", server.URL, nil)
	resp, err := client.Do(req)

	if err != nil {
		t.Fatalf("unexpected error during GET: %v", err)
	}
	defer resp.Body.Close()

	// Request should not be authorized
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unexpected non-fail response status: %v != %v", resp.StatusCode, http.StatusUnauthorized)
	}

	nonbcrypt := map[string]struct{}{
		"bilbo":   {},
		"DeokMan": {},
	}

	for i := 0; i < len(testUsers); i++ {
		userNumber = i
		req, err := http.NewRequest("GET", server.URL, nil)
		if err != nil {
			t.Fatalf("error allocating new request: %v", err)
		}

		req.SetBasicAuth(testUsers[i], testPasswords[i])

		resp, err = client.Do(req)
		if err != nil {
			t.Fatalf("unexpected error during GET: %v", err)
		}
		defer resp.Body.Close()

		if _, ok := nonbcrypt[testUsers[i]]; ok {
			// these are not allowed.
			// Request should be authorized
			if resp.StatusCode != http.StatusUnauthorized {
				t.Fatalf("unexpected non-success response status: %v != %v for %s %s", resp.StatusCode, http.StatusUnauthorized, testUsers[i], testPasswords[i])
			}
		} else {
			// Request should be authorized
			if resp.StatusCode != http.StatusNoContent {
				t.Fatalf("unexpected non-success response status: %v != %v for %s %s", resp.StatusCode, http.StatusNoContent, testUsers[i], testPasswords[i])
			}
		}
	}

}
