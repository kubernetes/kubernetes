package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestKeys_List(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account/keys", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"ssh_keys":[{"id":1},{"id":2}]}`)
	})

	keys, _, err := client.Keys.List(nil)
	if err != nil {
		t.Errorf("Keys.List returned error: %v", err)
	}

	expected := []Key{{ID: 1}, {ID: 2}}
	if !reflect.DeepEqual(keys, expected) {
		t.Errorf("Keys.List returned %+v, expected %+v", keys, expected)
	}
}

func TestKeys_ListKeysMultiplePages(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account/keys", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"droplets": [{"id":1},{"id":2}], "links":{"pages":{"next":"http://example.com/v2/account/keys/?page=2"}}}`)
	})

	_, resp, err := client.Keys.List(nil)
	if err != nil {
		t.Fatal(err)
	}
	checkCurrentPage(t, resp, 1)
}

func TestKeys_RetrievePageByNumber(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"keys": [{"id":1},{"id":2}],
		"links":{
			"pages":{
				"next":"http://example.com/v2/account/keys/?page=3",
				"prev":"http://example.com/v2/account/keys/?page=1",
				"last":"http://example.com/v2/account/keys/?page=3",
				"first":"http://example.com/v2/account/keys/?page=1"
			}
		}
	}`

	mux.HandleFunc("/v2/account/keys", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	opt := &ListOptions{Page: 2}
	_, resp, err := client.Keys.List(opt)
	if err != nil {
		t.Fatal(err)
	}
	checkCurrentPage(t, resp, 2)
}

func TestKeys_GetByID(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account/keys/12345", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"ssh_key": {"id":12345}}`)
	})

	keys, _, err := client.Keys.GetByID(12345)
	if err != nil {
		t.Errorf("Keys.GetByID returned error: %v", err)
	}

	expected := &Key{ID: 12345}
	if !reflect.DeepEqual(keys, expected) {
		t.Errorf("Keys.GetByID returned %+v, expected %+v", keys, expected)
	}
}

func TestKeys_GetByFingerprint(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account/keys/aa:bb:cc", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"ssh_key": {"fingerprint":"aa:bb:cc"}}`)
	})

	keys, _, err := client.Keys.GetByFingerprint("aa:bb:cc")
	if err != nil {
		t.Errorf("Keys.GetByFingerprint returned error: %v", err)
	}

	expected := &Key{Fingerprint: "aa:bb:cc"}
	if !reflect.DeepEqual(keys, expected) {
		t.Errorf("Keys.GetByFingerprint returned %+v, expected %+v", keys, expected)
	}
}

func TestKeys_Create(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &KeyCreateRequest{
		Name:      "name",
		PublicKey: "ssh-rsa longtextandstuff",
	}

	mux.HandleFunc("/v2/account/keys", func(w http.ResponseWriter, r *http.Request) {
		v := new(KeyCreateRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, createRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, createRequest)
		}

		fmt.Fprintf(w, `{"ssh_key":{"id":1}}`)
	})

	key, _, err := client.Keys.Create(createRequest)
	if err != nil {
		t.Errorf("Keys.Create returned error: %v", err)
	}

	expected := &Key{ID: 1}
	if !reflect.DeepEqual(key, expected) {
		t.Errorf("Keys.Create returned %+v, expected %+v", key, expected)
	}
}

func TestKeys_UpdateByID(t *testing.T) {
	setup()
	defer teardown()

	updateRequest := &KeyUpdateRequest{
		Name: "name",
	}

	mux.HandleFunc("/v2/account/keys/12345", func(w http.ResponseWriter, r *http.Request) {
		expected := map[string]interface{}{
			"name": "name",
		}

		var v map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		if !reflect.DeepEqual(v, expected) {
			t.Errorf("Request body = %#v, expected %#v", v, expected)
		}

		fmt.Fprintf(w, `{"ssh_key":{"id":1}}`)
	})

	key, _, err := client.Keys.UpdateByID(12345, updateRequest)
	if err != nil {
		t.Errorf("Keys.Update returned error: %v", err)
	} else {
		if id := key.ID; id != 1 {
			t.Errorf("expected id '%d', received '%d'", 1, id)
		}
	}
}

func TestKeys_UpdateByFingerprint(t *testing.T) {
	setup()
	defer teardown()

	updateRequest := &KeyUpdateRequest{
		Name: "name",
	}

	mux.HandleFunc("/v2/account/keys/3b:16:bf:e4:8b:00:8b:b8:59:8c:a9:d3:f0:19:45:fa", func(w http.ResponseWriter, r *http.Request) {
		expected := map[string]interface{}{
			"name": "name",
		}

		var v map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		if !reflect.DeepEqual(v, expected) {
			t.Errorf("Request body = %#v, expected %#v", v, expected)
		}

		fmt.Fprintf(w, `{"ssh_key":{"id":1}}`)
	})

	key, _, err := client.Keys.UpdateByFingerprint("3b:16:bf:e4:8b:00:8b:b8:59:8c:a9:d3:f0:19:45:fa", updateRequest)
	if err != nil {
		t.Errorf("Keys.Update returned error: %v", err)
	} else {
		if id := key.ID; id != 1 {
			t.Errorf("expected id '%d', received '%d'", 1, id)
		}
	}
}

func TestKeys_DestroyByID(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account/keys/12345", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Keys.DeleteByID(12345)
	if err != nil {
		t.Errorf("Keys.Delete returned error: %v", err)
	}
}

func TestKeys_DestroyByFingerprint(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/account/keys/aa:bb:cc", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Keys.DeleteByFingerprint("aa:bb:cc")
	if err != nil {
		t.Errorf("Keys.Delete returned error: %v", err)
	}
}

func TestKey_String(t *testing.T) {
	key := &Key{
		ID:          123,
		Name:        "Key",
		Fingerprint: "fingerprint",
		PublicKey:   "public key",
	}

	stringified := key.String()
	expected := `godo.Key{ID:123, Name:"Key", Fingerprint:"fingerprint", PublicKey:"public key"}`
	if expected != stringified {
		t.Errorf("Key.String returned %+v, expected %+v", stringified, expected)
	}
}
