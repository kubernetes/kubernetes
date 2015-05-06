package github

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestGitService_GetBlob(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/git/blobs/s", func(w http.ResponseWriter, r *http.Request) {
		if m := "GET"; m != r.Method {
			t.Errorf("Request method = %v, want %v", r.Method, m)
		}
		fmt.Fprint(w, `{
			  "sha": "s",
			  "content": "blob content"
			}`)
	})

	blob, _, err := client.Git.GetBlob("o", "r", "s")
	if err != nil {
		t.Errorf("Git.GetBlob returned error: %v", err)
	}

	want := Blob{
		SHA:     String("s"),
		Content: String("blob content"),
	}

	if !reflect.DeepEqual(*blob, want) {
		t.Errorf("Blob.Get returned %+v, want %+v", *blob, want)
	}
}

func TestGitService_GetBlob_invalidOwner(t *testing.T) {
	_, _, err := client.Git.GetBlob("%", "%", "%")
	testURLParseError(t, err)
}

func TestGitService_CreateBlob(t *testing.T) {
	setup()
	defer teardown()

	input := &Blob{
		SHA:      String("s"),
		Content:  String("blob content"),
		Encoding: String("utf-8"),
		Size:     Int(12),
	}

	mux.HandleFunc("/repos/o/r/git/blobs", func(w http.ResponseWriter, r *http.Request) {
		v := new(Blob)
		json.NewDecoder(r.Body).Decode(v)

		if m := "POST"; m != r.Method {
			t.Errorf("Request method = %v, want %v", r.Method, m)
		}

		want := input
		if !reflect.DeepEqual(v, want) {
			t.Errorf("Git.CreateBlob request body: %+v, want %+v", v, want)
		}

		fmt.Fprint(w, `{
		 "sha": "s",
		 "content": "blob content",
		 "encoding": "utf-8",
		 "size": 12
		}`)
	})

	blob, _, err := client.Git.CreateBlob("o", "r", input)
	if err != nil {
		t.Errorf("Git.CreateBlob returned error: %v", err)
	}

	want := input

	if !reflect.DeepEqual(*blob, *want) {
		t.Errorf("Git.CreateBlob returned %+v, want %+v", *blob, *want)
	}
}

func TestGitService_CreateBlob_invalidOwner(t *testing.T) {
	_, _, err := client.Git.CreateBlob("%", "%", &Blob{})
	testURLParseError(t, err)
}
