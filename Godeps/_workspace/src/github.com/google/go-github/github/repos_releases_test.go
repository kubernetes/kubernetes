// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"testing"
)

func TestRepositoriesService_ListReleases(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	releases, _, err := client.Repositories.ListReleases("o", "r", opt)
	if err != nil {
		t.Errorf("Repositories.ListReleases returned error: %v", err)
	}
	want := []RepositoryRelease{{ID: Int(1)}}
	if !reflect.DeepEqual(releases, want) {
		t.Errorf("Repositories.ListReleases returned %+v, want %+v", releases, want)
	}
}

func TestRepositoriesService_GetRelease(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	release, resp, err := client.Repositories.GetRelease("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.GetRelease returned error: %v\n%v", err, resp.Body)
	}

	want := &RepositoryRelease{ID: Int(1)}
	if !reflect.DeepEqual(release, want) {
		t.Errorf("Repositories.GetRelease returned %+v, want %+v", release, want)
	}
}

func TestRepositoriesService_CreateRelease(t *testing.T) {
	setup()
	defer teardown()

	input := &RepositoryRelease{Name: String("v1.0")}

	mux.HandleFunc("/repos/o/r/releases", func(w http.ResponseWriter, r *http.Request) {
		v := new(RepositoryRelease)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}
		fmt.Fprint(w, `{"id":1}`)
	})

	release, _, err := client.Repositories.CreateRelease("o", "r", input)
	if err != nil {
		t.Errorf("Repositories.CreateRelease returned error: %v", err)
	}

	want := &RepositoryRelease{ID: Int(1)}
	if !reflect.DeepEqual(release, want) {
		t.Errorf("Repositories.CreateRelease returned %+v, want %+v", release, want)
	}
}

func TestRepositoriesService_EditRelease(t *testing.T) {
	setup()
	defer teardown()

	input := &RepositoryRelease{Name: String("n")}

	mux.HandleFunc("/repos/o/r/releases/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(RepositoryRelease)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}
		fmt.Fprint(w, `{"id":1}`)
	})

	release, _, err := client.Repositories.EditRelease("o", "r", 1, input)
	if err != nil {
		t.Errorf("Repositories.EditRelease returned error: %v", err)
	}
	want := &RepositoryRelease{ID: Int(1)}
	if !reflect.DeepEqual(release, want) {
		t.Errorf("Repositories.EditRelease returned = %+v, want %+v", release, want)
	}
}

func TestRepositoriesService_DeleteRelease(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Repositories.DeleteRelease("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.DeleteRelease returned error: %v", err)
	}
}

func TestRepositoriesService_ListReleaseAssets(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases/1/assets", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{"page": "2"})
		fmt.Fprint(w, `[{"id":1}]`)
	})

	opt := &ListOptions{Page: 2}
	assets, _, err := client.Repositories.ListReleaseAssets("o", "r", 1, opt)
	if err != nil {
		t.Errorf("Repositories.ListReleaseAssets returned error: %v", err)
	}
	want := []ReleaseAsset{{ID: Int(1)}}
	if !reflect.DeepEqual(assets, want) {
		t.Errorf("Repositories.ListReleaseAssets returned %+v, want %+v", assets, want)
	}
}

func TestRepositoriesService_GetReleaseAsset(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases/assets/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id":1}`)
	})

	asset, _, err := client.Repositories.GetReleaseAsset("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.GetReleaseAsset returned error: %v", err)
	}
	want := &ReleaseAsset{ID: Int(1)}
	if !reflect.DeepEqual(asset, want) {
		t.Errorf("Repositories.GetReleaseAsset returned %+v, want %+v", asset, want)
	}
}

func TestRepositoriesService_EditReleaseAsset(t *testing.T) {
	setup()
	defer teardown()

	input := &ReleaseAsset{Name: String("n")}

	mux.HandleFunc("/repos/o/r/releases/assets/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(ReleaseAsset)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}
		fmt.Fprint(w, `{"id":1}`)
	})

	asset, _, err := client.Repositories.EditReleaseAsset("o", "r", 1, input)
	if err != nil {
		t.Errorf("Repositories.EditReleaseAsset returned error: %v", err)
	}
	want := &ReleaseAsset{ID: Int(1)}
	if !reflect.DeepEqual(asset, want) {
		t.Errorf("Repositories.EditReleaseAsset returned = %+v, want %+v", asset, want)
	}
}

func TestRepositoriesService_DeleteReleaseAsset(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases/assets/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Repositories.DeleteReleaseAsset("o", "r", 1)
	if err != nil {
		t.Errorf("Repositories.DeleteReleaseAsset returned error: %v", err)
	}
}

func TestRepositoriesService_UploadReleaseAsset(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/releases/1/assets", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "POST")
		testHeader(t, r, "Content-Type", "text/plain; charset=utf-8")
		testHeader(t, r, "Content-Length", "12")
		testFormValues(t, r, values{"name": "n"})
		testBody(t, r, "Upload me !\n")

		fmt.Fprintf(w, `{"id":1}`)
	})

	file, dir, err := openTestFile("upload.txt", "Upload me !\n")
	if err != nil {
		t.Fatalf("Unable to create temp file: %v", err)
	}
	defer os.RemoveAll(dir)

	opt := &UploadOptions{Name: "n"}
	asset, _, err := client.Repositories.UploadReleaseAsset("o", "r", 1, opt, file)
	if err != nil {
		t.Errorf("Repositories.UploadReleaseAssert returned error: %v", err)
	}
	want := &ReleaseAsset{ID: Int(1)}
	if !reflect.DeepEqual(asset, want) {
		t.Errorf("Repositories.UploadReleaseAssert returned %+v, want %+v", asset, want)
	}
}
