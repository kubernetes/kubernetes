// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestGistsService_List_specifiedUser(t *testing.T) {
	setup()
	defer teardown()

	since := "2013-01-01T00:00:00Z"

	mux.HandleFunc("/users/u/gists", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"since": since,
		})
		fmt.Fprint(w, `[{"id": "1"}]`)
	})

	opt := &GistListOptions{Since: time.Date(2013, time.January, 1, 0, 0, 0, 0, time.UTC)}
	gists, _, err := client.Gists.List("u", opt)

	if err != nil {
		t.Errorf("Gists.List returned error: %v", err)
	}

	want := []Gist{{ID: String("1")}}
	if !reflect.DeepEqual(gists, want) {
		t.Errorf("Gists.List returned %+v, want %+v", gists, want)
	}
}

func TestGistsService_List_authenticatedUser(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `[{"id": "1"}]`)
	})

	gists, _, err := client.Gists.List("", nil)
	if err != nil {
		t.Errorf("Gists.List returned error: %v", err)
	}

	want := []Gist{{ID: String("1")}}
	if !reflect.DeepEqual(gists, want) {
		t.Errorf("Gists.List returned %+v, want %+v", gists, want)
	}
}

func TestGistsService_List_invalidUser(t *testing.T) {
	_, _, err := client.Gists.List("%", nil)
	testURLParseError(t, err)
}

func TestGistsService_ListAll(t *testing.T) {
	setup()
	defer teardown()

	since := "2013-01-01T00:00:00Z"

	mux.HandleFunc("/gists/public", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"since": since,
		})
		fmt.Fprint(w, `[{"id": "1"}]`)
	})

	opt := &GistListOptions{Since: time.Date(2013, time.January, 1, 0, 0, 0, 0, time.UTC)}
	gists, _, err := client.Gists.ListAll(opt)

	if err != nil {
		t.Errorf("Gists.ListAll returned error: %v", err)
	}

	want := []Gist{{ID: String("1")}}
	if !reflect.DeepEqual(gists, want) {
		t.Errorf("Gists.ListAll returned %+v, want %+v", gists, want)
	}
}

func TestGistsService_ListStarred(t *testing.T) {
	setup()
	defer teardown()

	since := "2013-01-01T00:00:00Z"

	mux.HandleFunc("/gists/starred", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"since": since,
		})
		fmt.Fprint(w, `[{"id": "1"}]`)
	})

	opt := &GistListOptions{Since: time.Date(2013, time.January, 1, 0, 0, 0, 0, time.UTC)}
	gists, _, err := client.Gists.ListStarred(opt)

	if err != nil {
		t.Errorf("Gists.ListStarred returned error: %v", err)
	}

	want := []Gist{{ID: String("1")}}
	if !reflect.DeepEqual(gists, want) {
		t.Errorf("Gists.ListStarred returned %+v, want %+v", gists, want)
	}
}

func TestGistsService_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id": "1"}`)
	})

	gist, _, err := client.Gists.Get("1")

	if err != nil {
		t.Errorf("Gists.Get returned error: %v", err)
	}

	want := &Gist{ID: String("1")}
	if !reflect.DeepEqual(gist, want) {
		t.Errorf("Gists.Get returned %+v, want %+v", gist, want)
	}
}

func TestGistsService_Get_invalidID(t *testing.T) {
	_, _, err := client.Gists.Get("%")
	testURLParseError(t, err)
}

func TestGistsService_GetRevision(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/s", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, `{"id": "1"}`)
	})

	gist, _, err := client.Gists.GetRevision("1", "s")

	if err != nil {
		t.Errorf("Gists.Get returned error: %v", err)
	}

	want := &Gist{ID: String("1")}
	if !reflect.DeepEqual(gist, want) {
		t.Errorf("Gists.Get returned %+v, want %+v", gist, want)
	}
}

func TestGistsService_GetRevision_invalidID(t *testing.T) {
	_, _, err := client.Gists.GetRevision("%", "%")
	testURLParseError(t, err)
}

func TestGistsService_Create(t *testing.T) {
	setup()
	defer teardown()

	input := &Gist{
		Description: String("Gist description"),
		Public:      Bool(false),
		Files: map[GistFilename]GistFile{
			"test.txt": {Content: String("Gist file content")},
		},
	}

	mux.HandleFunc("/gists", func(w http.ResponseWriter, r *http.Request) {
		v := new(Gist)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w,
			`
			{
				"id": "1",
				"description": "Gist description",
				"public": false,
				"files": {
					"test.txt": {
						"filename": "test.txt"
					}
				}
			}`)
	})

	gist, _, err := client.Gists.Create(input)
	if err != nil {
		t.Errorf("Gists.Create returned error: %v", err)
	}

	want := &Gist{
		ID:          String("1"),
		Description: String("Gist description"),
		Public:      Bool(false),
		Files: map[GistFilename]GistFile{
			"test.txt": {Filename: String("test.txt")},
		},
	}
	if !reflect.DeepEqual(gist, want) {
		t.Errorf("Gists.Create returned %+v, want %+v", gist, want)
	}
}

func TestGistsService_Edit(t *testing.T) {
	setup()
	defer teardown()

	input := &Gist{
		Description: String("New description"),
		Files: map[GistFilename]GistFile{
			"new.txt": {Content: String("new file content")},
		},
	}

	mux.HandleFunc("/gists/1", func(w http.ResponseWriter, r *http.Request) {
		v := new(Gist)
		json.NewDecoder(r.Body).Decode(v)

		testMethod(t, r, "PATCH")
		if !reflect.DeepEqual(v, input) {
			t.Errorf("Request body = %+v, want %+v", v, input)
		}

		fmt.Fprint(w,
			`
			{
				"id": "1",
				"description": "new description",
				"public": false,
				"files": {
					"test.txt": {
						"filename": "test.txt"
					},
					"new.txt": {
						"filename": "new.txt"
					}
				}
			}`)
	})

	gist, _, err := client.Gists.Edit("1", input)
	if err != nil {
		t.Errorf("Gists.Edit returned error: %v", err)
	}

	want := &Gist{
		ID:          String("1"),
		Description: String("new description"),
		Public:      Bool(false),
		Files: map[GistFilename]GistFile{
			"test.txt": {Filename: String("test.txt")},
			"new.txt":  {Filename: String("new.txt")},
		},
	}
	if !reflect.DeepEqual(gist, want) {
		t.Errorf("Gists.Edit returned %+v, want %+v", gist, want)
	}
}

func TestGistsService_Edit_invalidID(t *testing.T) {
	_, _, err := client.Gists.Edit("%", nil)
	testURLParseError(t, err)
}

func TestGistsService_Delete(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Gists.Delete("1")
	if err != nil {
		t.Errorf("Gists.Delete returned error: %v", err)
	}
}

func TestGistsService_Delete_invalidID(t *testing.T) {
	_, err := client.Gists.Delete("%")
	testURLParseError(t, err)
}

func TestGistsService_Star(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/star", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "PUT")
	})

	_, err := client.Gists.Star("1")
	if err != nil {
		t.Errorf("Gists.Star returned error: %v", err)
	}
}

func TestGistsService_Star_invalidID(t *testing.T) {
	_, err := client.Gists.Star("%")
	testURLParseError(t, err)
}

func TestGistsService_Unstar(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/star", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Gists.Unstar("1")
	if err != nil {
		t.Errorf("Gists.Unstar returned error: %v", err)
	}
}

func TestGistsService_Unstar_invalidID(t *testing.T) {
	_, err := client.Gists.Unstar("%")
	testURLParseError(t, err)
}

func TestGistsService_IsStarred_hasStar(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/star", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNoContent)
	})

	star, _, err := client.Gists.IsStarred("1")
	if err != nil {
		t.Errorf("Gists.Starred returned error: %v", err)
	}
	if want := true; star != want {
		t.Errorf("Gists.Starred returned %+v, want %+v", star, want)
	}
}

func TestGistsService_IsStarred_noStar(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/star", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		w.WriteHeader(http.StatusNotFound)
	})

	star, _, err := client.Gists.IsStarred("1")
	if err != nil {
		t.Errorf("Gists.Starred returned error: %v", err)
	}
	if want := false; star != want {
		t.Errorf("Gists.Starred returned %+v, want %+v", star, want)
	}
}

func TestGistsService_IsStarred_invalidID(t *testing.T) {
	_, _, err := client.Gists.IsStarred("%")
	testURLParseError(t, err)
}

func TestGistsService_Fork(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/gists/1/forks", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "POST")
		fmt.Fprint(w, `{"id": "2"}`)
	})

	gist, _, err := client.Gists.Fork("1")

	if err != nil {
		t.Errorf("Gists.Fork returned error: %v", err)
	}

	want := &Gist{ID: String("2")}
	if !reflect.DeepEqual(gist, want) {
		t.Errorf("Gists.Fork returned %+v, want %+v", gist, want)
	}
}

func TestGistsService_Fork_invalidID(t *testing.T) {
	_, _, err := client.Gists.Fork("%")
	testURLParseError(t, err)
}
