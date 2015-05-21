package github

import (
	"fmt"
	"net/http"
	"reflect"

	"testing"
)

func TestSearchService_Repositories(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/search/repositories", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"q":        "blah",
			"sort":     "forks",
			"order":    "desc",
			"page":     "2",
			"per_page": "2",
		})

		fmt.Fprint(w, `{"total_count": 4, "items": [{"id":1},{"id":2}]}`)
	})

	opts := &SearchOptions{Sort: "forks", Order: "desc", ListOptions: ListOptions{Page: 2, PerPage: 2}}
	result, _, err := client.Search.Repositories("blah", opts)
	if err != nil {
		t.Errorf("Search.Repositories returned error: %v", err)
	}

	want := &RepositoriesSearchResult{
		Total:        Int(4),
		Repositories: []Repository{{ID: Int(1)}, {ID: Int(2)}},
	}
	if !reflect.DeepEqual(result, want) {
		t.Errorf("Search.Repositories returned %+v, want %+v", result, want)
	}
}

func TestSearchService_Issues(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/search/issues", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"q":        "blah",
			"sort":     "forks",
			"order":    "desc",
			"page":     "2",
			"per_page": "2",
		})

		fmt.Fprint(w, `{"total_count": 4, "items": [{"number":1},{"number":2}]}`)
	})

	opts := &SearchOptions{Sort: "forks", Order: "desc", ListOptions: ListOptions{Page: 2, PerPage: 2}}
	result, _, err := client.Search.Issues("blah", opts)
	if err != nil {
		t.Errorf("Search.Issues returned error: %v", err)
	}

	want := &IssuesSearchResult{
		Total:  Int(4),
		Issues: []Issue{{Number: Int(1)}, {Number: Int(2)}},
	}
	if !reflect.DeepEqual(result, want) {
		t.Errorf("Search.Issues returned %+v, want %+v", result, want)
	}
}

func TestSearchService_Users(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/search/users", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"q":        "blah",
			"sort":     "forks",
			"order":    "desc",
			"page":     "2",
			"per_page": "2",
		})

		fmt.Fprint(w, `{"total_count": 4, "items": [{"id":1},{"id":2}]}`)
	})

	opts := &SearchOptions{Sort: "forks", Order: "desc", ListOptions: ListOptions{Page: 2, PerPage: 2}}
	result, _, err := client.Search.Users("blah", opts)
	if err != nil {
		t.Errorf("Search.Issues returned error: %v", err)
	}

	want := &UsersSearchResult{
		Total: Int(4),
		Users: []User{{ID: Int(1)}, {ID: Int(2)}},
	}
	if !reflect.DeepEqual(result, want) {
		t.Errorf("Search.Users returned %+v, want %+v", result, want)
	}
}

func TestSearchService_Code(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/search/code", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r, values{
			"q":        "blah",
			"sort":     "forks",
			"order":    "desc",
			"page":     "2",
			"per_page": "2",
		})

		fmt.Fprint(w, `{"total_count": 4, "items": [{"name":"1"},{"name":"2"}]}`)
	})

	opts := &SearchOptions{Sort: "forks", Order: "desc", ListOptions: ListOptions{Page: 2, PerPage: 2}}
	result, _, err := client.Search.Code("blah", opts)
	if err != nil {
		t.Errorf("Search.Code returned error: %v", err)
	}

	want := &CodeSearchResult{
		Total:       Int(4),
		CodeResults: []CodeResult{{Name: String("1")}, {Name: String("2")}},
	}
	if !reflect.DeepEqual(result, want) {
		t.Errorf("Search.Code returned %+v, want %+v", result, want)
	}
}

func TestSearchService_CodeTextMatch(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/search/code", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")

		textMatchResponse := `
		{
			"total_count": 1,
			"items": [
				{
					"name":"gopher1",
					"text_matches": [
						{
							"fragment": "I'm afraid my friend what you have found\nIs a gopher who lives to feed",
							"matches": [
								{
									"text": "gopher",
									"indices": [
										14,
										21
							  	]
								}
						  ]
					  }
				  ]
				}
			]
		}
    `

		fmt.Fprint(w, textMatchResponse)
	})

	opts := &SearchOptions{Sort: "forks", Order: "desc", ListOptions: ListOptions{Page: 2, PerPage: 2}, TextMatch: true}
	result, _, err := client.Search.Code("blah", opts)
	if err != nil {
		t.Errorf("Search.Code returned error: %v", err)
	}

	wantedCodeResult := CodeResult{
		Name: String("gopher1"),
		TextMatches: []TextMatch{{
			Fragment: String("I'm afraid my friend what you have found\nIs a gopher who lives to feed"),
			Matches:  []Match{{Text: String("gopher"), Indices: []int{14, 21}}},
		},
		},
	}

	want := &CodeSearchResult{
		Total:       Int(1),
		CodeResults: []CodeResult{wantedCodeResult},
	}
	if !reflect.DeepEqual(result, want) {
		t.Errorf("Search.Code returned %+v, want %+v", result, want)
	}
}
