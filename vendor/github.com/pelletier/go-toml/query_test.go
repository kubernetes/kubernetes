package toml

import (
	"testing"
)

func assertArrayContainsInAnyOrder(t *testing.T, array []interface{}, objects ...interface{}) {
	if len(array) != len(objects) {
		t.Fatalf("array contains %d objects but %d are expected", len(array), len(objects))
	}

	for _, o := range objects {
		found := false
		for _, a := range array {
			if a == o {
				found = true
				break
			}
		}
		if !found {
			t.Fatal(o, "not found in array", array)
		}
	}
}

func TestQueryExample(t *testing.T) {
	config, _ := Load(`
      [[book]]
      title = "The Stand"
      author = "Stephen King"
      [[book]]
      title = "For Whom the Bell Tolls"
      author = "Ernest Hemmingway"
      [[book]]
      title = "Neuromancer"
      author = "William Gibson"
    `)

	authors, _ := config.Query("$.book.author")
	names := authors.Values()
	if len(names) != 3 {
		t.Fatalf("query should return 3 names but returned %d", len(names))
	}
	assertArrayContainsInAnyOrder(t, names, "Stephen King", "Ernest Hemmingway", "William Gibson")
}

func TestQueryReadmeExample(t *testing.T) {
	config, _ := Load(`
[postgres]
user = "pelletier"
password = "mypassword"
`)
	results, _ := config.Query("$..[user,password]")
	values := results.Values()
	if len(values) != 2 {
		t.Fatalf("query should return 2 values but returned %d", len(values))
	}
	assertArrayContainsInAnyOrder(t, values, "pelletier", "mypassword")
}

func TestQueryPathNotPresent(t *testing.T) {
	config, _ := Load(`a = "hello"`)
	results, err := config.Query("$.foo.bar")
	if err != nil {
		t.Fatalf("err should be nil. got %s instead", err)
	}
	if len(results.items) != 0 {
		t.Fatalf("no items should be matched. %d matched instead", len(results.items))
	}
}
