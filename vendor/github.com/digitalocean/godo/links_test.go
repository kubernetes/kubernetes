package godo

import (
	"encoding/json"
	"testing"
)

var (
	firstPageLinksJSONBlob = []byte(`{
		"links": {
			"pages": {
				"last": "https://api.digitalocean.com/v2/droplets/?page=3",
				"next": "https://api.digitalocean.com/v2/droplets/?page=2"
			}
		}
	}`)
	otherPageLinksJSONBlob = []byte(`{
		"links": {
			"pages": {
				"first": "https://api.digitalocean.com/v2/droplets/?page=1",
				"prev": "https://api.digitalocean.com/v2/droplets/?page=1",
				"last": "https://api.digitalocean.com/v2/droplets/?page=3",
				"next": "https://api.digitalocean.com/v2/droplets/?page=3"
			}
		}
	}`)
	lastPageLinksJSONBlob = []byte(`{
		"links": {
			"pages": {
				"first": "https://api.digitalocean.com/v2/droplets/?page=1",
				"prev": "https://api.digitalocean.com/v2/droplets/?page=2"
			}
		}
	}`)

	missingLinksJSONBlob = []byte(`{ }`)
)

type godoList struct {
	Links Links `json:"links"`
}

func loadLinksJSON(t *testing.T, j []byte) Links {
	var list godoList
	err := json.Unmarshal(j, &list)
	if err != nil {
		t.Fatal(err)
	}

	return list.Links
}

func TestLinks_ParseFirst(t *testing.T) {
	links := loadLinksJSON(t, firstPageLinksJSONBlob)
	_, err := links.CurrentPage()
	if err != nil {
		t.Fatal(err)
	}

	r := &Response{Links: &links}
	checkCurrentPage(t, r, 1)

	if links.IsLastPage() {
		t.Fatalf("shouldn't be last page")
	}
}

func TestLinks_ParseMiddle(t *testing.T) {
	links := loadLinksJSON(t, otherPageLinksJSONBlob)
	_, err := links.CurrentPage()
	if err != nil {
		t.Fatal(err)
	}

	r := &Response{Links: &links}
	checkCurrentPage(t, r, 2)

	if links.IsLastPage() {
		t.Fatalf("shouldn't be last page")
	}
}

func TestLinks_ParseLast(t *testing.T) {
	links := loadLinksJSON(t, lastPageLinksJSONBlob)
	_, err := links.CurrentPage()
	if err != nil {
		t.Fatal(err)
	}

	r := &Response{Links: &links}
	checkCurrentPage(t, r, 3)
	if !links.IsLastPage() {
		t.Fatalf("expected last page")
	}
}

func TestLinks_ParseMissing(t *testing.T) {
	links := loadLinksJSON(t, missingLinksJSONBlob)
	_, err := links.CurrentPage()
	if err != nil {
		t.Fatal(err)
	}

	r := &Response{Links: &links}
	checkCurrentPage(t, r, 1)
}

func TestLinks_ParseURL(t *testing.T) {
	type linkTest struct {
		name, url string
		expected  int
	}

	linkTests := []linkTest{
		{
			name:     "prev",
			url:      "https://api.digitalocean.com/v2/droplets/?page=1",
			expected: 1,
		},
		{
			name:     "last",
			url:      "https://api.digitalocean.com/v2/droplets/?page=5",
			expected: 5,
		},
		{
			name:     "nexta",
			url:      "https://api.digitalocean.com/v2/droplets/?page=2",
			expected: 2,
		},
	}

	for _, lT := range linkTests {
		p, err := pageForURL(lT.url)
		if err != nil {
			t.Fatal(err)
		}

		if p != lT.expected {
			t.Errorf("expected page for '%s' to be '%d', was '%d'",
				lT.url, lT.expected, p)
		}
	}

}

func TestLinks_ParseEmptyString(t *testing.T) {
	type linkTest struct {
		name, url string
		expected  int
	}

	linkTests := []linkTest{
		{
			name:     "none",
			url:      "http://example.com",
			expected: 0,
		},
		{
			name:     "bad",
			url:      "no url",
			expected: 0,
		},
		{
			name:     "empty",
			url:      "",
			expected: 0,
		},
	}

	for _, lT := range linkTests {
		_, err := pageForURL(lT.url)
		if err == nil {
			t.Fatalf("expected error for test '%s', but received none", lT.name)
		}
	}
}
