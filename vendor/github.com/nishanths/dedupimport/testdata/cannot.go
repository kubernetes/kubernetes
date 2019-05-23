package pkg

import (
	u "net/url"
	"net/url"
)

var google = url.QueryEscape("https://google.com/?q=something")

func fetch(url string) {
	_, _ = u.Parse(url)
	// ... other stuff ...
}