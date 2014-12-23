package pagination

import (
	"errors"

	"github.com/rackspace/gophercloud"
)

var (
	// ErrPageNotAvailable is returned from a Pager when a next or previous page is requested, but does not exist.
	ErrPageNotAvailable = errors.New("The requested page does not exist.")
)

// Page must be satisfied by the result type of any resource collection.
// It allows clients to interact with the resource uniformly, regardless of whether or not or how it's paginated.
// Generally, rather than implementing this interface directly, implementors should embed one of the concrete PageBase structs,
// instead.
// Depending on the pagination strategy of a particular resource, there may be an additional subinterface that the result type
// will need to implement.
type Page interface {

	// NextPageURL generates the URL for the page of data that follows this collection.
	// Return "" if no such page exists.
	NextPageURL() (string, error)

	// IsEmpty returns true if this Page has no items in it.
	IsEmpty() (bool, error)
}

// Pager knows how to advance through a specific resource collection, one page at a time.
type Pager struct {
	client *gophercloud.ServiceClient

	initialURL string

	createPage func(r PageResult) Page

	Err error

	// Headers supplies additional HTTP headers to populate on each paged request.
	Headers map[string]string
}

// NewPager constructs a manually-configured pager.
// Supply the URL for the first page, a function that requests a specific page given a URL, and a function that counts a page.
func NewPager(client *gophercloud.ServiceClient, initialURL string, createPage func(r PageResult) Page) Pager {
	return Pager{
		client:     client,
		initialURL: initialURL,
		createPage: createPage,
	}
}

// WithPageCreator returns a new Pager that substitutes a different page creation function. This is
// useful for overriding List functions in delegation.
func (p Pager) WithPageCreator(createPage func(r PageResult) Page) Pager {
	return Pager{
		client:     p.client,
		initialURL: p.initialURL,
		createPage: createPage,
	}
}

func (p Pager) fetchNextPage(url string) (Page, error) {
	resp, err := Request(p.client, p.Headers, url)
	if err != nil {
		return nil, err
	}

	remembered, err := PageResultFrom(resp)
	if err != nil {
		return nil, err
	}

	return p.createPage(remembered), nil
}

// EachPage iterates over each page returned by a Pager, yielding one at a time to a handler function.
// Return "false" from the handler to prematurely stop iterating.
func (p Pager) EachPage(handler func(Page) (bool, error)) error {
	if p.Err != nil {
		return p.Err
	}
	currentURL := p.initialURL
	for {
		currentPage, err := p.fetchNextPage(currentURL)
		if err != nil {
			return err
		}

		empty, err := currentPage.IsEmpty()
		if err != nil {
			return err
		}
		if empty {
			return nil
		}

		ok, err := handler(currentPage)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}

		currentURL, err = currentPage.NextPageURL()
		if err != nil {
			return err
		}
		if currentURL == "" {
			return nil
		}
	}
}
