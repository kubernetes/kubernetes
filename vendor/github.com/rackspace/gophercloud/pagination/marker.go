package pagination

// MarkerPage is a stricter Page interface that describes additional functionality required for use with NewMarkerPager.
// For convenience, embed the MarkedPageBase struct.
type MarkerPage interface {
	Page

	// LastMarker returns the last "marker" value on this page.
	LastMarker() (string, error)
}

// MarkerPageBase is a page in a collection that's paginated by "limit" and "marker" query parameters.
type MarkerPageBase struct {
	PageResult

	// Owner is a reference to the embedding struct.
	Owner MarkerPage
}

// NextPageURL generates the URL for the page of results after this one.
func (current MarkerPageBase) NextPageURL() (string, error) {
	currentURL := current.URL

	mark, err := current.Owner.LastMarker()
	if err != nil {
		return "", err
	}

	q := currentURL.Query()
	q.Set("marker", mark)
	currentURL.RawQuery = q.Encode()

	return currentURL.String(), nil
}

// GetBody returns the linked page's body. This method is needed to satisfy the
// Page interface.
func (current MarkerPageBase) GetBody() interface{} {
	return current.Body
}
