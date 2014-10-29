package pagination

// nullPage is an always-empty page that trivially satisfies all Page interfacts.
// It's useful to be returned along with an error.
type nullPage struct{}

// NextPageURL always returns "" to indicate that there are no more pages to return.
func (p nullPage) NextPageURL() (string, error) {
	return "", nil
}

// IsEmpty always returns true to prevent iteration over nullPages.
func (p nullPage) IsEmpty() (bool, error) {
	return true, nil
}

// LastMark always returns "" because the nullPage contains no items to have a mark.
func (p nullPage) LastMark() (string, error) {
	return "", nil
}
