package pagination

// SinglePageBase may be embedded in a Page that contains all of the results from an operation at once.
type SinglePageBase PageResult

// NextPageURL always returns "" to indicate that there are no more pages to return.
func (current SinglePageBase) NextPageURL() (string, error) {
	return "", nil
}

// GetBody returns the single page's body. This method is needed to satisfy the
// Page interface.
func (current SinglePageBase) GetBody() interface{} {
	return current.Body
}
