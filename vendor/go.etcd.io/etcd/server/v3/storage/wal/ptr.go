package wal

// toPtr returns a pointer to the given value.
// TODO: remove after upgrading to Go 1.26 which supports new(expr).
func toPtr[T any](v T) *T { return &v }
