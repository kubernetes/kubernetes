package serror

//go:generate stringer -type=StorageOSErrorKind error_kind.go
type StorageOSErrorKind int

// Known error kinds
const (
	UnknownError StorageOSErrorKind = iota
	APIUncontactable
	InvalidHostConfig
)
