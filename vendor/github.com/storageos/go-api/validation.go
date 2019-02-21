package storageos

import (
	"errors"
	"regexp"
)

const (
	// IDFormat are the characters allowed to represent an ID.
	IDFormat = `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`

	// NameFormat are the characters allowed to represent a name.
	NameFormat = `[a-zA-Z0-9][a-zA-Z0-9~_.-]+`
)

var (
	// IDPattern is a regular expression to validate a unique id against the
	// collection of restricted characters.
	IDPattern = regexp.MustCompile(`^` + IDFormat + `$`)

	// NamePattern is a regular expression to validate names against the
	// collection of restricted characters.
	NamePattern = regexp.MustCompile(`^` + NameFormat + `$`)

	// ErrNoRef is given when the reference given is invalid.
	ErrNoRef = errors.New("no ref provided or incorrect format")
	// ErrNoNamespace is given when the namespace given is invalid.
	ErrNoNamespace = errors.New("no namespace provided or incorrect format")
)

// ValidateNamespaceAndRef returns true if both the namespace and ref are valid.
func ValidateNamespaceAndRef(namespace, ref string) error {
	if !IsUUID(ref) && !IsName(ref) {
		return ErrNoRef
	}
	if !IsName(namespace) {
		return ErrNoNamespace
	}
	return nil
}

// ValidateNamespace returns true if the namespace uses a valid name.
func ValidateNamespace(namespace string) error {
	if !IsName(namespace) {
		return ErrNoNamespace
	}
	return nil
}

// IsUUID returns true if the string input is a valid UUID string.
func IsUUID(s string) bool {
	return IDPattern.MatchString(s)
}

// IsName returns true if the string input is a valid Name string.
func IsName(s string) bool {
	return NamePattern.MatchString(s)
}

// namespacedPath checks for valid input and returns api path for a namespaced
// objectType.  Use namespacedRefPath for objects.
func namespacedPath(namespace, objectType string) (string, error) {
	if err := ValidateNamespace(namespace); err != nil {
		return "", err
	}
	return "/namespaces/" + namespace + "/" + objectType, nil
}

// namespacedRefPath checks for valid input and returns api path for a single
// namespaced object.  Use namespacedPath for objects type path.
func namespacedRefPath(namespace, objectType, ref string) (string, error) {
	if err := ValidateNamespaceAndRef(namespace, ref); err != nil {
		return "", err
	}
	return "/namespaces/" + namespace + "/" + objectType + "/" + ref, nil
}
