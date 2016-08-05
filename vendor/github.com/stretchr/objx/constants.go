package objx

const (
	// PathSeparator is the character used to separate the elements
	// of the keypath.
	//
	// For example, `location.address.city`
	PathSeparator string = "."

	// SignatureSeparator is the character that is used to
	// separate the Base64 string from the security signature.
	SignatureSeparator = "_"
)
