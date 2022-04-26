package storer

// Storer is a basic storer for encoded objects and references.
type Storer interface {
	EncodedObjectStorer
	ReferenceStorer
}

// Initializer should be implemented by storers that require to perform any
// operation when creating a new repository (i.e. git init).
type Initializer interface {
	// Init performs initialization of the storer and returns the error, if
	// any.
	Init() error
}
