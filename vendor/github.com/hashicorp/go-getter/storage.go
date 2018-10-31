package getter

// Storage is an interface that knows how to lookup downloaded directories
// as well as download and update directories from their sources into the
// proper location.
type Storage interface {
	// Dir returns the directory on local disk where the directory source
	// can be loaded from.
	Dir(string) (string, bool, error)

	// Get will download and optionally update the given directory.
	Get(string, string, bool) error
}
