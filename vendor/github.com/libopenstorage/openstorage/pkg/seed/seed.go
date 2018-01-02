package seed

import (
	"errors"
	"net/url"
)

// Source defines the interface for keep track of volume driver mounts.
type Source interface {
	// String representation of this source
	String() string
	// Load from URI into dest.
	Load(dest string) error
	// Metadata for this source.
	MetadataRead(mdDir string) (string, error)
	// MetadataWrite for this source.
	MetadataWrite(mdDir string) error
}

var (
	// ErrUnsupported is returned for an unsupported seed source.
	ErrUnsupported = errors.New("Not supported")
)

// New returns a new instance of Source
func New(uri string, options map[string]string) (Source, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	switch u.Scheme {
	case "github":
		return NewGitSource(uri, options)
	}
	return nil, ErrUnsupported
}
