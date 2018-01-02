package labels

import (
	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

const (
	maxSize = 4096
)

// Validate a label's key and value are under 4096 bytes
func Validate(k, v string) error {
	if (len(k) + len(v)) > maxSize {
		if len(k) > 10 {
			k = k[:10]
		}
		return errors.Wrapf(errdefs.ErrInvalidArgument, "label key and value greater than maximum size (%d bytes), key: %s", maxSize, k)
	}
	return nil
}
