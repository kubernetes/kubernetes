package storage

import (
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/storage/driver"
)

// Exists provides a utility method to test whether or not a path exists in
// the given driver.
func exists(ctx context.Context, drv driver.StorageDriver, path string) (bool, error) {
	if _, err := drv.Stat(ctx, path); err != nil {
		switch err := err.(type) {
		case driver.PathNotFoundError:
			return false, nil
		default:
			return false, err
		}
	}

	return true, nil
}
