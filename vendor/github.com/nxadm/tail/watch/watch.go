// Copyright (c) 2019 FOSS contributors of https://github.com/nxadm/tail
// Copyright (c) 2015 HPE Software Inc. All rights reserved.
// Copyright (c) 2013 ActiveState Software Inc. All rights reserved.

package watch

import "gopkg.in/tomb.v1"

// FileWatcher monitors file-level events.
type FileWatcher interface {
	// BlockUntilExists blocks until the file comes into existence.
	BlockUntilExists(*tomb.Tomb) error

	// ChangeEvents reports on changes to a file, be it modification,
	// deletion, renames or truncations. Returned FileChanges group of
	// channels will be closed, thus become unusable, after a deletion
	// or truncation event.
	// In order to properly report truncations, ChangeEvents requires
	// the caller to pass their current offset in the file.
	ChangeEvents(*tomb.Tomb, int64) (*FileChanges, error)
}
