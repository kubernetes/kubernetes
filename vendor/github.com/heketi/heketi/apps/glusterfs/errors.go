//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package glusterfs

import (
	"errors"
)

var (
	ErrNoSpace          = errors.New("No space")
	ErrNotFound         = errors.New("Id not found")
	ErrConflict         = errors.New("The target exists, contains other items, or is in use.")
	ErrMaxBricks        = errors.New("Maximum number of bricks reached.")
	ErrMinimumBrickSize = errors.New("Minimum brick size limit reached.  Out of space.")
	ErrDbAccess         = errors.New("Unable to access db")
	ErrAccessList       = errors.New("Unable to access list")
	ErrKeyExists        = errors.New("Key already exists in the database")
	ErrNoReplacement    = errors.New("No Replacement was found for resource requested to be removed")
)
