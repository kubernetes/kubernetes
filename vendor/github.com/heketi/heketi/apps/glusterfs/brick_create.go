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
	"github.com/boltdb/bolt"
	"github.com/heketi/heketi/executors"
	"github.com/heketi/heketi/pkg/utils"
)

type CreateType int

const (
	CREATOR_CREATE CreateType = iota
	CREATOR_DESTROY
)

func createDestroyConcurrently(db *bolt.DB,
	executor executors.Executor,
	brick_entries []*BrickEntry,
	create_type CreateType) error {

	sg := utils.NewStatusGroup()

	// Create a goroutine for each brick
	for _, brick := range brick_entries {
		sg.Add(1)
		go func(b *BrickEntry) {
			defer sg.Done()
			if create_type == CREATOR_CREATE {
				sg.Err(b.Create(db, executor))
			} else {
				sg.Err(b.Destroy(db, executor))
			}
		}(brick)
	}

	// Wait here until all goroutines have returned.  If
	// any of errored, it would be cought here
	err := sg.Result()
	if err != nil {
		logger.Err(err)

		// Destroy all bricks and cleanup
		if create_type == CREATOR_CREATE {
			createDestroyConcurrently(db, executor, brick_entries, CREATOR_DESTROY)
		}
	}
	return err
}

func CreateBricks(db *bolt.DB, executor executors.Executor, brick_entries []*BrickEntry) error {
	return createDestroyConcurrently(db, executor, brick_entries, CREATOR_CREATE)
}

func DestroyBricks(db *bolt.DB, executor executors.Executor, brick_entries []*BrickEntry) error {
	return createDestroyConcurrently(db, executor, brick_entries, CREATOR_DESTROY)
}
