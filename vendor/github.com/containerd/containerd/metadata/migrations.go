package metadata

import "github.com/boltdb/bolt"

type migration struct {
	schema  string
	version int
	migrate func(*bolt.Tx) error
}

// migrations stores the list of database migrations
// for each update to the database schema. The migrations
// array MUST be ordered by version from least to greatest.
// The last entry in the array should correspond to the
// schemaVersion and dbVersion constants.
// A migration test MUST be added for each migration in
// the array.
// The migrate function can safely assume the version
// of the data it is migrating from is the previous version
// of the database.
var migrations = []migration{
	{
		schema:  "v1",
		version: 1,
		migrate: addChildLinks,
	},
}

// addChildLinks Adds children key to the snapshotters to enforce snapshot
// entries cannot be removed which have children
func addChildLinks(tx *bolt.Tx) error {
	v1bkt := tx.Bucket(bucketKeyVersion)
	if v1bkt == nil {
		return nil
	}

	// iterate through each namespace
	v1c := v1bkt.Cursor()

	for k, v := v1c.First(); k != nil; k, v = v1c.Next() {
		if v != nil {
			continue
		}
		nbkt := v1bkt.Bucket(k)

		sbkt := nbkt.Bucket(bucketKeyObjectSnapshots)
		if sbkt != nil {
			// Iterate through each snapshotter
			if err := sbkt.ForEach(func(sk, sv []byte) error {
				if sv != nil {
					return nil
				}
				snbkt := sbkt.Bucket(sk)

				// Iterate through each snapshot
				return snbkt.ForEach(func(k, v []byte) error {
					if v != nil {
						return nil
					}
					parent := snbkt.Bucket(k).Get(bucketKeyParent)
					if len(parent) > 0 {
						pbkt := snbkt.Bucket(parent)
						if pbkt == nil {
							// Not enforcing consistency during migration, skip
							return nil
						}
						cbkt, err := pbkt.CreateBucketIfNotExists(bucketKeyChildren)
						if err != nil {
							return err
						}
						if err := cbkt.Put(k, nil); err != nil {
							return err
						}
					}

					return nil
				})
			}); err != nil {
				return err
			}
		}
	}

	return nil
}
