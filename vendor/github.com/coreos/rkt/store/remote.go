// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package store implements a content-addressable-store on disk.
// It leverages the `diskv` package to store items in a simple
// key-value blob store: https://github.com/peterbourgon/diskv
package store

import (
	"database/sql"
	"time"
)

func NewRemote(aciurl, sigurl string) *Remote {
	r := &Remote{
		ACIURL: aciurl,
		SigURL: sigurl,
	}
	return r
}

type Remote struct {
	ACIURL string
	SigURL string
	ETag   string
	// The key in the blob store under which the ACI has been saved.
	BlobKey      string
	CacheMaxAge  int
	DownloadTime time.Time
}

// GetRemote tries to retrieve a remote with the given aciURL. found will be
// false if remote doesn't exist.
func GetRemote(tx *sql.Tx, aciURL string) (remote *Remote, found bool, err error) {
	remote = &Remote{}
	rows, err := tx.Query("SELECT * FROM remote WHERE aciurl == $1", aciURL)
	if err != nil {
		return nil, false, err
	}
	for rows.Next() {
		found = true
		if err := rows.Scan(&remote.ACIURL, &remote.SigURL, &remote.ETag, &remote.BlobKey, &remote.CacheMaxAge, &remote.DownloadTime); err != nil {
			return nil, false, err
		}
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}

	return remote, found, err
}

// WriteRemote adds or updates the provided Remote.
func WriteRemote(tx *sql.Tx, remote *Remote) error {
	// ql doesn't have an INSERT OR UPDATE function so
	// it's faster to remove and reinsert the row
	_, err := tx.Exec("DELETE FROM remote WHERE aciurl == $1", remote.ACIURL)
	if err != nil {
		return err
	}
	_, err = tx.Exec("INSERT INTO remote VALUES ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
	if err != nil {
		return err
	}
	return nil
}

// RemoveRemote removes the remote with the given blobKey.
func RemoveRemote(tx *sql.Tx, blobKey string) error {
	_, err := tx.Exec("DELETE FROM remote WHERE blobkey == $1", blobKey)
	if err != nil {
		return err
	}
	return nil
}
