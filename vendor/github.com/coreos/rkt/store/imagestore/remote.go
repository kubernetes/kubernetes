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
package imagestore

import (
	"database/sql"
	"errors"
	"time"
)

var (
	ErrRemoteNotFound = errors.New("remote target not found")
)

func NewRemote(aciurl, sigurl string) *Remote {
	r := &Remote{
		ACIURL: aciurl,
		SigURL: sigurl,
	}
	return r
}

type Remote struct {
	// ACIURL is the URL used to import an ACI.
	ACIURL string
	// SigURL is the URL used to import an ACI verification signature.
	SigURL string
	ETag   string
	// The key in the blob store under which the ACI has been saved.
	BlobKey      string
	CacheMaxAge  int
	DownloadTime time.Time
}

func remoteRowScan(rows *sql.Rows, remote *Remote) error {
	return rows.Scan(&remote.ACIURL, &remote.SigURL, &remote.ETag, &remote.BlobKey, &remote.CacheMaxAge, &remote.DownloadTime)
}

// GetRemote tries to retrieve a remote with the given aciURL.
// If remote doesn't exist, it returns ErrRemoteNotFound error.
func GetRemote(tx *sql.Tx, aciURL string) (*Remote, error) {
	rows, err := tx.Query("SELECT * FROM remote WHERE aciurl == $1", aciURL)
	if err != nil {
		return nil, err
	}

	if ok := rows.Next(); !ok {
		return nil, ErrRemoteNotFound
	}

	remote := &Remote{}
	if err := remoteRowScan(rows, remote); err != nil {
		return nil, err
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return remote, nil
}

// GetAllRemotes returns all the ACIInfos sorted by optional sortfields and
// with ascending or descending order.
func GetAllRemotes(tx *sql.Tx) ([]*Remote, error) {
	var remotes []*Remote
	query := "SELECT * from remote"

	rows, err := tx.Query(query)
	if err != nil {
		return nil, err

	}

	for rows.Next() {
		r := &Remote{}
		if err := remoteRowScan(rows, r); err != nil {
			return nil, err

		}

		remotes = append(remotes, r)

	}

	if err := rows.Err(); err != nil {
		return nil, err

	}

	return remotes, nil

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
