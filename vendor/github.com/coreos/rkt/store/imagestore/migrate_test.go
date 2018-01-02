// Copyright 2015 The rkt Authors
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

package imagestore

import (
	"database/sql"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/rkt/store/db"
	"github.com/davecgh/go-spew/spew"
	"github.com/jonboulle/clockwork"
)

type testdb interface {
	version() int
	populate(db *db.DB) error
	load(db *db.DB) error
	compare(db testdb) bool
}

type DBV0 struct {
	aciinfos []*ACIInfoV0_2
	remotes  []*RemoteV0_1
}

func (d *DBV0) version() int {
	return 0
}

func (d *DBV0) populate(db *db.DB) error {
	// As DBV0 and DBV1 have the same schema use a common populate
	// function.
	return populateDBV0_1(db, d.version(), d.aciinfos, d.remotes)
}

// load populates the given struct with the data in db.
// the given struct d should be empty
func (d *DBV0) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV0_2(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV0_1(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV0) compare(td testdb) bool {
	d2, ok := td.(*DBV0)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d2.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d2.remotes) {
		return false
	}
	return true
}

type DBV1 struct {
	aciinfos []*ACIInfoV0_2
	remotes  []*RemoteV0_1
}

func (d *DBV1) version() int {
	return 1
}
func (d *DBV1) populate(db *db.DB) error {
	return populateDBV0_1(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV1) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV0_2(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV0_1(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV1) compare(td testdb) bool {
	d2, ok := td.(*DBV1)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d2.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d2.remotes) {
		return false
	}
	return true
}

type DBV2 struct {
	aciinfos []*ACIInfoV0_2
	remotes  []*RemoteV2_7
}

func (d *DBV2) version() int {
	return 2
}
func (d *DBV2) populate(db *db.DB) error {
	return populateDBV2(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV2) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV0_2(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV2_7(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV2) compare(td testdb) bool {
	d2, ok := td.(*DBV2)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d2.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d2.remotes) {
		return false
	}
	return true
}

type DBV3 struct {
	aciinfos []*ACIInfoV3
	remotes  []*RemoteV2_7
}

func (d *DBV3) version() int {
	return 3
}
func (d *DBV3) populate(db *db.DB) error {
	return populateDBV3(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV3) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV3(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV2_7(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV3) compare(td testdb) bool {
	d3, ok := td.(*DBV3)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d3.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d3.remotes) {
		return false
	}
	return true
}

type DBV4 struct {
	aciinfos []*ACIInfoV4
	remotes  []*RemoteV2_7
}

func (d *DBV4) version() int {
	return 4
}
func (d *DBV4) populate(db *db.DB) error {
	return populateDBV4(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV4) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV4(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV2_7(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV4) compare(td testdb) bool {
	d4, ok := td.(*DBV4)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d4.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d4.remotes) {
		return false
	}
	return true
}

type DBV5 struct {
	aciinfos []*ACIInfoV5
	remotes  []*RemoteV2_7
}

func (d *DBV5) version() int {
	return 5
}
func (d *DBV5) populate(db *db.DB) error {
	return populateDBV5(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV5) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV5(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV2_7(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV5) compare(td testdb) bool {
	d5, ok := td.(*DBV5)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d5.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d5.remotes) {
		return false
	}
	return true
}

type DBV6 struct {
	aciinfos []*ACIInfoV6
	remotes  []*RemoteV2_7
}

func (d *DBV6) version() int {
	return 6
}
func (d *DBV6) populate(db *db.DB) error {
	return populateDBV6(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV6) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV6(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV2_7(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV6) compare(td testdb) bool {
	d6, ok := td.(*DBV6)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d6.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d6.remotes) {
		return false
	}
	return true
}

type DBV7 struct {
	aciinfos []*ACIInfoV7
	remotes  []*RemoteV2_7
}

func (d *DBV7) version() int {
	return 7
}
func (d *DBV7) populate(db *db.DB) error {
	return populateDBV7(db, d.version(), d.aciinfos, d.remotes)
}

func (d *DBV7) load(db *db.DB) error {
	fn := func(tx *sql.Tx) error {
		var err error
		d.aciinfos, err = getAllACIInfosV7(tx)
		if err != nil {
			return err
		}
		d.remotes, err = getAllRemoteV2_7(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}
	return nil
}

func (d *DBV7) compare(td testdb) bool {
	d7, ok := td.(*DBV7)
	if !ok {
		return false
	}
	if !compareSlicesNoOrder(d.aciinfos, d7.aciinfos) {
		return false
	}
	if !compareSlicesNoOrder(d.remotes, d7.remotes) {
		return false
	}
	return true
}

// The ACIInfo struct for different db versions. The ending VX_Y represent the
// first and the last version where the format isn't changed
// The latest existing struct should be updated when updating the db version
// without changing the struct format (ex. V0_1 to V0_2).
// A new struct and its relative function should be added if the format is changed.
// The same applies for all of the the other structs.
type ACIInfoV0_2 struct {
	BlobKey    string
	AppName    string
	ImportTime time.Time
	Latest     bool
}

type ACIInfoV3 struct {
	BlobKey    string
	Name       string
	ImportTime time.Time
	Latest     bool
}

type ACIInfoV4 struct {
	BlobKey    string
	Name       string
	ImportTime time.Time
	LastUsed   time.Time
	Latest     bool
}

type ACIInfoV5 struct {
	BlobKey       string
	Name          string
	ImportTime    time.Time
	LastUsed      time.Time
	Latest        bool
	Size          int64
	TreeStoreSize int64
}

type ACIInfoV6 struct {
	BlobKey         string
	Name            string
	ImportTime      time.Time
	LastUsed        time.Time
	Latest          bool
	Size            int64
	TreeStoreSize   int64
	InsecureOptions int64
}

type ACIInfoV7 struct {
	BlobKey       string
	Name          string
	ImportTime    time.Time
	LastUsed      time.Time
	Latest        bool
	Size          int64
	TreeStoreSize int64
}

func getAllACIInfosV0_2(tx *sql.Tx) ([]*ACIInfoV0_2, error) {
	var aciinfos []*ACIInfoV0_2
	rows, err := tx.Query("SELECT * FROM aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV0_2{}
		if err := rows.Scan(&aciinfo.BlobKey, &aciinfo.AppName, &aciinfo.ImportTime, &aciinfo.Latest); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

func getAllACIInfosV3(tx *sql.Tx) ([]*ACIInfoV3, error) {
	var aciinfos []*ACIInfoV3
	rows, err := tx.Query("SELECT * FROM aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV3{}
		if rows.Scan(&aciinfo.BlobKey, &aciinfo.Name, &aciinfo.ImportTime, &aciinfo.Latest); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

func getAllACIInfosV4(tx *sql.Tx) ([]*ACIInfoV4, error) {
	var aciinfos []*ACIInfoV4
	rows, err := tx.Query("SELECT * FROM aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV4{}
		if rows.Scan(&aciinfo.BlobKey, &aciinfo.Name, &aciinfo.ImportTime, &aciinfo.LastUsed, &aciinfo.Latest); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

func getAllACIInfosV5(tx *sql.Tx) ([]*ACIInfoV5, error) {
	var aciinfos []*ACIInfoV5
	rows, err := tx.Query("SELECT * FROM aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV5{}
		if rows.Scan(&aciinfo.BlobKey, &aciinfo.Name, &aciinfo.ImportTime, &aciinfo.LastUsed, &aciinfo.Latest, &aciinfo.Size, &aciinfo.TreeStoreSize); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

func getAllACIInfosV6(tx *sql.Tx) ([]*ACIInfoV6, error) {
	var aciinfos []*ACIInfoV6
	rows, err := tx.Query("SELECT * FROM aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV6{}
		if rows.Scan(&aciinfo.BlobKey, &aciinfo.Name, &aciinfo.ImportTime, &aciinfo.LastUsed, &aciinfo.Latest, &aciinfo.Size, &aciinfo.TreeStoreSize, &aciinfo.InsecureOptions); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

func getAllACIInfosV7(tx *sql.Tx) ([]*ACIInfoV7, error) {
	var aciinfos []*ACIInfoV7
	rows, err := tx.Query("SELECT * FROM aciinfo")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfoV7{}
		if rows.Scan(&aciinfo.BlobKey, &aciinfo.Name, &aciinfo.ImportTime, &aciinfo.LastUsed, &aciinfo.Latest, &aciinfo.Size, &aciinfo.TreeStoreSize); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, nil
}

type RemoteV0_1 struct {
	ACIURL  string
	SigURL  string
	ETag    string
	BlobKey string
}

func getAllRemoteV0_1(tx *sql.Tx) ([]*RemoteV0_1, error) {
	var remotes []*RemoteV0_1
	rows, err := tx.Query("SELECT * FROM remote")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		remote := &RemoteV0_1{}
		if err := rows.Scan(&remote.ACIURL, &remote.SigURL, &remote.ETag, &remote.BlobKey); err != nil {
			return nil, err
		}
		remotes = append(remotes, remote)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return remotes, nil
}

type RemoteV2_7 struct {
	ACIURL       string
	SigURL       string
	ETag         string
	BlobKey      string
	CacheMaxAge  int
	DownloadTime time.Time
}

func getAllRemoteV2_7(tx *sql.Tx) ([]*RemoteV2_7, error) {
	var remotes []*RemoteV2_7
	rows, err := tx.Query("SELECT * FROM remote")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		remote := &RemoteV2_7{}
		if err := rows.Scan(&remote.ACIURL, &remote.SigURL, &remote.ETag, &remote.BlobKey, &remote.CacheMaxAge, &remote.DownloadTime); err != nil {
			return nil, err
		}
		remotes = append(remotes, remote)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return remotes, nil
}

func populateDBV0_1(db *db.DB, dbVersion int, aciInfos []*ACIInfoV0_2, remotes []*RemoteV0_1) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, appname string, importtime time, latest bool);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS appnameidx ON aciinfo (appname)",
	}

	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT into aciinfo values ($1, $2, $3, $4)", aciinfo.BlobKey, aciinfo.AppName, aciinfo.ImportTime, aciinfo.Latest)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

func populateDBV2(db *db.DB, dbVersion int, aciInfos []*ACIInfoV0_2, remotes []*RemoteV2_7) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, appname string, importtime time, latest bool);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS appnameidx ON aciinfo (appname)",
	}

	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT into aciinfo values ($1, $2, $3, $4)", aciinfo.BlobKey, aciinfo.AppName, aciinfo.ImportTime, aciinfo.Latest)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

func populateDBV3(db *db.DB, dbVersion int, aciInfos []*ACIInfoV3, remotes []*RemoteV2_7) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, importtime time, latest bool, name string);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
	}
	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT into aciinfo values ($1, $2, $3, $4)", aciinfo.BlobKey, aciinfo.ImportTime, aciinfo.Latest, aciinfo.Name)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

func populateDBV4(db *db.DB, dbVersion int, aciInfos []*ACIInfoV4, remotes []*RemoteV2_7) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, name string, importtime time, lastusedtime time, latest bool);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
	}
	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT INTO aciinfo (blobkey, name, importtime, lastusedtime, latest) VALUES ($1, $2, $3, $4, $5)", aciinfo.BlobKey, aciinfo.Name, aciinfo.ImportTime, aciinfo.LastUsed, aciinfo.Latest)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

func populateDBV5(db *db.DB, dbVersion int, aciInfos []*ACIInfoV5, remotes []*RemoteV2_7) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64, treestoresize int64);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
	}
	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT INTO aciinfo (blobkey, name, importtime, lastused, latest, size, treestoresize) VALUES ($1, $2, $3, $4, $5, $6, $7)", aciinfo.BlobKey, aciinfo.Name, aciinfo.ImportTime, aciinfo.LastUsed, aciinfo.Latest, aciinfo.Size, aciinfo.TreeStoreSize)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

func populateDBV6(db *db.DB, dbVersion int, aciInfos []*ACIInfoV6, remotes []*RemoteV2_7) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64, treestoresize int64, insecureoptions int64 DEFAULT 0);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
	}
	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT INTO aciinfo (blobkey, name, importtime, lastused, latest, size, treestoresize, insecureoptions) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)", aciinfo.BlobKey, aciinfo.Name, aciinfo.ImportTime, aciinfo.LastUsed, aciinfo.Latest, aciinfo.Size, aciinfo.TreeStoreSize, aciinfo.InsecureOptions)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

func populateDBV7(db *db.DB, dbVersion int, aciInfos []*ACIInfoV7, remotes []*RemoteV2_7) error {
	var dbCreateStmts = [...]string{
		// version table
		"CREATE TABLE IF NOT EXISTS version (version int);",
		fmt.Sprintf("INSERT INTO version VALUES (%d)", dbVersion),

		// remote table. The primary key is "aciurl".
		"CREATE TABLE IF NOT EXISTS remote (aciurl string, sigurl string, etag string, blobkey string, cachemaxage int, downloadtime time);",
		"CREATE UNIQUE INDEX IF NOT EXISTS aciurlidx ON remote (aciurl)",

		// aciinfo table. The primary key is "blobkey" and it matches the key used to save that aci in the blob store
		"CREATE TABLE IF NOT EXISTS aciinfo (blobkey string, name string, importtime time, lastused time, latest bool, size int64, treestoresize int64);",
		"CREATE UNIQUE INDEX IF NOT EXISTS blobkeyidx ON aciinfo (blobkey)",
		"CREATE INDEX IF NOT EXISTS nameidx ON aciinfo (name)",
	}
	fn := func(tx *sql.Tx) error {
		for _, stmt := range dbCreateStmts {
			_, err := tx.Exec(stmt)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, aciinfo := range aciInfos {
			_, err := tx.Exec("INSERT INTO aciinfo (blobkey, name, importtime, lastused, latest, size, treestoresize) VALUES ($1, $2, $3, $4, $5, $6, $7)", aciinfo.BlobKey, aciinfo.Name, aciinfo.ImportTime, aciinfo.LastUsed, aciinfo.Latest, aciinfo.Size, aciinfo.TreeStoreSize)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	fn = func(tx *sql.Tx) error {
		for _, remote := range remotes {
			_, err := tx.Exec("INSERT into remote values ($1, $2, $3, $4, $5, $6)", remote.ACIURL, remote.SigURL, remote.ETag, remote.BlobKey, remote.CacheMaxAge, remote.DownloadTime)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if err := db.Do(fn); err != nil {
		return err
	}

	return nil
}

type migrateTest struct {
	predb  testdb
	postdb testdb
	// Needed to have the right DB type to load from
	curdb testdb
}

func testMigrate(tt migrateTest) error {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		return fmt.Errorf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)

	storeDir := filepath.Join(dir, "store")
	db, err := db.NewDB(filepath.Join(storeDir, "db"))
	if err != nil {
		return err
	}
	if err = tt.predb.populate(db); err != nil {
		return err
	}

	fn := func(tx *sql.Tx) error {
		err := migrate(tx, tt.postdb.version())
		if err != nil {
			return err
		}
		return nil
	}
	if err = db.Do(fn); err != nil {
		return err
	}

	var curDBVersion int
	fn = func(tx *sql.Tx) error {
		var err error
		curDBVersion, err = getDBVersion(tx)
		if err != nil {
			return err
		}
		return nil
	}
	if err = db.Do(fn); err != nil {
		return err
	}
	if curDBVersion != tt.postdb.version() {
		return fmt.Errorf("wrong db version: got %#v, want %#v", curDBVersion, tt.postdb.version())
	}

	if err := tt.curdb.load(db); err != nil {
		return err
	}

	if !tt.curdb.compare(tt.postdb) {
		return spew.Errorf("while comparing DBs:\n\tgot %#v\n\twant %#v\n", tt.curdb, tt.postdb)
	}

	return nil
}

func TestMigrate(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)

	blobKeys := []string{
		"sha512-aaaaaaaa",
		"sha512-bbbbbbbb",
	}

	names := []string{
		"example.com/app01",
		"example.com/app02",
	}

	sizes := []int64{
		0,
		2 << 8,
	}

	treeStoreSizes := []int64{
		0,
		2 << 8,
		2 << 16,
	}

	insecureOptions := []int64{
		0,
		2,
	}

	now := time.Now().UTC()

	// Injects a fake clock to store pkg clock
	clock = clockwork.NewFakeClockAt(now)

	tests := []migrateTest{
		{
			// Test migration from V0 to V1 with an empty database
			&DBV0{},
			&DBV1{},
			&DBV1{},
		},

		{
			// Test migration from V0 to V1
			&DBV0{
				[]*ACIInfoV0_2{
					{blobKeys[0], names[0], now, false},
					{blobKeys[1], names[1], now, true},
				},
				[]*RemoteV0_1{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0]},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1]},
				},
			},
			&DBV1{
				[]*ACIInfoV0_2{
					{blobKeys[0], names[0], now, false},
					{blobKeys[1], names[1], now, true},
				},
				[]*RemoteV0_1{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0]},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1]},
				},
			},
			&DBV1{},
		},

		{
			// Test migration from V1 to V2
			&DBV1{
				[]*ACIInfoV0_2{
					{blobKeys[0], "example.com/app01", now, false},
					{blobKeys[1], "example.com/app02", now, true},
				},
				[]*RemoteV0_1{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0]},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1]},
				},
			},
			&DBV2{
				[]*ACIInfoV0_2{
					{blobKeys[0], "example.com/app01", now, false},
					{blobKeys[1], "example.com/app02", now, true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV2{},
		},

		{
			// Test migration from V2 to V3
			&DBV2{
				[]*ACIInfoV0_2{
					{blobKeys[0], "example.com/app01", now, false},
					{blobKeys[1], "example.com/app02", now, true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV3{
				[]*ACIInfoV3{
					{blobKeys[0], "example.com/app01", now, false},
					{blobKeys[1], "example.com/app02", now, true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV3{},
		},

		{
			// Test migration from V3 to V4
			&DBV3{
				[]*ACIInfoV3{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, Latest: false},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, Latest: true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV4{
				[]*ACIInfoV4{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV4{},
		},

		{
			// Test migration from V4 to V5
			&DBV4{
				[]*ACIInfoV4{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV5{
				[]*ACIInfoV5{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV5{},
		},

		{
			// Test migration from V5 to V6
			&DBV5{
				[]*ACIInfoV5{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false, Size: sizes[0], TreeStoreSize: treeStoreSizes[0]},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true, Size: sizes[1], TreeStoreSize: treeStoreSizes[1]},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV6{
				[]*ACIInfoV6{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false, Size: sizes[0], TreeStoreSize: treeStoreSizes[0]},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true, Size: sizes[1], TreeStoreSize: treeStoreSizes[1]},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV6{},
		},
		{
			// Test migration from V6 to V7
			&DBV6{
				[]*ACIInfoV6{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false, Size: sizes[0], TreeStoreSize: treeStoreSizes[0], InsecureOptions: insecureOptions[0]},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true, Size: sizes[1], TreeStoreSize: treeStoreSizes[1], InsecureOptions: insecureOptions[1]},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV7{
				[]*ACIInfoV7{
					{BlobKey: blobKeys[0], Name: names[0], ImportTime: now, LastUsed: now, Latest: false, Size: sizes[0], TreeStoreSize: treeStoreSizes[0]},
					{BlobKey: blobKeys[1], Name: names[1], ImportTime: now, LastUsed: now, Latest: true, Size: sizes[1], TreeStoreSize: treeStoreSizes[1]},
				},
				[]*RemoteV2_7{
					{"http://example.com/app01.aci", "http://example.com/app01.aci.asc", "", blobKeys[0], 0, time.Time{}.UTC()},
					{"http://example.com/app02.aci", "http://example.com/app02.aci.asc", "", blobKeys[1], 0, time.Time{}.UTC()},
				},
			},
			&DBV7{},
		},
	}

	for i, tt := range tests {
		if err := testMigrate(tt); err != nil {
			t.Errorf("#%d: unexpected error: %v", i, err)
		}
	}
}

// compareSlices compare slices regardless of the slice elements order
func compareSlicesNoOrder(i1 interface{}, i2 interface{}) bool {
	s1 := interfaceToSlice(i1)
	s2 := interfaceToSlice(i2)

	if len(s1) != len(s2) {
		return false
	}

	seen := map[int]bool{}
	for _, v1 := range s1 {
		found := false
		for i2, v2 := range s2 {
			if _, ok := seen[i2]; ok {
				continue
			}

			if reflect.DeepEqual(v1, v2) {
				found = true
				seen[i2] = true
				continue
			}
		}

		if !found {
			return false
		}

	}
	return true
}

func interfaceToSlice(s interface{}) []interface{} {
	v := reflect.ValueOf(s)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		panic(fmt.Errorf("Expected slice or array, got %T", s))
	}
	l := v.Len()
	m := make([]interface{}, l)
	for i := 0; i < l; i++ {
		m[i] = v.Index(i).Interface()
	}
	return m
}
