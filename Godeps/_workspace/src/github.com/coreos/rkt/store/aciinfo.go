package store

import (
	"database/sql"
	"time"
)

// ACIInfo is used to store information about an ACI.
type ACIInfo struct {
	// BlobKey is the key in the blob/imageManifest store of the related
	// ACI file and is the db primary key.
	BlobKey string
	// AppName is the app name provided by the ACI.
	AppName string
	// ImportTime is the time this ACI was imported in the store.
	ImportTime time.Time
	// Latest defines if the ACI was imported using the latest pattern (no
	// version label was provided on ACI discovery)
	Latest bool
}

func NewACIInfo(blobKey string, latest bool, t time.Time) *ACIInfo {
	return &ACIInfo{
		BlobKey:    blobKey,
		Latest:     latest,
		ImportTime: t,
	}
}

// GetAciInfosWithKeyPrefix returns all the ACIInfos with a blobkey starting with the given prefix.
func GetACIInfosWithKeyPrefix(tx *sql.Tx, prefix string) ([]*ACIInfo, error) {
	aciinfos := []*ACIInfo{}
	rows, err := tx.Query("SELECT * from aciinfo WHERE hasPrefix(blobkey, $1)", prefix)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		aciinfo := &ACIInfo{}
		if err := rows.Scan(&aciinfo.BlobKey, &aciinfo.AppName, &aciinfo.ImportTime, &aciinfo.Latest); err != nil {
			return nil, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return aciinfos, err
}

// GetAciInfosWithAppName returns all the ACIInfos for a given appname. found will be
// false if no aciinfo exists.
func GetACIInfosWithAppName(tx *sql.Tx, appname string) ([]*ACIInfo, bool, error) {
	aciinfos := []*ACIInfo{}
	found := false
	rows, err := tx.Query("SELECT * from aciinfo WHERE appname == $1", appname)
	if err != nil {
		return nil, false, err
	}
	for rows.Next() {
		found = true
		aciinfo := &ACIInfo{}
		if err := rows.Scan(&aciinfo.BlobKey, &aciinfo.AppName, &aciinfo.ImportTime, &aciinfo.Latest); err != nil {
			return nil, false, err
		}
		aciinfos = append(aciinfos, aciinfo)
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}
	return aciinfos, found, err
}

// WriteACIInfo adds or updates the provided aciinfo.
func WriteACIInfo(tx *sql.Tx, aciinfo *ACIInfo) error {
	// ql doesn't have an INSERT OR UPDATE function so
	// it's faster to remove and reinsert the row
	_, err := tx.Exec("DELETE from aciinfo where blobkey == $1", aciinfo.BlobKey)
	if err != nil {
		return err
	}
	_, err = tx.Exec("INSERT into aciinfo values ($1, $2, $3, $4)", aciinfo.BlobKey, aciinfo.AppName, aciinfo.ImportTime, aciinfo.Latest)
	if err != nil {
		return err
	}

	return nil
}
