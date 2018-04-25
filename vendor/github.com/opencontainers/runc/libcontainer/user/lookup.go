package user

import (
	"errors"
)

var (
	// The current operating system does not provide the required data for user lookups.
	ErrUnsupported = errors.New("user lookup: operating system does not provide passwd-formatted data")
	// No matching entries found in file.
	ErrNoPasswdEntries = errors.New("no matching entries in passwd file")
	ErrNoGroupEntries  = errors.New("no matching entries in group file")
)

// LookupUser looks up a user by their username in /etc/passwd. If the user
// cannot be found (or there is no /etc/passwd file on the filesystem), then
// LookupUser returns an error.
func LookupUser(username string) (User, error) {
	return lookupUser(username)
}

// LookupUid looks up a user by their user id in /etc/passwd. If the user cannot
// be found (or there is no /etc/passwd file on the filesystem), then LookupId
// returns an error.
func LookupUid(uid int) (User, error) {
	return lookupUid(uid)
}

// LookupGroup looks up a group by its name in /etc/group. If the group cannot
// be found (or there is no /etc/group file on the filesystem), then LookupGroup
// returns an error.
func LookupGroup(groupname string) (Group, error) {
	return lookupGroup(groupname)
}

// LookupGid looks up a group by its group id in /etc/group. If the group cannot
// be found (or there is no /etc/group file on the filesystem), then LookupGid
// returns an error.
func LookupGid(gid int) (Group, error) {
	return lookupGid(gid)
}
