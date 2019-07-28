// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package user

import (
	"io"
	"os"
	"strconv"

	"golang.org/x/sys/unix"
)

// Unix-specific path to the passwd and group formatted files.
const (
	unixPasswdPath = "/etc/passwd"
	unixGroupPath  = "/etc/group"
)

func lookupUser(username string) (User, error) {
	return lookupUserFunc(func(u User) bool {
		return u.Name == username
	})
}

func lookupUid(uid int) (User, error) {
	return lookupUserFunc(func(u User) bool {
		return u.Uid == uid
	})
}

func lookupUserFunc(filter func(u User) bool) (User, error) {
	// Get operating system-specific passwd reader-closer.
	passwd, err := GetPasswd()
	if err != nil {
		return User{}, err
	}
	defer passwd.Close()

	// Get the users.
	users, err := ParsePasswdFilter(passwd, filter)
	if err != nil {
		return User{}, err
	}

	// No user entries found.
	if len(users) == 0 {
		return User{}, ErrNoPasswdEntries
	}

	// Assume the first entry is the "correct" one.
	return users[0], nil
}

func lookupGroup(groupname string) (Group, error) {
	return lookupGroupFunc(func(g Group) bool {
		return g.Name == groupname
	})
}

func lookupGid(gid int) (Group, error) {
	return lookupGroupFunc(func(g Group) bool {
		return g.Gid == gid
	})
}

func lookupGroupFunc(filter func(g Group) bool) (Group, error) {
	// Get operating system-specific group reader-closer.
	group, err := GetGroup()
	if err != nil {
		return Group{}, err
	}
	defer group.Close()

	// Get the users.
	groups, err := ParseGroupFilter(group, filter)
	if err != nil {
		return Group{}, err
	}

	// No user entries found.
	if len(groups) == 0 {
		return Group{}, ErrNoGroupEntries
	}

	// Assume the first entry is the "correct" one.
	return groups[0], nil
}

func GetPasswdPath() (string, error) {
	return unixPasswdPath, nil
}

func GetPasswd() (io.ReadCloser, error) {
	return os.Open(unixPasswdPath)
}

func GetGroupPath() (string, error) {
	return unixGroupPath, nil
}

func GetGroup() (io.ReadCloser, error) {
	return os.Open(unixGroupPath)
}

// CurrentUser looks up the current user by their user id in /etc/passwd. If the
// user cannot be found (or there is no /etc/passwd file on the filesystem),
// then CurrentUser returns an error.
func CurrentUser() (User, error) {
	return LookupUid(unix.Getuid())
}

// CurrentGroup looks up the current user's group by their primary group id's
// entry in /etc/passwd. If the group cannot be found (or there is no
// /etc/group file on the filesystem), then CurrentGroup returns an error.
func CurrentGroup() (Group, error) {
	return LookupGid(unix.Getgid())
}

func currentUserSubIDs(fileName string) ([]SubID, error) {
	u, err := CurrentUser()
	if err != nil {
		return nil, err
	}
	filter := func(entry SubID) bool {
		return entry.Name == u.Name || entry.Name == strconv.Itoa(u.Uid)
	}
	return ParseSubIDFileFilter(fileName, filter)
}

func CurrentUserSubUIDs() ([]SubID, error) {
	return currentUserSubIDs("/etc/subuid")
}

func CurrentUserSubGIDs() ([]SubID, error) {
	return currentUserSubIDs("/etc/subgid")
}

func CurrentProcessUIDMap() ([]IDMap, error) {
	return ParseIDMapFile("/proc/self/uid_map")
}

func CurrentProcessGIDMap() ([]IDMap, error) {
	return ParseIDMapFile("/proc/self/gid_map")
}
