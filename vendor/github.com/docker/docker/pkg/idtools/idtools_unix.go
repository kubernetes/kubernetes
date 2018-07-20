// +build !windows

package idtools

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"

	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/runc/libcontainer/user"
)

var (
	entOnce   sync.Once
	getentCmd string
)

func mkdirAs(path string, mode os.FileMode, ownerUID, ownerGID int, mkAll, chownExisting bool) error {
	// make an array containing the original path asked for, plus (for mkAll == true)
	// all path components leading up to the complete path that don't exist before we MkdirAll
	// so that we can chown all of them properly at the end.  If chownExisting is false, we won't
	// chown the full directory path if it exists
	var paths []string

	stat, err := system.Stat(path)
	if err == nil {
		if !stat.IsDir() {
			return &os.PathError{Op: "mkdir", Path: path, Err: syscall.ENOTDIR}
		}
		if !chownExisting {
			return nil
		}

		// short-circuit--we were called with an existing directory and chown was requested
		return lazyChown(path, ownerUID, ownerGID, stat)
	}

	if os.IsNotExist(err) {
		paths = []string{path}
	}

	if mkAll {
		// walk back to "/" looking for directories which do not exist
		// and add them to the paths array for chown after creation
		dirPath := path
		for {
			dirPath = filepath.Dir(dirPath)
			if dirPath == "/" {
				break
			}
			if _, err := os.Stat(dirPath); err != nil && os.IsNotExist(err) {
				paths = append(paths, dirPath)
			}
		}
		if err := system.MkdirAll(path, mode, ""); err != nil {
			return err
		}
	} else {
		if err := os.Mkdir(path, mode); err != nil && !os.IsExist(err) {
			return err
		}
	}
	// even if it existed, we will chown the requested path + any subpaths that
	// didn't exist when we called MkdirAll
	for _, pathComponent := range paths {
		if err := lazyChown(pathComponent, ownerUID, ownerGID, nil); err != nil {
			return err
		}
	}
	return nil
}

// CanAccess takes a valid (existing) directory and a uid, gid pair and determines
// if that uid, gid pair has access (execute bit) to the directory
func CanAccess(path string, pair IDPair) bool {
	statInfo, err := system.Stat(path)
	if err != nil {
		return false
	}
	fileMode := os.FileMode(statInfo.Mode())
	permBits := fileMode.Perm()
	return accessible(statInfo.UID() == uint32(pair.UID),
		statInfo.GID() == uint32(pair.GID), permBits)
}

func accessible(isOwner, isGroup bool, perms os.FileMode) bool {
	if isOwner && (perms&0100 == 0100) {
		return true
	}
	if isGroup && (perms&0010 == 0010) {
		return true
	}
	if perms&0001 == 0001 {
		return true
	}
	return false
}

// LookupUser uses traditional local system files lookup (from libcontainer/user) on a username,
// followed by a call to `getent` for supporting host configured non-files passwd and group dbs
func LookupUser(username string) (user.User, error) {
	// first try a local system files lookup using existing capabilities
	usr, err := user.LookupUser(username)
	if err == nil {
		return usr, nil
	}
	// local files lookup failed; attempt to call `getent` to query configured passwd dbs
	usr, err = getentUser(fmt.Sprintf("%s %s", "passwd", username))
	if err != nil {
		return user.User{}, err
	}
	return usr, nil
}

// LookupUID uses traditional local system files lookup (from libcontainer/user) on a uid,
// followed by a call to `getent` for supporting host configured non-files passwd and group dbs
func LookupUID(uid int) (user.User, error) {
	// first try a local system files lookup using existing capabilities
	usr, err := user.LookupUid(uid)
	if err == nil {
		return usr, nil
	}
	// local files lookup failed; attempt to call `getent` to query configured passwd dbs
	return getentUser(fmt.Sprintf("%s %d", "passwd", uid))
}

func getentUser(args string) (user.User, error) {
	reader, err := callGetent(args)
	if err != nil {
		return user.User{}, err
	}
	users, err := user.ParsePasswd(reader)
	if err != nil {
		return user.User{}, err
	}
	if len(users) == 0 {
		return user.User{}, fmt.Errorf("getent failed to find passwd entry for %q", strings.Split(args, " ")[1])
	}
	return users[0], nil
}

// LookupGroup uses traditional local system files lookup (from libcontainer/user) on a group name,
// followed by a call to `getent` for supporting host configured non-files passwd and group dbs
func LookupGroup(groupname string) (user.Group, error) {
	// first try a local system files lookup using existing capabilities
	group, err := user.LookupGroup(groupname)
	if err == nil {
		return group, nil
	}
	// local files lookup failed; attempt to call `getent` to query configured group dbs
	return getentGroup(fmt.Sprintf("%s %s", "group", groupname))
}

// LookupGID uses traditional local system files lookup (from libcontainer/user) on a group ID,
// followed by a call to `getent` for supporting host configured non-files passwd and group dbs
func LookupGID(gid int) (user.Group, error) {
	// first try a local system files lookup using existing capabilities
	group, err := user.LookupGid(gid)
	if err == nil {
		return group, nil
	}
	// local files lookup failed; attempt to call `getent` to query configured group dbs
	return getentGroup(fmt.Sprintf("%s %d", "group", gid))
}

func getentGroup(args string) (user.Group, error) {
	reader, err := callGetent(args)
	if err != nil {
		return user.Group{}, err
	}
	groups, err := user.ParseGroup(reader)
	if err != nil {
		return user.Group{}, err
	}
	if len(groups) == 0 {
		return user.Group{}, fmt.Errorf("getent failed to find groups entry for %q", strings.Split(args, " ")[1])
	}
	return groups[0], nil
}

func callGetent(args string) (io.Reader, error) {
	entOnce.Do(func() { getentCmd, _ = resolveBinary("getent") })
	// if no `getent` command on host, can't do anything else
	if getentCmd == "" {
		return nil, fmt.Errorf("")
	}
	out, err := execCmd(getentCmd, args)
	if err != nil {
		exitCode, errC := system.GetExitCode(err)
		if errC != nil {
			return nil, err
		}
		switch exitCode {
		case 1:
			return nil, fmt.Errorf("getent reported invalid parameters/database unknown")
		case 2:
			terms := strings.Split(args, " ")
			return nil, fmt.Errorf("getent unable to find entry %q in %s database", terms[1], terms[0])
		case 3:
			return nil, fmt.Errorf("getent database doesn't support enumeration")
		default:
			return nil, err
		}

	}
	return bytes.NewReader(out), nil
}

// lazyChown performs a chown only if the uid/gid don't match what's requested
// Normally a Chown is a no-op if uid/gid match, but in some cases this can still cause an error, e.g. if the
// dir is on an NFS share, so don't call chown unless we absolutely must.
func lazyChown(p string, uid, gid int, stat *system.StatT) error {
	if stat == nil {
		var err error
		stat, err = system.Stat(p)
		if err != nil {
			return err
		}
	}
	if stat.UID() == uint32(uid) && stat.GID() == uint32(gid) {
		return nil
	}
	return os.Chown(p, uid, gid)
}
