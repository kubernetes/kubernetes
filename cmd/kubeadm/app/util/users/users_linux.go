//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package users

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/pkg/errors"

	"k8s.io/klog/v2"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// EntryMap holds a map of user or group entries.
type EntryMap struct {
	entries map[string]*entry
}

// UsersAndGroups is a structure that holds entry maps of users and groups.
// It is returned by AddUsersAndGroups.
type UsersAndGroups struct {
	// Users is an entry map of users.
	Users *EntryMap
	// Groups is an entry map of groups.
	Groups *EntryMap
}

// entry is a structure that holds information about a UNIX user or group.
// It partialially conforms parsing of both users from /etc/passwd and groups from /etc/group.
type entry struct {
	name      string
	id        int64
	gid       int64
	userNames []string
	shell     string
}

// limits is used to hold information about the minimum and maximum system ranges for UID and GID.
type limits struct {
	minUID, maxUID, minGID, maxGID int64
}

const (
	// These are constants used when parsing /etc/passwd or /etc/group in terms of how many
	// fields and entry has.
	totalFieldsGroup = 4
	totalFieldsUser  = 7

	// klogLevel holds the klog level to use for output.
	klogLevel = 5

	// noshell holds a path to a binary to disable shell login.
	noshell = "/bin/false"

	// These are constants for the default system paths on Linux.
	fileEtcLoginDefs = "/etc/login.defs"
	fileEtcPasswd    = "/etc/passwd"
	fileEtcGroup     = "/etc/group"
)

var (
	// these entries hold the users and groups to create as defined in:
	// https://git.k8s.io/enhancements/keps/sig-cluster-lifecycle/kubeadm/2568-kubeadm-non-root-control-plane
	usersToCreateSpec = []*entry{
		{name: constants.EtcdUserName},
		{name: constants.KubeAPIServerUserName},
		{name: constants.KubeControllerManagerUserName},
		{name: constants.KubeSchedulerUserName},
	}
	groupsToCreateSpec = []*entry{
		{name: constants.EtcdUserName, userNames: []string{constants.EtcdUserName}},
		{name: constants.KubeAPIServerUserName, userNames: []string{constants.KubeAPIServerUserName}},
		{name: constants.KubeControllerManagerUserName, userNames: []string{constants.KubeControllerManagerUserName}},
		{name: constants.KubeSchedulerUserName, userNames: []string{constants.KubeSchedulerUserName}},
		{name: constants.ServiceAccountKeyReadersGroupName, userNames: []string{constants.KubeAPIServerUserName, constants.KubeControllerManagerUserName}},
	}

	// defaultLimits holds the default limits in case values are missing in /etc/login.defs
	defaultLimits = &limits{minUID: 100, maxUID: 999, minGID: 100, maxGID: 999}
)

// ID returns the ID for an entry based on the entry name.
// In case of a user entry it returns the user UID.
// In case of a group entry it returns the group GID.
// It returns nil if no such entry exists.
func (u *EntryMap) ID(name string) *int64 {
	entry, ok := u.entries[name]
	if !ok {
		return nil
	}
	id := entry.id
	return &id
}

// String converts an EntryMap object to a readable string.
func (u *EntryMap) String() string {
	lines := make([]string, 0, len(u.entries))
	for k, e := range u.entries {
		lines = append(lines, fmt.Sprintf("%s{%d,%d};", k, e.id, e.gid))
	}
	sort.Strings(lines)
	return strings.Join(lines, "")
}

// AddUsersAndGroups is a public wrapper around addUsersAndGroupsImpl with default system file paths.
func AddUsersAndGroups() (*UsersAndGroups, error) {
	return addUsersAndGroupsImpl(fileEtcLoginDefs, fileEtcPasswd, fileEtcGroup)
}

// addUsersAndGroupsImpl adds the managed users and groups to the files specified
// by pathUsers and pathGroups. It uses the file specified with pathLoginDef to
// determine limits for UID and GID. If managed users and groups exist in these files
// validation is performed on them. The function returns a pointer to a Users object
// that can be used to return UID and GID of managed users.
func addUsersAndGroupsImpl(pathLoginDef, pathUsers, pathGroups string) (*UsersAndGroups, error) {
	klog.V(1).Info("Adding managed users and groups")
	klog.V(klogLevel).Infof("Parsing %q", pathLoginDef)

	// Read and parse /etc/login.def. Some distributions might be missing this file, which makes
	// them non-standard. If an error occurs fallback to defaults by passing an empty string
	// to parseLoginDefs().
	var loginDef string
	f, close, err := openFileWithLock(pathLoginDef)
	if err != nil {
		klog.V(1).Infof("Could not open %q, using default system limits: %v", pathLoginDef, err)
	} else {
		loginDef, err = readFile(f)
		if err != nil {
			klog.V(1).Infof("Could not read %q, using default system limits: %v", pathLoginDef, err)
		}
		close()
	}
	limits, err := parseLoginDefs(loginDef)
	if err != nil {
		return nil, err
	}

	klog.V(klogLevel).Infof("Using system UID/GID limits: %+v", limits)
	klog.V(klogLevel).Infof("Parsing %q and %q", pathUsers, pathGroups)

	// Open /etc/passwd and /etc/group with locks.
	fUsers, close, err := openFileWithLock(pathUsers)
	if err != nil {
		return nil, err
	}
	defer close()
	fGroups, close, err := openFileWithLock(pathGroups)
	if err != nil {
		return nil, err
	}
	defer close()

	// Read the files.
	fileUsers, err := readFile(fUsers)
	if err != nil {
		return nil, err
	}
	fileGroups, err := readFile(fGroups)
	if err != nil {
		return nil, err
	}

	// Parse the files.
	users, err := parseEntries(fileUsers, totalFieldsUser)
	if err != nil {
		return nil, errors.Wrapf(err, "could not parse %q", pathUsers)
	}
	groups, err := parseEntries(fileGroups, totalFieldsGroup)
	if err != nil {
		return nil, errors.Wrapf(err, "could not parse %q", pathGroups)
	}

	klog.V(klogLevel).Info("Validating existing users and groups")

	// Validate for existing tracked entries based on limits.
	usersToCreate, groupsToCreate, err := validateEntries(users, groups, limits)
	if err != nil {
		return nil, errors.Wrap(err, "error validating existing users and groups")
	}

	// Allocate and assign IDs to users / groups.
	allocUIDs, err := allocateIDs(users, limits.minUID, limits.maxUID, len(usersToCreate))
	if err != nil {
		return nil, err
	}
	allocGIDs, err := allocateIDs(groups, limits.minGID, limits.maxGID, len(groupsToCreate))
	if err != nil {
		return nil, err
	}
	if err := assignUserAndGroupIDs(groups, usersToCreate, groupsToCreate, allocUIDs, allocGIDs); err != nil {
		return nil, err
	}

	if len(usersToCreate) > 0 {
		klog.V(klogLevel).Infof("Adding users: %s", entriesToString(usersToCreate))
	}
	if len(groupsToCreate) > 0 {
		klog.V(klogLevel).Infof("Adding groups: %s", entriesToString(groupsToCreate))
	}

	// Add users and groups.
	fileUsers = addEntries(fileUsers, usersToCreate, createUser)
	fileGroups = addEntries(fileGroups, groupsToCreate, createGroup)

	// Write the files.
	klog.V(klogLevel).Infof("Writing %q and %q", pathUsers, pathGroups)
	if err := writeFile(fUsers, fileUsers); err != nil {
		return nil, err
	}
	if err := writeFile(fGroups, fileGroups); err != nil {
		return nil, err
	}

	// Prepare the maps of users and groups.
	usersConcat := append(users, usersToCreate...)
	mapUsers, err := entriesToEntryMap(usersConcat, usersToCreateSpec)
	if err != nil {
		return nil, err
	}
	groupsConcat := append(groups, groupsToCreate...)
	mapGroups, err := entriesToEntryMap(groupsConcat, groupsToCreateSpec)
	if err != nil {
		return nil, err
	}
	return &UsersAndGroups{Users: mapUsers, Groups: mapGroups}, nil
}

// RemoveUsersAndGroups is a public wrapper around removeUsersAndGroupsImpl with
// default system file paths.
func RemoveUsersAndGroups() error {
	return removeUsersAndGroupsImpl(fileEtcPasswd, fileEtcGroup)
}

// removeUsersAndGroupsImpl removes the managed users and groups from the files specified
// by pathUsers and pathGroups.
func removeUsersAndGroupsImpl(pathUsers, pathGroups string) error {
	klog.V(1).Info("Removing managed users and groups")
	klog.V(klogLevel).Infof("Opening %q and %q", pathUsers, pathGroups)

	// Open /etc/passwd and /etc/group.
	fUsers, close, err := openFileWithLock(pathUsers)
	if err != nil {
		return err
	}
	defer close()
	fGroups, close, err := openFileWithLock(pathGroups)
	if err != nil {
		return err
	}
	defer close()

	// Read the files.
	fileUsers, err := readFile(fUsers)
	if err != nil {
		return err
	}
	fileGroups, err := readFile(fGroups)
	if err != nil {
		return err
	}

	klog.V(klogLevel).Infof("Removing users: %s", entriesToString(usersToCreateSpec))
	klog.V(klogLevel).Infof("Removing groups: %s", entriesToString(groupsToCreateSpec))

	// Delete users / groups.
	fileUsers, _ = removeEntries(fileUsers, usersToCreateSpec)
	fileGroups, _ = removeEntries(fileGroups, groupsToCreateSpec)

	klog.V(klogLevel).Infof("Writing %q and %q", pathUsers, pathGroups)

	// Write the files.
	if err := writeFile(fUsers, fileUsers); err != nil {
		return err
	}
	if err := writeFile(fGroups, fileGroups); err != nil {
		return err
	}

	return nil
}

// parseLoginDefs can be used to parse an /etc/login.defs file and obtain system ranges for UID and GID.
// Passing an empty string will return the defaults. The defaults are 100-999 for both UID and GID.
func parseLoginDefs(file string) (*limits, error) {
	l := *defaultLimits
	if len(file) == 0 {
		return &l, nil
	}
	var mapping = map[string]*int64{
		"SYS_UID_MIN": &l.minUID,
		"SYS_UID_MAX": &l.maxUID,
		"SYS_GID_MIN": &l.minGID,
		"SYS_GID_MAX": &l.maxGID,
	}
	lines := strings.Split(file, "\n")
	for i, line := range lines {
		for k, v := range mapping {
			// A line must start with one of the definitions
			if !strings.HasPrefix(line, k) {
				continue
			}
			line = strings.TrimPrefix(line, k)
			line = strings.TrimSpace(line)
			val, err := strconv.ParseInt(line, 10, 64)
			if err != nil {
				return nil, errors.Wrapf(err, "could not parse value for %s at line %d", k, i)
			}
			*v = val
		}
	}
	return &l, nil
}

// parseEntries can be used to parse an /etc/passwd or /etc/group file as their format is similar.
// It returns a slice of entries obtained from the file.
// https://www.cyberciti.biz/faq/understanding-etcpasswd-file-format/
// https://www.cyberciti.biz/faq/understanding-etcgroup-file/
func parseEntries(file string, totalFields int) ([]*entry, error) {
	if totalFields != totalFieldsUser && totalFields != totalFieldsGroup {
		return nil, errors.Errorf("unsupported total fields for entry parsing: %d", totalFields)
	}
	lines := strings.Split(file, "\n")
	entries := []*entry{}
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		fields := strings.Split(line, ":")
		if len(fields) != totalFields {
			return nil, errors.Errorf("entry must have %d fields separated by ':', "+
				"got %d at line %d: %s", totalFields, len(fields), i, line)
		}
		id, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil {
			return nil, errors.Wrapf(err, "error parsing id at line %d", i)
		}
		entry := &entry{name: fields[0], id: id}
		if totalFields == totalFieldsGroup {
			entry.userNames = strings.Split(fields[3], ",")
		} else {
			gid, err := strconv.ParseInt(fields[3], 10, 64)
			if err != nil {
				return nil, errors.Wrapf(err, "error parsing GID at line %d", i)
			}
			entry.gid = gid
			entry.shell = fields[6]
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// validateEntries takes user and group entries and validates if these entries are valid based on limits,
// mapping between users and groups and specs. Returns slices of missing user and group entries that must be created.
// Returns an error if existing users and groups do not match requirements.
func validateEntries(users, groups []*entry, limits *limits) ([]*entry, []*entry, error) {
	u := []*entry{}
	g := []*entry{}
	// Validate users
	for _, uc := range usersToCreateSpec {
		for _, user := range users {
			if uc.name != user.name {
				continue
			}
			// Found existing user
			if user.id < limits.minUID || user.id > limits.maxUID {
				return nil, nil, errors.Errorf("UID %d for user %q is outside the system UID range: %d - %d",
					user.id, user.name, limits.minUID, limits.maxUID)
			}
			if user.shell != noshell {
				return nil, nil, errors.Errorf("user %q has unexpected shell %q; expected %q",
					user.name, user.shell, noshell)
			}
			for _, g := range groups {
				if g.id != user.gid {
					continue
				}
				// Found matching group GID for user GID
				if g.name != uc.name {
					return nil, nil, errors.Errorf("user %q has GID %d but the group with that GID is not named %q",
						uc.name, g.id, uc.name)
				}
				goto skipUser // Valid group GID and name; skip
			}
			return nil, nil, errors.Errorf("could not find group with GID %d for user %q", user.gid, user.name)
		}
		u = append(u, uc)
	skipUser:
	}
	// validate groups
	for _, gc := range groupsToCreateSpec {
		for _, group := range groups {
			if gc.name != group.name {
				continue
			}
			if group.id < limits.minGID || group.id > limits.maxGID {
				return nil, nil, errors.Errorf("GID %d for user %q is outside the system UID range: %d - %d",
					group.id, group.name, limits.minGID, limits.maxGID)
			}
			u1 := strings.Join(gc.userNames, ",")
			u2 := strings.Join(group.userNames, ",")
			if u1 != u2 {
				return nil, nil, errors.Errorf("expected users %q for group %q; got %q",
					u1, gc.name, u2)
			}
			goto skipGroup // group has valid users; skip
		}
		g = append(g, gc)
	skipGroup:
	}
	return u, g, nil
}

// allocateIDs takes a list of entries and based on minimum and maximum ID allocates a "total" of IDs.
func allocateIDs(entries []*entry, min, max int64, total int) ([]int64, error) {
	if total == 0 {
		return []int64{}, nil
	}
	ids := make([]int64, 0, total)
	for i := min; i < max+1; i++ {
		i64 := int64(i)
		for _, e := range entries {
			if i64 == e.id {
				goto continueLoop
			}
		}
		ids = append(ids, i64)
		if len(ids) == total {
			return ids, nil
		}
	continueLoop:
	}
	return nil, errors.Errorf("could not allocate %d IDs based on existing entries in the range: %d - %d",
		total, min, max)
}

// addEntries takes /etc/passwd or /etc/group file content and appends entries to it based
// on a createEntry function. Returns the updated contents of the file.
func addEntries(file string, entries []*entry, createEntry func(*entry) string) string {
	out := file
	newLines := make([]string, 0, len(entries))
	for _, e := range entries {
		newLines = append(newLines, createEntry(e))
	}
	newLinesStr := ""
	if len(newLines) > 0 {
		if !strings.HasSuffix(out, "\n") { // Append a new line if its missing.
			newLinesStr = "\n"
		}
		newLinesStr += strings.Join(newLines, "\n") + "\n"
	}
	return out + newLinesStr
}

// removeEntries takes /etc/passwd or /etc/group file content and deletes entries from them
// by name matching. Returns the updated contents of the file and the number of entries removed.
func removeEntries(file string, entries []*entry) (string, int) {
	lines := strings.Split(file, "\n")
	total := len(lines) - len(entries)
	if total < 0 {
		total = 0
	}
	newLines := make([]string, 0, total)
	removed := 0
	for _, line := range lines {
		for _, entry := range entries {
			if strings.HasPrefix(line, entry.name+":") {
				removed++
				goto continueLoop
			}
		}
		newLines = append(newLines, line)
	continueLoop:
	}
	return strings.Join(newLines, "\n"), removed
}

// assignUserAndGroupIDs takes the list of existing groups, the users and groups to be created,
// and assigns UIDs and GIDs to the users and groups to be created based on a list of provided UIDs and GIDs.
// Returns an error if not enough UIDs or GIDs are passed. It does not perform any other validation.
func assignUserAndGroupIDs(groups, usersToCreate, groupsToCreate []*entry, uids, gids []int64) error {
	if len(gids) < len(groupsToCreate) {
		return errors.Errorf("not enough GIDs to assign to groups: have %d, want %d", len(gids), len(groupsToCreate))
	}
	if len(uids) < len(usersToCreate) {
		return errors.Errorf("not enough UIDs to assign to users: have %d, want %d", len(uids), len(usersToCreate))
	}
	for i := range groupsToCreate {
		groupsToCreate[i].id = gids[i]
	}
	// Concat the list of old and new groups to find a matching GID.
	groupsConcat := append([]*entry{}, groups...)
	groupsConcat = append(groupsConcat, groupsToCreate...)
	for i := range usersToCreate {
		usersToCreate[i].id = uids[i]
		for _, g := range groupsConcat {
			if usersToCreate[i].name == g.name {
				usersToCreate[i].gid = g.id
				break
			}
		}
	}
	return nil
}

// createGroup is a helper function to produce a group from entry.
func createGroup(e *entry) string {
	return fmt.Sprintf("%s:x:%d:%s", e.name, e.id, strings.Join(e.userNames, ","))
}

// createUser is a helper function to produce a user from entry.
func createUser(e *entry) string {
	return fmt.Sprintf("%s:x:%d:%d:::/bin/false", e.name, e.id, e.gid)
}

// entriesToEntryMap takes a list of entries and prepares an EntryMap object.
func entriesToEntryMap(entries, spec []*entry) (*EntryMap, error) {
	m := map[string]*entry{}
	for _, spec := range spec {
		for _, e := range entries {
			if spec.name == e.name {
				entry := *e
				m[e.name] = &entry
				goto continueLoop
			}
		}
		return nil, errors.Errorf("could not find entry %q in the list", spec.name)
	continueLoop:
	}
	return &EntryMap{entries: m}, nil
}

// entriesToString is a utility to convert a list of entries to string.
func entriesToString(entries []*entry) string {
	lines := make([]string, 0, len(entries))
	for _, e := range entries {
		lines = append(lines, e.name)
	}
	sort.Strings(lines)
	return strings.Join(lines, ",")
}

// openFileWithLock opens the file at path by acquiring an exclive write lock.
// The returned close() function should be called to release the lock and close the file.
// If a lock cannot be obtained the function fails after a period of time.
func openFileWithLock(path string) (f *os.File, close func(), err error) {
	f, err = os.OpenFile(path, os.O_RDWR, os.ModePerm)
	if err != nil {
		return nil, nil, err
	}
	deadline := time.Now().Add(time.Second * 5)
	for {
		// If another process is holding a write lock, this call will exit
		// with an error. F_SETLK is used instead of F_SETLKW to avoid
		// the case where a runaway process grabs the exclusive lock and
		// blocks this call indefinitely.
		// https://man7.org/linux/man-pages/man2/fcntl.2.html
		lock := syscall.Flock_t{Type: syscall.F_WRLCK}
		if err = syscall.FcntlFlock(f.Fd(), syscall.F_SETLK, &lock); err == nil {
			break
		}
		time.Sleep(200 * time.Millisecond)
		if time.Now().After(deadline) {
			err = errors.Wrapf(err, "timeout attempting to obtain lock on file %q", path)
			break
		}
	}
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	close = func() {
		// This function should be called once operations with the file are finished.
		// It unlocks the file and closes it.
		unlock := syscall.Flock_t{Type: syscall.F_UNLCK}
		syscall.FcntlFlock(f.Fd(), syscall.F_SETLK, &unlock)
		f.Close()
	}
	return f, close, nil
}

// readFile reads a File into a string.
func readFile(f *os.File) (string, error) {
	buf := bytes.NewBuffer(nil)
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	if _, err := io.Copy(buf, f); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// writeFile writes a string to a File.
func writeFile(f *os.File, str string) error {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if _, err := f.Write([]byte(str)); err != nil {
		return err
	}
	if err := f.Truncate(int64(len(str))); err != nil {
		return err
	}
	return nil
}

// UpdatePathOwnerAndPermissions updates the owner and permissions of the given path.
// If the path is a directory it is not recursively updated.
func UpdatePathOwnerAndPermissions(path string, uid, gid int64, perms uint32) error {
	if err := os.Chown(path, int(uid), int(gid)); err != nil {
		return errors.Wrapf(err, "failed to update owner of %q to uid: %d and gid: %d", path, uid, gid)
	}
	fm := os.FileMode(perms)
	if err := os.Chmod(path, fm); err != nil {
		return errors.Wrapf(err, "failed to update permissions of %q to %s", path, fm.String())
	}
	return nil
}

// UpdatePathOwner recursively updates the owners of a directory.
// It is equivalent to calling `chown -R uid:gid /path/to/dir`.
func UpdatePathOwner(dirPath string, uid, gid int64) error {
	err := filepath.WalkDir(dirPath, func(path string, d os.DirEntry, err error) error {
		if err := os.Chown(path, int(uid), int(gid)); err != nil {
			return errors.Wrapf(err, "failed to update owner of %q to uid: %d and gid: %d", path, uid, gid)
		}
		return nil
	})
	return err
}
