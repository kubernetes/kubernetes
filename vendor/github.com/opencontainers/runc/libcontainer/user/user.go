package user

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

const (
	minID = 0
	maxID = 1<<31 - 1 // for 32-bit systems compatibility
)

var (
	// ErrNoPasswdEntries is returned if no matching entries were found in /etc/group.
	ErrNoPasswdEntries = errors.New("no matching entries in passwd file")
	// ErrNoGroupEntries is returned if no matching entries were found in /etc/passwd.
	ErrNoGroupEntries = errors.New("no matching entries in group file")
	// ErrRange is returned if a UID or GID is outside of the valid range.
	ErrRange = fmt.Errorf("uids and gids must be in range %d-%d", minID, maxID)
)

type User struct {
	Name  string
	Pass  string
	Uid   int
	Gid   int
	Gecos string
	Home  string
	Shell string
}

type Group struct {
	Name string
	Pass string
	Gid  int
	List []string
}

// SubID represents an entry in /etc/sub{u,g}id
type SubID struct {
	Name  string
	SubID int64
	Count int64
}

// IDMap represents an entry in /proc/PID/{u,g}id_map
type IDMap struct {
	ID       int64
	ParentID int64
	Count    int64
}

func parseLine(line []byte, v ...interface{}) {
	parseParts(bytes.Split(line, []byte(":")), v...)
}

func parseParts(parts [][]byte, v ...interface{}) {
	if len(parts) == 0 {
		return
	}

	for i, p := range parts {
		// Ignore cases where we don't have enough fields to populate the arguments.
		// Some configuration files like to misbehave.
		if len(v) <= i {
			break
		}

		// Use the type of the argument to figure out how to parse it, scanf() style.
		// This is legit.
		switch e := v[i].(type) {
		case *string:
			*e = string(p)
		case *int:
			// "numbers", with conversion errors ignored because of some misbehaving configuration files.
			*e, _ = strconv.Atoi(string(p))
		case *int64:
			*e, _ = strconv.ParseInt(string(p), 10, 64)
		case *[]string:
			// Comma-separated lists.
			if len(p) != 0 {
				*e = strings.Split(string(p), ",")
			} else {
				*e = []string{}
			}
		default:
			// Someone goof'd when writing code using this function. Scream so they can hear us.
			panic(fmt.Sprintf("parseLine only accepts {*string, *int, *int64, *[]string} as arguments! %#v is not a pointer!", e))
		}
	}
}

func ParsePasswdFile(path string) ([]User, error) {
	passwd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer passwd.Close()
	return ParsePasswd(passwd)
}

func ParsePasswd(passwd io.Reader) ([]User, error) {
	return ParsePasswdFilter(passwd, nil)
}

func ParsePasswdFileFilter(path string, filter func(User) bool) ([]User, error) {
	passwd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer passwd.Close()
	return ParsePasswdFilter(passwd, filter)
}

func ParsePasswdFilter(r io.Reader, filter func(User) bool) ([]User, error) {
	if r == nil {
		return nil, fmt.Errorf("nil source for passwd-formatted data")
	}

	var (
		s   = bufio.NewScanner(r)
		out = []User{}
	)

	for s.Scan() {
		line := bytes.TrimSpace(s.Bytes())
		if len(line) == 0 {
			continue
		}

		// see: man 5 passwd
		//  name:password:UID:GID:GECOS:directory:shell
		// Name:Pass:Uid:Gid:Gecos:Home:Shell
		//  root:x:0:0:root:/root:/bin/bash
		//  adm:x:3:4:adm:/var/adm:/bin/false
		p := User{}
		parseLine(line, &p.Name, &p.Pass, &p.Uid, &p.Gid, &p.Gecos, &p.Home, &p.Shell)

		if filter == nil || filter(p) {
			out = append(out, p)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

func ParseGroupFile(path string) ([]Group, error) {
	group, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	defer group.Close()
	return ParseGroup(group)
}

func ParseGroup(group io.Reader) ([]Group, error) {
	return ParseGroupFilter(group, nil)
}

func ParseGroupFileFilter(path string, filter func(Group) bool) ([]Group, error) {
	group, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer group.Close()
	return ParseGroupFilter(group, filter)
}

func ParseGroupFilter(r io.Reader, filter func(Group) bool) ([]Group, error) {
	if r == nil {
		return nil, fmt.Errorf("nil source for group-formatted data")
	}
	rd := bufio.NewReader(r)
	out := []Group{}

	// Read the file line-by-line.
	for {
		var (
			isPrefix  bool
			wholeLine []byte
			err       error
		)

		// Read the next line. We do so in chunks (as much as reader's
		// buffer is able to keep), check if we read enough columns
		// already on each step and store final result in wholeLine.
		for {
			var line []byte
			line, isPrefix, err = rd.ReadLine()

			if err != nil {
				// We should return no error if EOF is reached
				// without a match.
				if err == io.EOF { //nolint:errorlint // comparison with io.EOF is legit, https://github.com/polyfloyd/go-errorlint/pull/12
					err = nil
				}
				return out, err
			}

			// Simple common case: line is short enough to fit in a
			// single reader's buffer.
			if !isPrefix && len(wholeLine) == 0 {
				wholeLine = line
				break
			}

			wholeLine = append(wholeLine, line...)

			// Check if we read the whole line already.
			if !isPrefix {
				break
			}
		}

		// There's no spec for /etc/passwd or /etc/group, but we try to follow
		// the same rules as the glibc parser, which allows comments and blank
		// space at the beginning of a line.
		wholeLine = bytes.TrimSpace(wholeLine)
		if len(wholeLine) == 0 || wholeLine[0] == '#' {
			continue
		}

		// see: man 5 group
		//  group_name:password:GID:user_list
		// Name:Pass:Gid:List
		//  root:x:0:root
		//  adm:x:4:root,adm,daemon
		p := Group{}
		parseLine(wholeLine, &p.Name, &p.Pass, &p.Gid, &p.List)

		if filter == nil || filter(p) {
			out = append(out, p)
		}
	}
}

type ExecUser struct {
	Uid   int
	Gid   int
	Sgids []int
	Home  string
}

// GetExecUserPath is a wrapper for GetExecUser. It reads data from each of the
// given file paths and uses that data as the arguments to GetExecUser. If the
// files cannot be opened for any reason, the error is ignored and a nil
// io.Reader is passed instead.
func GetExecUserPath(userSpec string, defaults *ExecUser, passwdPath, groupPath string) (*ExecUser, error) {
	var passwd, group io.Reader

	if passwdFile, err := os.Open(passwdPath); err == nil {
		passwd = passwdFile
		defer passwdFile.Close()
	}

	if groupFile, err := os.Open(groupPath); err == nil {
		group = groupFile
		defer groupFile.Close()
	}

	return GetExecUser(userSpec, defaults, passwd, group)
}

// GetExecUser parses a user specification string (using the passwd and group
// readers as sources for /etc/passwd and /etc/group data, respectively). In
// the case of blank fields or missing data from the sources, the values in
// defaults is used.
//
// GetExecUser will return an error if a user or group literal could not be
// found in any entry in passwd and group respectively.
//
// Examples of valid user specifications are:
//     * ""
//     * "user"
//     * "uid"
//     * "user:group"
//     * "uid:gid
//     * "user:gid"
//     * "uid:group"
//
// It should be noted that if you specify a numeric user or group id, they will
// not be evaluated as usernames (only the metadata will be filled). So attempting
// to parse a user with user.Name = "1337" will produce the user with a UID of
// 1337.
func GetExecUser(userSpec string, defaults *ExecUser, passwd, group io.Reader) (*ExecUser, error) {
	if defaults == nil {
		defaults = new(ExecUser)
	}

	// Copy over defaults.
	user := &ExecUser{
		Uid:   defaults.Uid,
		Gid:   defaults.Gid,
		Sgids: defaults.Sgids,
		Home:  defaults.Home,
	}

	// Sgids slice *cannot* be nil.
	if user.Sgids == nil {
		user.Sgids = []int{}
	}

	// Allow for userArg to have either "user" syntax, or optionally "user:group" syntax
	var userArg, groupArg string
	parseLine([]byte(userSpec), &userArg, &groupArg)

	// Convert userArg and groupArg to be numeric, so we don't have to execute
	// Atoi *twice* for each iteration over lines.
	uidArg, uidErr := strconv.Atoi(userArg)
	gidArg, gidErr := strconv.Atoi(groupArg)

	// Find the matching user.
	users, err := ParsePasswdFilter(passwd, func(u User) bool {
		if userArg == "" {
			// Default to current state of the user.
			return u.Uid == user.Uid
		}

		if uidErr == nil {
			// If the userArg is numeric, always treat it as a UID.
			return uidArg == u.Uid
		}

		return u.Name == userArg
	})

	// If we can't find the user, we have to bail.
	if err != nil && passwd != nil {
		if userArg == "" {
			userArg = strconv.Itoa(user.Uid)
		}
		return nil, fmt.Errorf("unable to find user %s: %v", userArg, err)
	}

	var matchedUserName string
	if len(users) > 0 {
		// First match wins, even if there's more than one matching entry.
		matchedUserName = users[0].Name
		user.Uid = users[0].Uid
		user.Gid = users[0].Gid
		user.Home = users[0].Home
	} else if userArg != "" {
		// If we can't find a user with the given username, the only other valid
		// option is if it's a numeric username with no associated entry in passwd.

		if uidErr != nil {
			// Not numeric.
			return nil, fmt.Errorf("unable to find user %s: %v", userArg, ErrNoPasswdEntries)
		}
		user.Uid = uidArg

		// Must be inside valid uid range.
		if user.Uid < minID || user.Uid > maxID {
			return nil, ErrRange
		}

		// Okay, so it's numeric. We can just roll with this.
	}

	// On to the groups. If we matched a username, we need to do this because of
	// the supplementary group IDs.
	if groupArg != "" || matchedUserName != "" {
		groups, err := ParseGroupFilter(group, func(g Group) bool {
			// If the group argument isn't explicit, we'll just search for it.
			if groupArg == "" {
				// Check if user is a member of this group.
				for _, u := range g.List {
					if u == matchedUserName {
						return true
					}
				}
				return false
			}

			if gidErr == nil {
				// If the groupArg is numeric, always treat it as a GID.
				return gidArg == g.Gid
			}

			return g.Name == groupArg
		})
		if err != nil && group != nil {
			return nil, fmt.Errorf("unable to find groups for spec %v: %v", matchedUserName, err)
		}

		// Only start modifying user.Gid if it is in explicit form.
		if groupArg != "" {
			if len(groups) > 0 {
				// First match wins, even if there's more than one matching entry.
				user.Gid = groups[0].Gid
			} else {
				// If we can't find a group with the given name, the only other valid
				// option is if it's a numeric group name with no associated entry in group.

				if gidErr != nil {
					// Not numeric.
					return nil, fmt.Errorf("unable to find group %s: %v", groupArg, ErrNoGroupEntries)
				}
				user.Gid = gidArg

				// Must be inside valid gid range.
				if user.Gid < minID || user.Gid > maxID {
					return nil, ErrRange
				}

				// Okay, so it's numeric. We can just roll with this.
			}
		} else if len(groups) > 0 {
			// Supplementary group ids only make sense if in the implicit form.
			user.Sgids = make([]int, len(groups))
			for i, group := range groups {
				user.Sgids[i] = group.Gid
			}
		}
	}

	return user, nil
}

// GetAdditionalGroups looks up a list of groups by name or group id
// against the given /etc/group formatted data. If a group name cannot
// be found, an error will be returned. If a group id cannot be found,
// or the given group data is nil, the id will be returned as-is
// provided it is in the legal range.
func GetAdditionalGroups(additionalGroups []string, group io.Reader) ([]int, error) {
	groups := []Group{}
	if group != nil {
		var err error
		groups, err = ParseGroupFilter(group, func(g Group) bool {
			for _, ag := range additionalGroups {
				if g.Name == ag || strconv.Itoa(g.Gid) == ag {
					return true
				}
			}
			return false
		})
		if err != nil {
			return nil, fmt.Errorf("Unable to find additional groups %v: %v", additionalGroups, err)
		}
	}

	gidMap := make(map[int]struct{})
	for _, ag := range additionalGroups {
		var found bool
		for _, g := range groups {
			// if we found a matched group either by name or gid, take the
			// first matched as correct
			if g.Name == ag || strconv.Itoa(g.Gid) == ag {
				if _, ok := gidMap[g.Gid]; !ok {
					gidMap[g.Gid] = struct{}{}
					found = true
					break
				}
			}
		}
		// we asked for a group but didn't find it. let's check to see
		// if we wanted a numeric group
		if !found {
			gid, err := strconv.ParseInt(ag, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("Unable to find group %s", ag)
			}
			// Ensure gid is inside gid range.
			if gid < minID || gid > maxID {
				return nil, ErrRange
			}
			gidMap[int(gid)] = struct{}{}
		}
	}
	gids := []int{}
	for gid := range gidMap {
		gids = append(gids, gid)
	}
	return gids, nil
}

// GetAdditionalGroupsPath is a wrapper around GetAdditionalGroups
// that opens the groupPath given and gives it as an argument to
// GetAdditionalGroups.
func GetAdditionalGroupsPath(additionalGroups []string, groupPath string) ([]int, error) {
	var group io.Reader

	if groupFile, err := os.Open(groupPath); err == nil {
		group = groupFile
		defer groupFile.Close()
	}
	return GetAdditionalGroups(additionalGroups, group)
}

func ParseSubIDFile(path string) ([]SubID, error) {
	subid, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer subid.Close()
	return ParseSubID(subid)
}

func ParseSubID(subid io.Reader) ([]SubID, error) {
	return ParseSubIDFilter(subid, nil)
}

func ParseSubIDFileFilter(path string, filter func(SubID) bool) ([]SubID, error) {
	subid, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer subid.Close()
	return ParseSubIDFilter(subid, filter)
}

func ParseSubIDFilter(r io.Reader, filter func(SubID) bool) ([]SubID, error) {
	if r == nil {
		return nil, fmt.Errorf("nil source for subid-formatted data")
	}

	var (
		s   = bufio.NewScanner(r)
		out = []SubID{}
	)

	for s.Scan() {
		line := bytes.TrimSpace(s.Bytes())
		if len(line) == 0 {
			continue
		}

		// see: man 5 subuid
		p := SubID{}
		parseLine(line, &p.Name, &p.SubID, &p.Count)

		if filter == nil || filter(p) {
			out = append(out, p)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

func ParseIDMapFile(path string) ([]IDMap, error) {
	r, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return ParseIDMap(r)
}

func ParseIDMap(r io.Reader) ([]IDMap, error) {
	return ParseIDMapFilter(r, nil)
}

func ParseIDMapFileFilter(path string, filter func(IDMap) bool) ([]IDMap, error) {
	r, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return ParseIDMapFilter(r, filter)
}

func ParseIDMapFilter(r io.Reader, filter func(IDMap) bool) ([]IDMap, error) {
	if r == nil {
		return nil, fmt.Errorf("nil source for idmap-formatted data")
	}

	var (
		s   = bufio.NewScanner(r)
		out = []IDMap{}
	)

	for s.Scan() {
		line := bytes.TrimSpace(s.Bytes())
		if len(line) == 0 {
			continue
		}

		// see: man 7 user_namespaces
		p := IDMap{}
		parseParts(bytes.Fields(line), &p.ID, &p.ParentID, &p.Count)

		if filter == nil || filter(p) {
			out = append(out, p)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return out, nil
}
