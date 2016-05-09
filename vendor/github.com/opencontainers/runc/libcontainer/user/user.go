package user

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

const (
	minId = 0
	maxId = 1<<31 - 1 //for 32-bit systems compatibility
)

var (
	ErrRange = fmt.Errorf("Uids and gids must be in range %d-%d", minId, maxId)
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

func parseLine(line string, v ...interface{}) {
	if line == "" {
		return
	}

	parts := strings.Split(line, ":")
	for i, p := range parts {
		if len(v) <= i {
			// if we have more "parts" than we have places to put them, bail for great "tolerance" of naughty configuration files
			break
		}

		switch e := v[i].(type) {
		case *string:
			// "root", "adm", "/bin/bash"
			*e = p
		case *int:
			// "0", "4", "1000"
			// ignore string to int conversion errors, for great "tolerance" of naughty configuration files
			*e, _ = strconv.Atoi(p)
		case *[]string:
			// "", "root", "root,adm,daemon"
			if p != "" {
				*e = strings.Split(p, ",")
			} else {
				*e = []string{}
			}
		default:
			// panic, because this is a programming/logic error, not a runtime one
			panic("parseLine expects only pointers!  argument " + strconv.Itoa(i) + " is not a pointer!")
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
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := strings.TrimSpace(s.Text())
		if text == "" {
			continue
		}

		// see: man 5 passwd
		//  name:password:UID:GID:GECOS:directory:shell
		// Name:Pass:Uid:Gid:Gecos:Home:Shell
		//  root:x:0:0:root:/root:/bin/bash
		//  adm:x:3:4:adm:/var/adm:/bin/false
		p := User{}
		parseLine(
			text,
			&p.Name, &p.Pass, &p.Uid, &p.Gid, &p.Gecos, &p.Home, &p.Shell,
		)

		if filter == nil || filter(p) {
			out = append(out, p)
		}
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

	var (
		s   = bufio.NewScanner(r)
		out = []Group{}
	)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		text := s.Text()
		if text == "" {
			continue
		}

		// see: man 5 group
		//  group_name:password:GID:user_list
		// Name:Pass:Gid:List
		//  root:x:0:root
		//  adm:x:4:root,adm,daemon
		p := Group{}
		parseLine(
			text,
			&p.Name, &p.Pass, &p.Gid, &p.List,
		)

		if filter == nil || filter(p) {
			out = append(out, p)
		}
	}

	return out, nil
}

type ExecUser struct {
	Uid, Gid int
	Sgids    []int
	Home     string
}

// GetExecUserPath is a wrapper for GetExecUser. It reads data from each of the
// given file paths and uses that data as the arguments to GetExecUser. If the
// files cannot be opened for any reason, the error is ignored and a nil
// io.Reader is passed instead.
func GetExecUserPath(userSpec string, defaults *ExecUser, passwdPath, groupPath string) (*ExecUser, error) {
	passwd, err := os.Open(passwdPath)
	if err != nil {
		passwd = nil
	} else {
		defer passwd.Close()
	}

	group, err := os.Open(groupPath)
	if err != nil {
		group = nil
	} else {
		defer group.Close()
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
func GetExecUser(userSpec string, defaults *ExecUser, passwd, group io.Reader) (*ExecUser, error) {
	var (
		userArg, groupArg string
		name              string
	)

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

	// allow for userArg to have either "user" syntax, or optionally "user:group" syntax
	parseLine(userSpec, &userArg, &groupArg)

	users, err := ParsePasswdFilter(passwd, func(u User) bool {
		if userArg == "" {
			return u.Uid == user.Uid
		}
		return u.Name == userArg || strconv.Itoa(u.Uid) == userArg
	})
	if err != nil && passwd != nil {
		if userArg == "" {
			userArg = strconv.Itoa(user.Uid)
		}
		return nil, fmt.Errorf("Unable to find user %v: %v", userArg, err)
	}

	haveUser := users != nil && len(users) > 0
	if haveUser {
		// if we found any user entries that matched our filter, let's take the first one as "correct"
		name = users[0].Name
		user.Uid = users[0].Uid
		user.Gid = users[0].Gid
		user.Home = users[0].Home
	} else if userArg != "" {
		// we asked for a user but didn't find them...  let's check to see if we wanted a numeric user
		user.Uid, err = strconv.Atoi(userArg)
		if err != nil {
			// not numeric - we have to bail
			return nil, fmt.Errorf("Unable to find user %v", userArg)
		}

		// Must be inside valid uid range.
		if user.Uid < minId || user.Uid > maxId {
			return nil, ErrRange
		}

		// if userArg couldn't be found in /etc/passwd but is numeric, just roll with it - this is legit
	}

	if groupArg != "" || name != "" {
		groups, err := ParseGroupFilter(group, func(g Group) bool {
			// Explicit group format takes precedence.
			if groupArg != "" {
				return g.Name == groupArg || strconv.Itoa(g.Gid) == groupArg
			}

			// Check if user is a member.
			for _, u := range g.List {
				if u == name {
					return true
				}
			}

			return false
		})
		if err != nil && group != nil {
			return nil, fmt.Errorf("Unable to find groups for user %v: %v", users[0].Name, err)
		}

		haveGroup := groups != nil && len(groups) > 0
		if groupArg != "" {
			if haveGroup {
				// if we found any group entries that matched our filter, let's take the first one as "correct"
				user.Gid = groups[0].Gid
			} else {
				// we asked for a group but didn't find id...  let's check to see if we wanted a numeric group
				user.Gid, err = strconv.Atoi(groupArg)
				if err != nil {
					// not numeric - we have to bail
					return nil, fmt.Errorf("Unable to find group %v", groupArg)
				}

				// Ensure gid is inside gid range.
				if user.Gid < minId || user.Gid > maxId {
					return nil, ErrRange
				}

				// if groupArg couldn't be found in /etc/group but is numeric, just roll with it - this is legit
			}
		} else if haveGroup {
			// If implicit group format, fill supplementary gids.
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
	var groups = []Group{}
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
			gid, err := strconv.Atoi(ag)
			if err != nil {
				return nil, fmt.Errorf("Unable to find group %s", ag)
			}
			// Ensure gid is inside gid range.
			if gid < minId || gid > maxId {
				return nil, ErrRange
			}
			gidMap[gid] = struct{}{}
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
	group, err := os.Open(groupPath)
	if err == nil {
		defer group.Close()
	}
	return GetAdditionalGroups(additionalGroups, group)
}
