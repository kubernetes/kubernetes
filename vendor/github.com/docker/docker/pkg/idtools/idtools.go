package idtools

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// IDMap contains a single entry for user namespace range remapping. An array
// of IDMap entries represents the structure that will be provided to the Linux
// kernel for creating a user namespace.
type IDMap struct {
	ContainerID int `json:"container_id"`
	HostID      int `json:"host_id"`
	Size        int `json:"size"`
}

type subIDRange struct {
	Start  int
	Length int
}

type ranges []subIDRange

func (e ranges) Len() int           { return len(e) }
func (e ranges) Swap(i, j int)      { e[i], e[j] = e[j], e[i] }
func (e ranges) Less(i, j int) bool { return e[i].Start < e[j].Start }

const (
	subuidFileName string = "/etc/subuid"
	subgidFileName string = "/etc/subgid"
)

// MkdirAllAs creates a directory (include any along the path) and then modifies
// ownership to the requested uid/gid.  If the directory already exists, this
// function will still change ownership to the requested uid/gid pair.
// Deprecated: Use MkdirAllAndChown
func MkdirAllAs(path string, mode os.FileMode, ownerUID, ownerGID int) error {
	return mkdirAs(path, mode, ownerUID, ownerGID, true, true)
}

// MkdirAs creates a directory and then modifies ownership to the requested uid/gid.
// If the directory already exists, this function still changes ownership
// Deprecated: Use MkdirAndChown with a IDPair
func MkdirAs(path string, mode os.FileMode, ownerUID, ownerGID int) error {
	return mkdirAs(path, mode, ownerUID, ownerGID, false, true)
}

// MkdirAllAndChown creates a directory (include any along the path) and then modifies
// ownership to the requested uid/gid.  If the directory already exists, this
// function will still change ownership to the requested uid/gid pair.
func MkdirAllAndChown(path string, mode os.FileMode, ids IDPair) error {
	return mkdirAs(path, mode, ids.UID, ids.GID, true, true)
}

// MkdirAndChown creates a directory and then modifies ownership to the requested uid/gid.
// If the directory already exists, this function still changes ownership
func MkdirAndChown(path string, mode os.FileMode, ids IDPair) error {
	return mkdirAs(path, mode, ids.UID, ids.GID, false, true)
}

// MkdirAllAndChownNew creates a directory (include any along the path) and then modifies
// ownership ONLY of newly created directories to the requested uid/gid. If the
// directories along the path exist, no change of ownership will be performed
func MkdirAllAndChownNew(path string, mode os.FileMode, ids IDPair) error {
	return mkdirAs(path, mode, ids.UID, ids.GID, true, false)
}

// GetRootUIDGID retrieves the remapped root uid/gid pair from the set of maps.
// If the maps are empty, then the root uid/gid will default to "real" 0/0
func GetRootUIDGID(uidMap, gidMap []IDMap) (int, int, error) {
	uid, err := toHost(0, uidMap)
	if err != nil {
		return -1, -1, err
	}
	gid, err := toHost(0, gidMap)
	if err != nil {
		return -1, -1, err
	}
	return uid, gid, nil
}

// toContainer takes an id mapping, and uses it to translate a
// host ID to the remapped ID. If no map is provided, then the translation
// assumes a 1-to-1 mapping and returns the passed in id
func toContainer(hostID int, idMap []IDMap) (int, error) {
	if idMap == nil {
		return hostID, nil
	}
	for _, m := range idMap {
		if (hostID >= m.HostID) && (hostID <= (m.HostID + m.Size - 1)) {
			contID := m.ContainerID + (hostID - m.HostID)
			return contID, nil
		}
	}
	return -1, fmt.Errorf("Host ID %d cannot be mapped to a container ID", hostID)
}

// toHost takes an id mapping and a remapped ID, and translates the
// ID to the mapped host ID. If no map is provided, then the translation
// assumes a 1-to-1 mapping and returns the passed in id #
func toHost(contID int, idMap []IDMap) (int, error) {
	if idMap == nil {
		return contID, nil
	}
	for _, m := range idMap {
		if (contID >= m.ContainerID) && (contID <= (m.ContainerID + m.Size - 1)) {
			hostID := m.HostID + (contID - m.ContainerID)
			return hostID, nil
		}
	}
	return -1, fmt.Errorf("Container ID %d cannot be mapped to a host ID", contID)
}

// IDPair is a UID and GID pair
type IDPair struct {
	UID int
	GID int
}

// IDMappings contains a mappings of UIDs and GIDs
type IDMappings struct {
	uids []IDMap
	gids []IDMap
}

// NewIDMappings takes a requested user and group name and
// using the data from /etc/sub{uid,gid} ranges, creates the
// proper uid and gid remapping ranges for that user/group pair
func NewIDMappings(username, groupname string) (*IDMappings, error) {
	subuidRanges, err := parseSubuid(username)
	if err != nil {
		return nil, err
	}
	subgidRanges, err := parseSubgid(groupname)
	if err != nil {
		return nil, err
	}
	if len(subuidRanges) == 0 {
		return nil, fmt.Errorf("No subuid ranges found for user %q", username)
	}
	if len(subgidRanges) == 0 {
		return nil, fmt.Errorf("No subgid ranges found for group %q", groupname)
	}

	return &IDMappings{
		uids: createIDMap(subuidRanges),
		gids: createIDMap(subgidRanges),
	}, nil
}

// NewIDMappingsFromMaps creates a new mapping from two slices
// Deprecated: this is a temporary shim while transitioning to IDMapping
func NewIDMappingsFromMaps(uids []IDMap, gids []IDMap) *IDMappings {
	return &IDMappings{uids: uids, gids: gids}
}

// RootPair returns a uid and gid pair for the root user. The error is ignored
// because a root user always exists, and the defaults are correct when the uid
// and gid maps are empty.
func (i *IDMappings) RootPair() IDPair {
	uid, gid, _ := GetRootUIDGID(i.uids, i.gids)
	return IDPair{UID: uid, GID: gid}
}

// ToHost returns the host UID and GID for the container uid, gid.
// Remapping is only performed if the ids aren't already the remapped root ids
func (i *IDMappings) ToHost(pair IDPair) (IDPair, error) {
	var err error
	target := i.RootPair()

	if pair.UID != target.UID {
		target.UID, err = toHost(pair.UID, i.uids)
		if err != nil {
			return target, err
		}
	}

	if pair.GID != target.GID {
		target.GID, err = toHost(pair.GID, i.gids)
	}
	return target, err
}

// ToContainer returns the container UID and GID for the host uid and gid
func (i *IDMappings) ToContainer(pair IDPair) (int, int, error) {
	uid, err := toContainer(pair.UID, i.uids)
	if err != nil {
		return -1, -1, err
	}
	gid, err := toContainer(pair.GID, i.gids)
	return uid, gid, err
}

// Empty returns true if there are no id mappings
func (i *IDMappings) Empty() bool {
	return len(i.uids) == 0 && len(i.gids) == 0
}

// UIDs return the UID mapping
// TODO: remove this once everything has been refactored to use pairs
func (i *IDMappings) UIDs() []IDMap {
	return i.uids
}

// GIDs return the UID mapping
// TODO: remove this once everything has been refactored to use pairs
func (i *IDMappings) GIDs() []IDMap {
	return i.gids
}

func createIDMap(subidRanges ranges) []IDMap {
	idMap := []IDMap{}

	// sort the ranges by lowest ID first
	sort.Sort(subidRanges)
	containerID := 0
	for _, idrange := range subidRanges {
		idMap = append(idMap, IDMap{
			ContainerID: containerID,
			HostID:      idrange.Start,
			Size:        idrange.Length,
		})
		containerID = containerID + idrange.Length
	}
	return idMap
}

func parseSubuid(username string) (ranges, error) {
	return parseSubidFile(subuidFileName, username)
}

func parseSubgid(username string) (ranges, error) {
	return parseSubidFile(subgidFileName, username)
}

// parseSubidFile will read the appropriate file (/etc/subuid or /etc/subgid)
// and return all found ranges for a specified username. If the special value
// "ALL" is supplied for username, then all ranges in the file will be returned
func parseSubidFile(path, username string) (ranges, error) {
	var rangeList ranges

	subidFile, err := os.Open(path)
	if err != nil {
		return rangeList, err
	}
	defer subidFile.Close()

	s := bufio.NewScanner(subidFile)
	for s.Scan() {
		if err := s.Err(); err != nil {
			return rangeList, err
		}

		text := strings.TrimSpace(s.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		parts := strings.Split(text, ":")
		if len(parts) != 3 {
			return rangeList, fmt.Errorf("Cannot parse subuid/gid information: Format not correct for %s file", path)
		}
		if parts[0] == username || username == "ALL" {
			startid, err := strconv.Atoi(parts[1])
			if err != nil {
				return rangeList, fmt.Errorf("String to int conversion failed during subuid/gid parsing of %s: %v", path, err)
			}
			length, err := strconv.Atoi(parts[2])
			if err != nil {
				return rangeList, fmt.Errorf("String to int conversion failed during subuid/gid parsing of %s: %v", path, err)
			}
			rangeList = append(rangeList, subIDRange{startid, length})
		}
	}
	return rangeList, nil
}
