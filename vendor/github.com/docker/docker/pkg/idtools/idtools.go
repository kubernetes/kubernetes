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
func MkdirAllAs(path string, mode os.FileMode, ownerUID, ownerGID int) error {
	return mkdirAs(path, mode, ownerUID, ownerGID, true, true)
}

// MkdirAllNewAs creates a directory (include any along the path) and then modifies
// ownership ONLY of newly created directories to the requested uid/gid. If the
// directories along the path exist, no change of ownership will be performed
func MkdirAllNewAs(path string, mode os.FileMode, ownerUID, ownerGID int) error {
	return mkdirAs(path, mode, ownerUID, ownerGID, true, false)
}

// MkdirAs creates a directory and then modifies ownership to the requested uid/gid.
// If the directory already exists, this function still changes ownership
func MkdirAs(path string, mode os.FileMode, ownerUID, ownerGID int) error {
	return mkdirAs(path, mode, ownerUID, ownerGID, false, true)
}

// GetRootUIDGID retrieves the remapped root uid/gid pair from the set of maps.
// If the maps are empty, then the root uid/gid will default to "real" 0/0
func GetRootUIDGID(uidMap, gidMap []IDMap) (int, int, error) {
	var uid, gid int

	if uidMap != nil {
		xUID, err := ToHost(0, uidMap)
		if err != nil {
			return -1, -1, err
		}
		uid = xUID
	}
	if gidMap != nil {
		xGID, err := ToHost(0, gidMap)
		if err != nil {
			return -1, -1, err
		}
		gid = xGID
	}
	return uid, gid, nil
}

// ToContainer takes an id mapping, and uses it to translate a
// host ID to the remapped ID. If no map is provided, then the translation
// assumes a 1-to-1 mapping and returns the passed in id
func ToContainer(hostID int, idMap []IDMap) (int, error) {
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

// ToHost takes an id mapping and a remapped ID, and translates the
// ID to the mapped host ID. If no map is provided, then the translation
// assumes a 1-to-1 mapping and returns the passed in id #
func ToHost(contID int, idMap []IDMap) (int, error) {
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

// CreateIDMappings takes a requested user and group name and
// using the data from /etc/sub{uid,gid} ranges, creates the
// proper uid and gid remapping ranges for that user/group pair
func CreateIDMappings(username, groupname string) ([]IDMap, []IDMap, error) {
	subuidRanges, err := parseSubuid(username)
	if err != nil {
		return nil, nil, err
	}
	subgidRanges, err := parseSubgid(groupname)
	if err != nil {
		return nil, nil, err
	}
	if len(subuidRanges) == 0 {
		return nil, nil, fmt.Errorf("No subuid ranges found for user %q", username)
	}
	if len(subgidRanges) == 0 {
		return nil, nil, fmt.Errorf("No subgid ranges found for group %q", groupname)
	}

	return createIDMap(subuidRanges), createIDMap(subgidRanges), nil
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
