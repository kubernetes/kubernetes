package mountinfo

import "strings"

// FilterFunc is a type defining a callback function for GetMount(),
// used to filter out mountinfo entries we're not interested in,
// and/or stop further processing if we found what we wanted.
//
// It takes a pointer to the Info struct (not fully populated,
// currently only Mountpoint, Fstype, Source, and (on Linux)
// VfsOpts are filled in), and returns two booleans:
//
//  - skip: true if the entry should be skipped
//  - stop: true if parsing should be stopped after the entry
type FilterFunc func(*Info) (skip, stop bool)

// PrefixFilter discards all entries whose mount points
// do not start with a specific prefix
func PrefixFilter(prefix string) FilterFunc {
	return func(m *Info) (bool, bool) {
		skip := !strings.HasPrefix(m.Mountpoint, prefix)
		return skip, false
	}
}

// SingleEntryFilter looks for a specific entry
func SingleEntryFilter(mp string) FilterFunc {
	return func(m *Info) (bool, bool) {
		if m.Mountpoint == mp {
			return false, true // don't skip, stop now
		}
		return true, false // skip, keep going
	}
}

// ParentsFilter returns all entries whose mount points
// can be parents of a path specified, discarding others.
//
// For example, given `/var/lib/docker/something`, entries
// like `/var/lib/docker`, `/var` and `/` are returned.
func ParentsFilter(path string) FilterFunc {
	return func(m *Info) (bool, bool) {
		skip := !strings.HasPrefix(path, m.Mountpoint)
		return skip, false
	}
}

// FstypeFilter returns all entries that match provided fstype(s).
func FstypeFilter(fstype ...string) FilterFunc {
	return func(m *Info) (bool, bool) {
		for _, t := range fstype {
			if m.Fstype == t {
				return false, false // don't skeep, keep going
			}
		}
		return true, false // skip, keep going
	}
}
