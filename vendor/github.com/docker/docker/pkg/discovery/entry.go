package discovery

import "net"

// NewEntry creates a new entry.
func NewEntry(url string) (*Entry, error) {
	host, port, err := net.SplitHostPort(url)
	if err != nil {
		return nil, err
	}
	return &Entry{host, port}, nil
}

// An Entry represents a host.
type Entry struct {
	Host string
	Port string
}

// Equals returns true if cmp contains the same data.
func (e *Entry) Equals(cmp *Entry) bool {
	return e.Host == cmp.Host && e.Port == cmp.Port
}

// String returns the string form of an entry.
func (e *Entry) String() string {
	return net.JoinHostPort(e.Host, e.Port)
}

// Entries is a list of *Entry with some helpers.
type Entries []*Entry

// Equals returns true if cmp contains the same data.
func (e Entries) Equals(cmp Entries) bool {
	// Check if the file has really changed.
	if len(e) != len(cmp) {
		return false
	}
	for i := range e {
		if !e[i].Equals(cmp[i]) {
			return false
		}
	}
	return true
}

// Contains returns true if the Entries contain a given Entry.
func (e Entries) Contains(entry *Entry) bool {
	for _, curr := range e {
		if curr.Equals(entry) {
			return true
		}
	}
	return false
}

// Diff compares two entries and returns the added and removed entries.
func (e Entries) Diff(cmp Entries) (Entries, Entries) {
	added := Entries{}
	for _, entry := range cmp {
		if !e.Contains(entry) {
			added = append(added, entry)
		}
	}

	removed := Entries{}
	for _, entry := range e {
		if !cmp.Contains(entry) {
			removed = append(removed, entry)
		}
	}

	return added, removed
}

// CreateEntries returns an array of entries based on the given addresses.
func CreateEntries(addrs []string) (Entries, error) {
	entries := Entries{}
	if addrs == nil {
		return entries, nil
	}

	for _, addr := range addrs {
		if len(addr) == 0 {
			continue
		}
		entry, err := NewEntry(addr)
		if err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}
	return entries, nil
}
