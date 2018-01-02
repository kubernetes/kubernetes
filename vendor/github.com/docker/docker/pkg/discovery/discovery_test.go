package discovery

import (
	"testing"

	"github.com/go-check/check"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { check.TestingT(t) }

type DiscoverySuite struct{}

var _ = check.Suite(&DiscoverySuite{})

func (s *DiscoverySuite) TestNewEntry(c *check.C) {
	entry, err := NewEntry("127.0.0.1:2375")
	c.Assert(err, check.IsNil)
	c.Assert(entry.Equals(&Entry{Host: "127.0.0.1", Port: "2375"}), check.Equals, true)
	c.Assert(entry.String(), check.Equals, "127.0.0.1:2375")

	entry, err = NewEntry("[2001:db8:0:f101::2]:2375")
	c.Assert(err, check.IsNil)
	c.Assert(entry.Equals(&Entry{Host: "2001:db8:0:f101::2", Port: "2375"}), check.Equals, true)
	c.Assert(entry.String(), check.Equals, "[2001:db8:0:f101::2]:2375")

	_, err = NewEntry("127.0.0.1")
	c.Assert(err, check.NotNil)
}

func (s *DiscoverySuite) TestParse(c *check.C) {
	scheme, uri := parse("127.0.0.1:2375")
	c.Assert(scheme, check.Equals, "nodes")
	c.Assert(uri, check.Equals, "127.0.0.1:2375")

	scheme, uri = parse("localhost:2375")
	c.Assert(scheme, check.Equals, "nodes")
	c.Assert(uri, check.Equals, "localhost:2375")

	scheme, uri = parse("scheme://127.0.0.1:2375")
	c.Assert(scheme, check.Equals, "scheme")
	c.Assert(uri, check.Equals, "127.0.0.1:2375")

	scheme, uri = parse("scheme://localhost:2375")
	c.Assert(scheme, check.Equals, "scheme")
	c.Assert(uri, check.Equals, "localhost:2375")

	scheme, uri = parse("")
	c.Assert(scheme, check.Equals, "nodes")
	c.Assert(uri, check.Equals, "")
}

func (s *DiscoverySuite) TestCreateEntries(c *check.C) {
	entries, err := CreateEntries(nil)
	c.Assert(entries, check.DeepEquals, Entries{})
	c.Assert(err, check.IsNil)

	entries, err = CreateEntries([]string{"127.0.0.1:2375", "127.0.0.2:2375", "[2001:db8:0:f101::2]:2375", ""})
	c.Assert(err, check.IsNil)
	expected := Entries{
		&Entry{Host: "127.0.0.1", Port: "2375"},
		&Entry{Host: "127.0.0.2", Port: "2375"},
		&Entry{Host: "2001:db8:0:f101::2", Port: "2375"},
	}
	c.Assert(entries.Equals(expected), check.Equals, true)

	_, err = CreateEntries([]string{"127.0.0.1", "127.0.0.2"})
	c.Assert(err, check.NotNil)
}

func (s *DiscoverySuite) TestContainsEntry(c *check.C) {
	entries, err := CreateEntries([]string{"127.0.0.1:2375", "127.0.0.2:2375", ""})
	c.Assert(err, check.IsNil)
	c.Assert(entries.Contains(&Entry{Host: "127.0.0.1", Port: "2375"}), check.Equals, true)
	c.Assert(entries.Contains(&Entry{Host: "127.0.0.3", Port: "2375"}), check.Equals, false)
}

func (s *DiscoverySuite) TestEntriesEquality(c *check.C) {
	entries := Entries{
		&Entry{Host: "127.0.0.1", Port: "2375"},
		&Entry{Host: "127.0.0.2", Port: "2375"},
	}

	// Same
	c.Assert(entries.Equals(Entries{
		&Entry{Host: "127.0.0.1", Port: "2375"},
		&Entry{Host: "127.0.0.2", Port: "2375"},
	}), check.
		Equals, true)

	// Different size
	c.Assert(entries.Equals(Entries{
		&Entry{Host: "127.0.0.1", Port: "2375"},
		&Entry{Host: "127.0.0.2", Port: "2375"},
		&Entry{Host: "127.0.0.3", Port: "2375"},
	}), check.
		Equals, false)

	// Different content
	c.Assert(entries.Equals(Entries{
		&Entry{Host: "127.0.0.1", Port: "2375"},
		&Entry{Host: "127.0.0.42", Port: "2375"},
	}), check.
		Equals, false)

}

func (s *DiscoverySuite) TestEntriesDiff(c *check.C) {
	entry1 := &Entry{Host: "1.1.1.1", Port: "1111"}
	entry2 := &Entry{Host: "2.2.2.2", Port: "2222"}
	entry3 := &Entry{Host: "3.3.3.3", Port: "3333"}
	entries := Entries{entry1, entry2}

	// No diff
	added, removed := entries.Diff(Entries{entry2, entry1})
	c.Assert(added, check.HasLen, 0)
	c.Assert(removed, check.HasLen, 0)

	// Add
	added, removed = entries.Diff(Entries{entry2, entry3, entry1})
	c.Assert(added, check.HasLen, 1)
	c.Assert(added.Contains(entry3), check.Equals, true)
	c.Assert(removed, check.HasLen, 0)

	// Remove
	added, removed = entries.Diff(Entries{entry2})
	c.Assert(added, check.HasLen, 0)
	c.Assert(removed, check.HasLen, 1)
	c.Assert(removed.Contains(entry1), check.Equals, true)

	// Add and remove
	added, removed = entries.Diff(Entries{entry1, entry3})
	c.Assert(added, check.HasLen, 1)
	c.Assert(added.Contains(entry3), check.Equals, true)
	c.Assert(removed, check.HasLen, 1)
	c.Assert(removed.Contains(entry2), check.Equals, true)
}
