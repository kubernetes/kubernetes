package index

import (
	"bytes"
	"strings"
	"time"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

func (s *IndexSuite) TestEncode(c *C) {
	idx := &Index{
		Version: 2,
		Entries: []*Entry{{
			CreatedAt:  time.Now(),
			ModifiedAt: time.Now(),
			Dev:        4242,
			Inode:      424242,
			UID:        84,
			GID:        8484,
			Size:       42,
			Stage:      TheirMode,
			Hash:       plumbing.NewHash("e25b29c8946e0e192fae2edc1dabf7be71e8ecf3"),
			Name:       "foo",
		}, {
			CreatedAt:  time.Now(),
			ModifiedAt: time.Now(),
			Name:       "bar",
			Size:       82,
		}, {
			CreatedAt:  time.Now(),
			ModifiedAt: time.Now(),
			Name:       strings.Repeat(" ", 20),
			Size:       82,
		}},
	}

	buf := bytes.NewBuffer(nil)
	e := NewEncoder(buf)
	err := e.Encode(idx)
	c.Assert(err, IsNil)

	output := &Index{}
	d := NewDecoder(buf)
	err = d.Decode(output)
	c.Assert(err, IsNil)

	c.Assert(idx, DeepEquals, output)

	c.Assert(output.Entries[0].Name, Equals, strings.Repeat(" ", 20))
	c.Assert(output.Entries[1].Name, Equals, "bar")
	c.Assert(output.Entries[2].Name, Equals, "foo")

}

func (s *IndexSuite) TestEncodeUnsuportedVersion(c *C) {
	idx := &Index{Version: 3}

	buf := bytes.NewBuffer(nil)
	e := NewEncoder(buf)
	err := e.Encode(idx)
	c.Assert(err, Equals, ErrUnsupportedVersion)
}

func (s *IndexSuite) TestEncodeWithIntentToAddUnsuportedVersion(c *C) {
	idx := &Index{
		Version: 2,
		Entries: []*Entry{{IntentToAdd: true}},
	}

	buf := bytes.NewBuffer(nil)
	e := NewEncoder(buf)
	err := e.Encode(idx)
	c.Assert(err, Equals, ErrUnsupportedVersion)
}

func (s *IndexSuite) TestEncodeWithSkipWorktreeUnsuportedVersion(c *C) {
	idx := &Index{
		Version: 2,
		Entries: []*Entry{{SkipWorktree: true}},
	}

	buf := bytes.NewBuffer(nil)
	e := NewEncoder(buf)
	err := e.Encode(idx)
	c.Assert(err, Equals, ErrUnsupportedVersion)
}
