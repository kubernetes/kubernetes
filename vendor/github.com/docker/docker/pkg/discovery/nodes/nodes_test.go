package nodes

import (
	"testing"

	"github.com/docker/docker/pkg/discovery"

	"github.com/go-check/check"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { check.TestingT(t) }

type DiscoverySuite struct{}

var _ = check.Suite(&DiscoverySuite{})

func (s *DiscoverySuite) TestInitialize(c *check.C) {
	d := &Discovery{}
	d.Initialize("1.1.1.1:1111,2.2.2.2:2222", 0, 0, nil)
	c.Assert(len(d.entries), check.Equals, 2)
	c.Assert(d.entries[0].String(), check.Equals, "1.1.1.1:1111")
	c.Assert(d.entries[1].String(), check.Equals, "2.2.2.2:2222")
}

func (s *DiscoverySuite) TestInitializeWithPattern(c *check.C) {
	d := &Discovery{}
	d.Initialize("1.1.1.[1:2]:1111,2.2.2.[2:4]:2222", 0, 0, nil)
	c.Assert(len(d.entries), check.Equals, 5)
	c.Assert(d.entries[0].String(), check.Equals, "1.1.1.1:1111")
	c.Assert(d.entries[1].String(), check.Equals, "1.1.1.2:1111")
	c.Assert(d.entries[2].String(), check.Equals, "2.2.2.2:2222")
	c.Assert(d.entries[3].String(), check.Equals, "2.2.2.3:2222")
	c.Assert(d.entries[4].String(), check.Equals, "2.2.2.4:2222")
}

func (s *DiscoverySuite) TestWatch(c *check.C) {
	d := &Discovery{}
	d.Initialize("1.1.1.1:1111,2.2.2.2:2222", 0, 0, nil)
	expected := discovery.Entries{
		&discovery.Entry{Host: "1.1.1.1", Port: "1111"},
		&discovery.Entry{Host: "2.2.2.2", Port: "2222"},
	}
	ch, _ := d.Watch(nil)
	c.Assert(expected.Equals(<-ch), check.Equals, true)
}

func (s *DiscoverySuite) TestRegister(c *check.C) {
	d := &Discovery{}
	c.Assert(d.Register("0.0.0.0"), check.NotNil)
}
