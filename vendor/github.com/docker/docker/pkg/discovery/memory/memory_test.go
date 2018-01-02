package memory

import (
	"testing"

	"github.com/docker/docker/pkg/discovery"
	"github.com/go-check/check"
)

// Hook up gocheck into the "go test" runner.
func Test(t *testing.T) { check.TestingT(t) }

type discoverySuite struct{}

var _ = check.Suite(&discoverySuite{})

func (s *discoverySuite) TestWatch(c *check.C) {
	d := &Discovery{}
	d.Initialize("foo", 1000, 0, nil)
	stopCh := make(chan struct{})
	ch, errCh := d.Watch(stopCh)

	// We have to drain the error channel otherwise Watch will get stuck.
	go func() {
		for range errCh {
		}
	}()

	expected := discovery.Entries{
		&discovery.Entry{Host: "1.1.1.1", Port: "1111"},
	}

	c.Assert(d.Register("1.1.1.1:1111"), check.IsNil)
	c.Assert(<-ch, check.DeepEquals, expected)

	expected = discovery.Entries{
		&discovery.Entry{Host: "1.1.1.1", Port: "1111"},
		&discovery.Entry{Host: "2.2.2.2", Port: "2222"},
	}

	c.Assert(d.Register("2.2.2.2:2222"), check.IsNil)
	c.Assert(<-ch, check.DeepEquals, expected)

	// Stop and make sure it closes all channels.
	close(stopCh)
	c.Assert(<-ch, check.IsNil)
	c.Assert(<-errCh, check.IsNil)
}
