package capability

import (
	"testing"

	check "gopkg.in/check.v1"
)

func Test(t *testing.T) { check.TestingT(t) }

type SuiteCapabilities struct{}

var _ = check.Suite(&SuiteCapabilities{})

func (s *SuiteCapabilities) TestIsEmpty(c *check.C) {
	cap := NewList()
	c.Assert(cap.IsEmpty(), check.Equals, true)
}

func (s *SuiteCapabilities) TestDecode(c *check.C) {
	cap := NewList()
	err := cap.Decode([]byte("symref=foo symref=qux thin-pack"))
	c.Assert(err, check.IsNil)

	c.Assert(cap.m, check.HasLen, 2)
	c.Assert(cap.Get(SymRef), check.DeepEquals, []string{"foo", "qux"})
	c.Assert(cap.Get(ThinPack), check.IsNil)
}

func (s *SuiteCapabilities) TestDecodeWithLeadingSpace(c *check.C) {
	cap := NewList()
	err := cap.Decode([]byte(" report-status"))
	c.Assert(err, check.IsNil)

	c.Assert(cap.m, check.HasLen, 1)
	c.Assert(cap.Supports(ReportStatus), check.Equals, true)
}

func (s *SuiteCapabilities) TestDecodeEmpty(c *check.C) {
	cap := NewList()
	err := cap.Decode(nil)
	c.Assert(err, check.IsNil)
	c.Assert(cap, check.DeepEquals, NewList())
}

func (s *SuiteCapabilities) TestDecodeWithErrArguments(c *check.C) {
	cap := NewList()
	err := cap.Decode([]byte("thin-pack=foo"))
	c.Assert(err, check.Equals, ErrArguments)
}

func (s *SuiteCapabilities) TestDecodeWithEqual(c *check.C) {
	cap := NewList()
	err := cap.Decode([]byte("agent=foo=bar"))
	c.Assert(err, check.IsNil)

	c.Assert(cap.m, check.HasLen, 1)
	c.Assert(cap.Get(Agent), check.DeepEquals, []string{"foo=bar"})
}

func (s *SuiteCapabilities) TestDecodeWithUnknownCapability(c *check.C) {
	cap := NewList()
	err := cap.Decode([]byte("foo"))
	c.Assert(err, check.IsNil)
	c.Assert(cap.Supports(Capability("foo")), check.Equals, true)
}

func (s *SuiteCapabilities) TestString(c *check.C) {
	cap := NewList()
	cap.Set(Agent, "bar")
	cap.Set(SymRef, "foo:qux")
	cap.Set(ThinPack)

	c.Assert(cap.String(), check.Equals, "agent=bar symref=foo:qux thin-pack")
}

func (s *SuiteCapabilities) TestStringSort(c *check.C) {
	cap := NewList()
	cap.Set(Agent, "bar")
	cap.Set(SymRef, "foo:qux")
	cap.Set(ThinPack)

	c.Assert(cap.String(), check.Equals, "agent=bar symref=foo:qux thin-pack")
}

func (s *SuiteCapabilities) TestSet(c *check.C) {
	cap := NewList()
	err := cap.Add(SymRef, "foo", "qux")
	c.Assert(err, check.IsNil)
	err = cap.Set(SymRef, "bar")
	c.Assert(err, check.IsNil)

	c.Assert(cap.m, check.HasLen, 1)
	c.Assert(cap.Get(SymRef), check.DeepEquals, []string{"bar"})
}

func (s *SuiteCapabilities) TestSetEmpty(c *check.C) {
	cap := NewList()
	err := cap.Set(Agent, "bar")
	c.Assert(err, check.IsNil)

	c.Assert(cap.Get(Agent), check.HasLen, 1)
}

func (s *SuiteCapabilities) TestGetEmpty(c *check.C) {
	cap := NewList()
	c.Assert(cap.Get(Agent), check.HasLen, 0)
}

func (s *SuiteCapabilities) TestDelete(c *check.C) {
	cap := NewList()
	cap.Delete(SymRef)

	err := cap.Add(Sideband)
	c.Assert(err, check.IsNil)
	err = cap.Set(SymRef, "bar")
	c.Assert(err, check.IsNil)
	err = cap.Set(Sideband64k)
	c.Assert(err, check.IsNil)

	cap.Delete(SymRef)

	c.Assert(cap.String(), check.Equals, "side-band side-band-64k")
}

func (s *SuiteCapabilities) TestAdd(c *check.C) {
	cap := NewList()
	err := cap.Add(SymRef, "foo", "qux")
	c.Assert(err, check.IsNil)

	err = cap.Add(ThinPack)
	c.Assert(err, check.IsNil)

	c.Assert(cap.String(), check.Equals, "symref=foo symref=qux thin-pack")
}

func (s *SuiteCapabilities) TestAddUnknownCapability(c *check.C) {
	cap := NewList()
	err := cap.Add(Capability("foo"))
	c.Assert(err, check.IsNil)
	c.Assert(cap.Supports(Capability("foo")), check.Equals, true)
}

func (s *SuiteCapabilities) TestAddErrArgumentsRequired(c *check.C) {
	cap := NewList()
	err := cap.Add(SymRef)
	c.Assert(err, check.Equals, ErrArgumentsRequired)
}

func (s *SuiteCapabilities) TestAddErrArgumentsNotAllowed(c *check.C) {
	cap := NewList()
	err := cap.Add(OFSDelta, "foo")
	c.Assert(err, check.Equals, ErrArguments)
}

func (s *SuiteCapabilities) TestAddErrArgumendts(c *check.C) {
	cap := NewList()
	err := cap.Add(SymRef, "")
	c.Assert(err, check.Equals, ErrEmtpyArgument)
}

func (s *SuiteCapabilities) TestAddErrMultipleArguments(c *check.C) {
	cap := NewList()
	err := cap.Add(Agent, "foo")
	c.Assert(err, check.IsNil)

	err = cap.Add(Agent, "bar")
	c.Assert(err, check.Equals, ErrMultipleArguments)
}

func (s *SuiteCapabilities) TestAddErrMultipleArgumentsAtTheSameTime(c *check.C) {
	cap := NewList()
	err := cap.Add(Agent, "foo", "bar")
	c.Assert(err, check.Equals, ErrMultipleArguments)
}

func (s *SuiteCapabilities) TestAll(c *check.C) {
	cap := NewList()
	c.Assert(NewList().All(), check.IsNil)

	cap.Add(Agent, "foo")
	c.Assert(cap.All(), check.DeepEquals, []Capability{Agent})

	cap.Add(OFSDelta)
	c.Assert(cap.All(), check.DeepEquals, []Capability{Agent, OFSDelta})
}
