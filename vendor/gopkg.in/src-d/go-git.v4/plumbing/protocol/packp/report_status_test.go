package packp

import (
	"bytes"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"

	. "gopkg.in/check.v1"
)

type ReportStatusSuite struct{}

var _ = Suite(&ReportStatusSuite{})

func (s *ReportStatusSuite) TestError(c *C) {
	rs := NewReportStatus()
	rs.UnpackStatus = "ok"
	c.Assert(rs.Error(), IsNil)
	rs.UnpackStatus = "OK"
	c.Assert(rs.Error(), ErrorMatches, "unpack error: OK")
	rs.UnpackStatus = ""
	c.Assert(rs.Error(), ErrorMatches, "unpack error: ")

	cs := &CommandStatus{ReferenceName: plumbing.ReferenceName("ref")}
	rs.UnpackStatus = "ok"
	rs.CommandStatuses = append(rs.CommandStatuses, cs)

	cs.Status = "ok"
	c.Assert(rs.Error(), IsNil)
	cs.Status = "OK"
	c.Assert(rs.Error(), ErrorMatches, "command error on ref: OK")
	cs.Status = ""
	c.Assert(rs.Error(), ErrorMatches, "command error on ref: ")
}

func (s *ReportStatusSuite) testEncodeDecodeOk(c *C, rs *ReportStatus, lines ...string) {
	s.testDecodeOk(c, rs, lines...)
	s.testEncodeOk(c, rs, lines...)
}

func (s *ReportStatusSuite) testDecodeOk(c *C, expected *ReportStatus, lines ...string) {
	r := toPktLines(c, lines)
	rs := NewReportStatus()
	c.Assert(rs.Decode(r), IsNil)
	c.Assert(rs, DeepEquals, expected)
}

func (s *ReportStatusSuite) testDecodeError(c *C, errorMatch string, lines ...string) {
	r := toPktLines(c, lines)
	rs := NewReportStatus()
	c.Assert(rs.Decode(r), ErrorMatches, errorMatch)
}

func (s *ReportStatusSuite) testEncodeOk(c *C, input *ReportStatus, lines ...string) {
	expected := pktlines(c, lines...)
	var buf bytes.Buffer
	c.Assert(input.Encode(&buf), IsNil)
	obtained := buf.Bytes()

	comment := Commentf("\nobtained = %s\nexpected = %s\n", string(obtained), string(expected))

	c.Assert(obtained, DeepEquals, expected, comment)
}

func (s *ReportStatusSuite) TestEncodeDecodeOkOneReference(c *C) {
	rs := NewReportStatus()
	rs.UnpackStatus = "ok"
	rs.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testEncodeDecodeOk(c, rs,
		"unpack ok\n",
		"ok refs/heads/master\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestEncodeDecodeOkOneReferenceFailed(c *C) {
	rs := NewReportStatus()
	rs.UnpackStatus = "my error"
	rs.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "command error",
	}}

	s.testEncodeDecodeOk(c, rs,
		"unpack my error\n",
		"ng refs/heads/master command error\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestEncodeDecodeOkMoreReferences(c *C) {
	rs := NewReportStatus()
	rs.UnpackStatus = "ok"
	rs.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}, {
		ReferenceName: plumbing.ReferenceName("refs/heads/a"),
		Status:        "ok",
	}, {
		ReferenceName: plumbing.ReferenceName("refs/heads/b"),
		Status:        "ok",
	}}

	s.testEncodeDecodeOk(c, rs,
		"unpack ok\n",
		"ok refs/heads/master\n",
		"ok refs/heads/a\n",
		"ok refs/heads/b\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestEncodeDecodeOkMoreReferencesFailed(c *C) {
	rs := NewReportStatus()
	rs.UnpackStatus = "my error"
	rs.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}, {
		ReferenceName: plumbing.ReferenceName("refs/heads/a"),
		Status:        "command error",
	}, {
		ReferenceName: plumbing.ReferenceName("refs/heads/b"),
		Status:        "ok",
	}}

	s.testEncodeDecodeOk(c, rs,
		"unpack my error\n",
		"ok refs/heads/master\n",
		"ng refs/heads/a command error\n",
		"ok refs/heads/b\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestEncodeDecodeOkNoReferences(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"

	s.testEncodeDecodeOk(c, expected,
		"unpack ok\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestEncodeDecodeOkNoReferencesFailed(c *C) {
	rs := NewReportStatus()
	rs.UnpackStatus = "my error"

	s.testEncodeDecodeOk(c, rs,
		"unpack my error\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestDecodeErrorOneReferenceNoFlush(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "missing flush",
		"unpack ok\n",
		"ok refs/heads/master\n",
	)
}

func (s *ReportStatusSuite) TestDecodeErrorEmpty(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "unexpected EOF")
}

func (s *ReportStatusSuite) TestDecodeErrorMalformed(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "malformed unpack status: unpackok",
		"unpackok\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestDecodeErrorMalformed2(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "malformed unpack status: UNPACK OK",
		"UNPACK OK\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestDecodeErrorMalformedCommandStatus(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "malformed command status: ko refs/heads/master",
		"unpack ok\n",
		"ko refs/heads/master\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestDecodeErrorMalformedCommandStatus2(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "malformed command status: ng refs/heads/master",
		"unpack ok\n",
		"ng refs/heads/master\n",
		pktline.FlushString,
	)
}

func (s *ReportStatusSuite) TestDecodeErrorPrematureFlush(c *C) {
	expected := NewReportStatus()
	expected.UnpackStatus = "ok"
	expected.CommandStatuses = []*CommandStatus{{
		ReferenceName: plumbing.ReferenceName("refs/heads/master"),
		Status:        "ok",
	}}

	s.testDecodeError(c, "premature flush",
		pktline.FlushString,
	)
}
