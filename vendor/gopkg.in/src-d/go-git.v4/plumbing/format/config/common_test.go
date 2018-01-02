package config

import (
	"testing"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type CommonSuite struct{}

var _ = Suite(&CommonSuite{})

func (s *CommonSuite) TestConfig_SetOption(c *C) {
	obtained := New().SetOption("section", NoSubsection, "key1", "value1")
	expected := &Config{
		Sections: []*Section{
			{
				Name: "section",
				Options: []*Option{
					{Key: "key1", Value: "value1"},
				},
			},
		},
	}
	c.Assert(obtained, DeepEquals, expected)
	obtained = obtained.SetOption("section", NoSubsection, "key1", "value1")
	c.Assert(obtained, DeepEquals, expected)

	obtained = New().SetOption("section", "subsection", "key1", "value1")
	expected = &Config{
		Sections: []*Section{
			{
				Name: "section",
				Subsections: []*Subsection{
					{
						Name: "subsection",
						Options: []*Option{
							{Key: "key1", Value: "value1"},
						},
					},
				},
			},
		},
	}
	c.Assert(obtained, DeepEquals, expected)
	obtained = obtained.SetOption("section", "subsection", "key1", "value1")
	c.Assert(obtained, DeepEquals, expected)
}

func (s *CommonSuite) TestConfig_AddOption(c *C) {
	obtained := New().AddOption("section", NoSubsection, "key1", "value1")
	expected := &Config{
		Sections: []*Section{
			{
				Name: "section",
				Options: []*Option{
					{Key: "key1", Value: "value1"},
				},
			},
		},
	}
	c.Assert(obtained, DeepEquals, expected)
}

func (s *CommonSuite) TestConfig_RemoveSection(c *C) {
	sect := New().
		AddOption("section1", NoSubsection, "key1", "value1").
		AddOption("section2", NoSubsection, "key1", "value1")
	expected := New().
		AddOption("section1", NoSubsection, "key1", "value1")
	c.Assert(sect.RemoveSection("other"), DeepEquals, sect)
	c.Assert(sect.RemoveSection("section2"), DeepEquals, expected)
}

func (s *CommonSuite) TestConfig_RemoveSubsection(c *C) {
	sect := New().
		AddOption("section1", "sub1", "key1", "value1").
		AddOption("section1", "sub2", "key1", "value1")
	expected := New().
		AddOption("section1", "sub1", "key1", "value1")
	c.Assert(sect.RemoveSubsection("section1", "other"), DeepEquals, sect)
	c.Assert(sect.RemoveSubsection("other", "other"), DeepEquals, sect)
	c.Assert(sect.RemoveSubsection("section1", "sub2"), DeepEquals, expected)
}
