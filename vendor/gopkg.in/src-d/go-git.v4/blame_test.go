package git

import (
	"github.com/src-d/go-git-fixtures"
	"gopkg.in/src-d/go-git.v4/plumbing"

	. "gopkg.in/check.v1"
)

type BlameSuite struct {
	BaseSuite
}

var _ = Suite(&BlameSuite{})

type blameTest struct {
	repo   string
	rev    string
	path   string
	blames []string // the commits blamed for each line
}

// run a blame on all the suite's tests
func (s *BlameSuite) TestBlame(c *C) {
	for _, t := range blameTests {
		r := s.NewRepositoryFromPackfile(fixtures.ByURL(t.repo).One())

		exp := s.mockBlame(c, t, r)
		commit, err := r.CommitObject(plumbing.NewHash(t.rev))
		c.Assert(err, IsNil)

		obt, err := Blame(commit, t.path)
		c.Assert(err, IsNil)
		c.Assert(obt, DeepEquals, exp)
	}
}

func (s *BlameSuite) mockBlame(c *C, t blameTest, r *Repository) (blame *BlameResult) {
	commit, err := r.CommitObject(plumbing.NewHash(t.rev))
	c.Assert(err, IsNil, Commentf("%v: repo=%s, rev=%s", err, t.repo, t.rev))

	f, err := commit.File(t.path)
	c.Assert(err, IsNil)
	lines, err := f.Lines()
	c.Assert(err, IsNil)
	c.Assert(len(t.blames), Equals, len(lines), Commentf(
		"repo=%s, path=%s, rev=%s: the number of lines in the file and the number of expected blames differ (len(blames)=%d, len(lines)=%d)\nblames=%#q\nlines=%#q", t.repo, t.path, t.rev, len(t.blames), len(lines), t.blames, lines))

	blamedLines := make([]*Line, 0, len(t.blames))
	for i := range t.blames {
		commit, err := r.CommitObject(plumbing.NewHash(t.blames[i]))
		c.Assert(err, IsNil)
		l := &Line{
			Author: commit.Author.Email,
			Text:   lines[i],
		}
		blamedLines = append(blamedLines, l)
	}

	return &BlameResult{
		Path:  t.path,
		Rev:   plumbing.NewHash(t.rev),
		Lines: blamedLines,
	}
}

// utility function to avoid writing so many repeated commits
func repeat(s string, n int) []string {
	if n < 0 {
		panic("repeat: n < 0")
	}
	r := make([]string, 0, n)
	for i := 0; i < n; i++ {
		r = append(r, s)
	}

	return r
}

// utility function to concat slices
func concat(vargs ...[]string) []string {
	var r []string
	for _, ss := range vargs {
		r = append(r, ss...)
	}

	return r
}

var blameTests = [...]blameTest{
	// use the blame2humantest.bash script to easily add more tests.
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "binary.jpg", concat(
		repeat("35e85108805c84807bc66a02d91535e1e24b38b9", 285),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "CHANGELOG", concat(
		repeat("b8e471f58bcbca63b07bda20e428190409c2db47", 1),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "go/example.go", concat(
		repeat("918c48b83bd081e863dbe1b80f8998f058cd8294", 142),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "json/long.json", concat(
		repeat("af2d6a6954d532f8ffb47615169c8fdf9d383a1a", 6492),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "json/short.json", concat(
		repeat("af2d6a6954d532f8ffb47615169c8fdf9d383a1a", 22),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "LICENSE", concat(
		repeat("b029517f6300c2da0f4b651b8642506cd6aaf45d", 22),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "php/crappy.php", concat(
		repeat("918c48b83bd081e863dbe1b80f8998f058cd8294", 259),
	)},
	{"https://github.com/git-fixtures/basic.git", "6ecf0ef2c2dffb796033e5a02219af86ec6584e5", "vendor/foo.go", concat(
		repeat("6ecf0ef2c2dffb796033e5a02219af86ec6584e5", 7),
	)},
	/*
		// Failed
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "InstallSpinnaker.sh", concat(
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 2),
			repeat("a47d0aaeda421f06df248ad65bd58230766bf118", 1),
			repeat("23673af3ad70b50bba7fdafadc2323302f5ba520", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 29),
			repeat("9a06d3f20eabb254d0a1e2ff7735ef007ccd595e", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 4),
			repeat("a47d0aaeda421f06df248ad65bd58230766bf118", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 5),
			repeat("0c5bb1e4392e751f884f3c57de5d4aee72c40031", 2),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 3),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 7),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 2),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 5),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 7),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 3),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 6),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 10),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 4),
			repeat("0c5bb1e4392e751f884f3c57de5d4aee72c40031", 2),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 2),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 4),
			repeat("23673af3ad70b50bba7fdafadc2323302f5ba520", 4),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 4),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("0c5bb1e4392e751f884f3c57de5d4aee72c40031", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 13),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 2),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 6),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 2),
			repeat("0c5bb1e4392e751f884f3c57de5d4aee72c40031", 1),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 4),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 3),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 2),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 4),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 3),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 15),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 1),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 1),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 8),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 2),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 12),
			repeat("505577dc87d300cf562dc4702a05a5615d90d855", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 5),
			repeat("370d61cdbc1f3c90db6759f1599ccbabd40ad6c1", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 4),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 1),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 5),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 3),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 2),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 2),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 9),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 1),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 3),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 4),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("8eb116de9128c314ac8a6f5310ca500b8c74f5db", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 6),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 6),
			repeat("d2f6214b625db706384b378a29cc4c22237db97a", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 3),
			repeat("d2f6214b625db706384b378a29cc4c22237db97a", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 4),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("c9c2a0ec03968ab17e8b16fdec9661eb1dbea173", 1),
			repeat("d2f6214b625db706384b378a29cc4c22237db97a", 2),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 12),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 5),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 3),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 5),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 3),
			repeat("a47d0aaeda421f06df248ad65bd58230766bf118", 5),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 5),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 2),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 1),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("b2c7142082d52b09ca20228606c31c7479c0833e", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("495c7118e7cf757aa04eab410b64bfb5b5149ad2", 1),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 1),
			repeat("495c7118e7cf757aa04eab410b64bfb5b5149ad2", 3),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 1),
			repeat("495c7118e7cf757aa04eab410b64bfb5b5149ad2", 1),
			repeat("50d0556563599366f29cb286525780004fa5a317", 1),
			repeat("dd2d03c19658ff96d371aef00e75e2e54702da0e", 1),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 1),
			repeat("dd2d03c19658ff96d371aef00e75e2e54702da0e", 2),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 2),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 1),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("b5c6053a46993b20d1b91e7b7206bffa54669ad7", 1),
			repeat("9e74d009894d73dd07773ea6b3bdd8323db980f7", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("d4b48a39aba7d3bd3e8abef2274a95b112d1ae73", 4),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 1),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 1),
			repeat("d2f6214b625db706384b378a29cc4c22237db97a", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 3),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 2),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 2),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 4),
			repeat("d2f6214b625db706384b378a29cc4c22237db97a", 1),
			repeat("b7015a5d36990d69a054482556127b9c7404a24a", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 5),
			repeat("b41d7c0e5b20bbe7c8eb6606731a3ff68f4e3941", 2),
			repeat("d2f6214b625db706384b378a29cc4c22237db97a", 1),
			repeat("ce9f123d790717599aaeb76bc62510de437761be", 5),
			repeat("ba486de7c025457963701114c683dcd4708e1dee", 4),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 1),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 3),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 1),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 3),
			repeat("6328ee836affafc1b52127147b5ca07300ac78e6", 2),
			repeat("01e65d67eed8afcb67a6bdf1c962541f62b299c9", 3),
			repeat("3de4f77c105f700f50d9549d32b9a05a01b46c4b", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 2),
			repeat("370d61cdbc1f3c90db6759f1599ccbabd40ad6c1", 6),
			repeat("dd7e66c862209e8b912694a582a09c0db3227f0d", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 2),
			repeat("dd7e66c862209e8b912694a582a09c0db3227f0d", 3),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("dd7e66c862209e8b912694a582a09c0db3227f0d", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 3),
		)},
	*/
	{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "pylib/spinnaker/reconfigure_spinnaker.py", concat(
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 22),
		repeat("c89dab0d42f1856d157357e9010f8cc6a12f5b1f", 7),
	)},
	{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "pylib/spinnaker/validate_configuration.py", concat(
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 29),
		repeat("1e3d328a2cabda5d0aaddc5dec65271343e0dc37", 19),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 15),
		repeat("b5d999e2986e190d81767cd3cfeda0260f9f6fb8", 1),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 12),
		repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 1),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
		repeat("b5d999e2986e190d81767cd3cfeda0260f9f6fb8", 8),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
		repeat("b5d999e2986e190d81767cd3cfeda0260f9f6fb8", 4),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 46),
		repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 1),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
		repeat("1e3d328a2cabda5d0aaddc5dec65271343e0dc37", 42),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
		repeat("1e3d328a2cabda5d0aaddc5dec65271343e0dc37", 1),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 3),
		repeat("1e3d328a2cabda5d0aaddc5dec65271343e0dc37", 1),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
		repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 8),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
		repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 2),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 3),
		repeat("1e3d328a2cabda5d0aaddc5dec65271343e0dc37", 3),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 12),
		repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 10),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 69),
		repeat("b5d999e2986e190d81767cd3cfeda0260f9f6fb8", 7),
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
	)},
	{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "pylib/spinnaker/run.py", concat(
		repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 185),
	)},
	/*
		// Fail by 3
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "pylib/spinnaker/configurator.py", concat(
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 53),
			repeat("c89dab0d42f1856d157357e9010f8cc6a12f5b1f", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
			repeat("e805183c72f0426fb073728c01901c2fd2db1da6", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 6),
			repeat("023d4fb17b76e0fe0764971df8b8538b735a1d67", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 36),
			repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 3),
			repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 3),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
			repeat("c89dab0d42f1856d157357e9010f8cc6a12f5b1f", 13),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("c89dab0d42f1856d157357e9010f8cc6a12f5b1f", 18),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("1e14f94bcf82694fdc7e2dcbbfdbbed58db0f4d9", 1),
			repeat("023d4fb17b76e0fe0764971df8b8538b735a1d67", 17),
			repeat("c89dab0d42f1856d157357e9010f8cc6a12f5b1f", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 43),
		)},
	*/
	{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "pylib/spinnaker/__init__.py", []string{}},
	{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "gradle/wrapper/gradle-wrapper.jar", concat(
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 1),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 7),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 2),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 2),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 3),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 1),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 10),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 11),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 29),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 7),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 58),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 1),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 1),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 2),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 2),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 13),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 4),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 3),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 13),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 2),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 9),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 3),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 1),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 17),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 3),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 6),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 6),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 3),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 5),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 4),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 3),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 2),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 1),
		repeat("11d6c1020b1765e236ca65b2709d37b5bfdba0f4", 6),
		repeat("bc02440df2ff95a014a7b3cb11b98c3a2bded777", 55),
	)},
	{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "config/settings.js", concat(
		repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 17),
		repeat("99534ecc895fe17a1d562bb3049d4168a04d0865", 1),
		repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 43),
		repeat("d2838db9f6ef9628645e7d04cd9658a83e8708ea", 1),
		repeat("637ba49300f701cfbd859c1ccf13c4f39a9ba1c8", 1),
		repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 13),
	)},
	/*
		// fail a few lines
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "config/default-spinnaker-local.yml", concat(
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 9),
			repeat("5e09821cbd7d710405b61cab0a795c2982a71b9c", 2),
			repeat("99534ecc895fe17a1d562bb3049d4168a04d0865", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 2),
			repeat("a596972a661d9a7deca8abd18b52ce1a39516e89", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 5),
			repeat("5e09821cbd7d710405b61cab0a795c2982a71b9c", 2),
			repeat("a596972a661d9a7deca8abd18b52ce1a39516e89", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 5),
			repeat("5e09821cbd7d710405b61cab0a795c2982a71b9c", 1),
			repeat("8980daf661408a3faa1f22c225702a5c1d11d5c9", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 25),
			repeat("caf6d62e8285d4681514dd8027356fb019bc97ff", 1),
			repeat("eaf7614cad81e8ab5c813dd4821129d0c04ea449", 1),
			repeat("caf6d62e8285d4681514dd8027356fb019bc97ff", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 24),
			repeat("974b775a8978b120ff710cac93a21c7387b914c9", 2),
			repeat("3ce7b902a51bac2f10994f7d1f251b616c975e54", 1),
			repeat("5a2a845bc08974a36d599a4a4b7e25be833823b0", 6),
			repeat("41e96c54a478e5d09dd07ed7feb2d8d08d8c7e3c", 14),
			repeat("7c8d9a6081d9cb7a56c479bfe64d70540ea32795", 5),
			repeat("5a2a845bc08974a36d599a4a4b7e25be833823b0", 2),
		)},
	*/
	/*
		// fail one line
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "config/spinnaker.yml", concat(
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 32),
			repeat("41e96c54a478e5d09dd07ed7feb2d8d08d8c7e3c", 2),
			repeat("5a2a845bc08974a36d599a4a4b7e25be833823b0", 1),
			repeat("41e96c54a478e5d09dd07ed7feb2d8d08d8c7e3c", 6),
			repeat("5a2a845bc08974a36d599a4a4b7e25be833823b0", 2),
			repeat("41e96c54a478e5d09dd07ed7feb2d8d08d8c7e3c", 2),
			repeat("5a2a845bc08974a36d599a4a4b7e25be833823b0", 2),
			repeat("41e96c54a478e5d09dd07ed7feb2d8d08d8c7e3c", 3),
			repeat("7c8d9a6081d9cb7a56c479bfe64d70540ea32795", 3),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 50),
			repeat("974b775a8978b120ff710cac93a21c7387b914c9", 2),
			repeat("d4553dac205023fa77652308af1a2d1cf52138fb", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 9),
			repeat("caf6d62e8285d4681514dd8027356fb019bc97ff", 1),
			repeat("eaf7614cad81e8ab5c813dd4821129d0c04ea449", 1),
			repeat("caf6d62e8285d4681514dd8027356fb019bc97ff", 1),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 39),
			repeat("079e42e7c979541b6fab7343838f7b9fd4a360cd", 6),
			repeat("ae904e8d60228c21c47368f6a10f1cc9ca3aeebf", 15),
		)},
	*/
	/*
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "dev/install_development.sh", concat(
			repeat("99534ecc895fe17a1d562bb3049d4168a04d0865", 1),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 71),
		)},
	*/
	/*
		// FAIL two lines interchanged
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "dev/bootstrap_dev.sh", concat(
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 95),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 10),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 7),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 3),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 12),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 2),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 3),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 6),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 4),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
			repeat("376599177551c3f04ccc94d71bbb4d037dec0c3f", 2),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 17),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 2),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 2),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 3),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 3),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 5),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 5),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 8),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 4),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 1),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 6),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 4),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 10),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 2),
			repeat("fc28a378558cdb5bbc08b6dcb96ee77c5b716760", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 1),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 8),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 1),
			repeat("fc28a378558cdb5bbc08b6dcb96ee77c5b716760", 1),
			repeat("d1ff4e13e9e0b500821aa558373878f93487e34b", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 4),
			repeat("24551a5d486969a2972ee05e87f16444890f9555", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 2),
			repeat("24551a5d486969a2972ee05e87f16444890f9555", 1),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 8),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 13),
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 5),
			repeat("24551a5d486969a2972ee05e87f16444890f9555", 1),
			repeat("838aed816872c52ed435e4876a7b64dba0bed500", 8),
		)},
	*/
	/*
		// FAIL move?
		{"https://github.com/spinnaker/spinnaker.git", "f39d86f59a0781f130e8de6b2115329c1fbe9545", "dev/create_google_dev_vm.sh", concat(
			repeat("a24001f6938d425d0e7504bdf5d27fc866a85c3d", 20),
		)},
	*/
}
