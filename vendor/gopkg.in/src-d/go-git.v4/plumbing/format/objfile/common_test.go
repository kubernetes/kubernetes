package objfile

import (
	"encoding/base64"
	"testing"

	. "gopkg.in/check.v1"
	"gopkg.in/src-d/go-git.v4/plumbing"
)

type objfileFixture struct {
	hash    string              // hash of data
	t       plumbing.ObjectType // object type
	content string              // base64-encoded content
	data    string              // base64-encoded objfile data
}

var objfileFixtures = []objfileFixture{
	{
		"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
		plumbing.BlobObject,
		base64.StdEncoding.EncodeToString([]byte("")),
		"eAFLyslPUjBgAAAJsAHw",
	},
	{
		"a8a940627d132695a9769df883f85992f0ff4a43",
		plumbing.BlobObject,
		base64.StdEncoding.EncodeToString([]byte("this is a test")),
		"eAFLyslPUjA0YSjJyCxWAKJEhZLU4hIAUDYHOg==",
	},
	{
		"4dc2174801ac4a3d36886210fd086fbe134cf7b2",
		plumbing.BlobObject,
		base64.StdEncoding.EncodeToString([]byte("this\nis\n\n\na\nmultiline\n\ntest.\n")),
		"eAFLyslPUjCyZCjJyCzmAiIurkSu3NKcksyczLxULq6S1OISPS4A1I8LMQ==",
	},
	{
		"13e6f47dd57798bfdc728d91f5c6d7f40c5bb5fc",
		plumbing.BlobObject,
		base64.StdEncoding.EncodeToString([]byte("this tests\r\nCRLF\r\nencoded files.\r\n")),
		"eAFLyslPUjA2YSjJyCxWKEktLinm5XIO8nHj5UrNS85PSU1RSMvMSS3W4+UCABp3DNE=",
	},
	{
		"72a7bc4667ab068e954172437b993d9fbaa137cb",
		plumbing.BlobObject,
		base64.StdEncoding.EncodeToString([]byte("test@example.com")),
		"eAFLyslPUjA0YyhJLS5xSK1IzC3ISdVLzs8FAGVtCIA=",
	},
	{
		"bb2b40e85ec0455d1de72daff71583f0dd72a33f",
		plumbing.BlobObject,
		base64.StdEncoding.EncodeToString([]byte("package main\r\n\r\nimport (\r\n\t\"fmt\"\r\n\t\"io\"\r\n\t\"os\"\r\n\r\n\t\"gopkg.in/src-d/go-git.v3\"\r\n)\r\n\r\nfunc main() {\r\n\tfmt.Printf(\"Retrieving %q ...\\n\", os.Args[2])\r\n\tr, err := git.NewRepository(os.Args[2], nil)\r\n\tif err != nil {\r\n\t\tpanic(err)\r\n\t}\r\n\r\n\tif err := r.Pull(\"origin\", \"refs/heads/master\"); err != nil {\r\n\t\tpanic(err)\r\n\t}\r\n\r\n\tdumpCommits(r)\r\n}\r\n\r\nfunc dumpCommits(r *git.Repository) {\r\n\titer := r.Commits()\r\n\tdefer iter.Close()\r\n\r\n\tfor {\r\n\t\tcommit, err := iter.Next()\r\n\t\tif err != nil {\r\n\t\t\tif err == io.EOF {\r\n\t\t\t\tbreak\r\n\t\t\t}\r\n\r\n\t\t\tpanic(err)\r\n\t\t}\r\n\r\n\t\tfmt.Println(commit)\r\n\t}\r\n}\r\n")),
		"eAGNUU1LAzEU9JpC/0NcEFJps2ARQdmDFD3W0qt6SHez8dHdZH1JqyL+d/Oy/aDgQVh47LzJTGayatyKX99MzzpVrpXRvFVgh4PhANrOYeBiOGBZ3YaMJrg0nI+D/o3r1kaCzT2Wkyo3bmIgyO00rkfEqDe2TIJixL/jgagjFwg21CJb6oCgt2ANv3jnUsoXm4258/IejX++eo0CDMdcI/LbgpPuXH8sdec8BIdf4sgccwsN0aFO9POCgGTIOmWhFFGE9j/p1jtWFEW52DSNyByCAXLPUNc+f9Oq8nmrfNCYje7+o1lt2m7m2haCF2SVnFL6kw2/pBzHEH0rEH0oI8q9BF220nWEaSdnjfNaRDDCtcM+WZnsDgUl4lx/BuKxv6rYY0XBwcmHp8deh7EVarWmQ7uC2Glre/TweI0VvTk5xaTx+wWX66Gs",
	},
	{
		"e94db0f9ffca44dc7bade6a3591f544183395a7c",
		plumbing.TreeObject,
		"MTAwNjQ0IFRlc3QgMS50eHQAqKlAYn0TJpWpdp34g/hZkvD/SkMxMDA2NDQgVGVzdCAyLnR4dABNwhdIAaxKPTaIYhD9CG++E0z3sjEwMDY0NCBUZXN0IDMudHh0ABPm9H3Vd5i/3HKNkfXG1/QMW7X8MTAwNjQ0IFRlc3QgNC50eHQAcqe8RmerBo6VQXJDe5k9n7qhN8sxMDA2NDQgVGVzdCA1LnR4dAC7K0DoXsBFXR3nLa/3FYPw3XKjPw==",
		"eAErKUpNVTC0NGAwNDAwMzFRCEktLlEw1CupKGFYsdIhqVZYberKsrk/mn9ETvrw38sZWZURWJXvIXEPxjVetmYdSQJ/OfL3Cft834SsyhisSvjZl9qr5TP23ynqnfj12PUvPNFb/yCrMgGrKlq+xy19NVvfVMci5+qZtvN3LTQ/jazKFKxqt7bDi7gDrrGyz3XXfxdt/nC3aLE9AA2STmk=",
	},
	{
		"9d7f8a56eaf92469dee8a856e716a03387ddb076",
		plumbing.CommitObject,
		"dHJlZSBlOTRkYjBmOWZmY2E0NGRjN2JhZGU2YTM1OTFmNTQ0MTgzMzk1YTdjCmF1dGhvciBKb3NodWEgU2pvZGluZyA8am9zaHVhLnNqb2RpbmdAc2NqYWxsaWFuY2UuY29tPiAxNDU2NTMxNTgzIC0wODAwCmNvbW1pdHRlciBKb3NodWEgU2pvZGluZyA8am9zaHVhLnNqb2RpbmdAc2NqYWxsaWFuY2UuY29tPiAxNDU2NTMxNTgzIC0wODAwCgpUZXN0IENvbW1pdAo=",
		"eAGtjksOgjAUAF33FO8CktZ+aBNjTNy51Qs8Xl8FAjSh5f4SvILLmcVkKM/zUOEi3amuzMDBxE6mkBKhMZHaDiM71DaoZI1RXutgsSWBW+3zCs9c+g3hNeY4LB+4jgc35cf3QiNO04ALcUN5voEy1lmtrNdwll5Ksdt9oPIfUuLNpcLjCIov3ApFmQ==",
	},
}

func Test(t *testing.T) { TestingT(t) }
