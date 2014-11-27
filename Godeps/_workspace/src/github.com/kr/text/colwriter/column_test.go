package colwriter

import (
	"bytes"
	"testing"
)

var src = `
.git
.gitignore
.godir
Procfile:
README.md
api.go
apps.go
auth.go
darwin.go
data.go
dyno.go:
env.go
git.go
help.go
hkdist
linux.go
ls.go
main.go
plugin.go
run.go
scale.go
ssh.go
tail.go
term
unix.go
update.go
version.go
windows.go
`[1:]

var tests = []struct{
	wid  int
	flag uint
	src  string
	want string
}{
	{80, 0, "", ""},
	{80, 0, src, `
.git       README.md  darwin.go  git.go     ls.go      scale.go   unix.go
.gitignore api.go     data.go    help.go    main.go    ssh.go     update.go
.godir     apps.go    dyno.go:   hkdist     plugin.go  tail.go    version.go
Procfile:  auth.go    env.go     linux.go   run.go     term       windows.go
`[1:]},
	{80, BreakOnColon, src, `
.git       .gitignore .godir

Procfile:
README.md api.go    apps.go   auth.go   darwin.go data.go

dyno.go:
env.go     hkdist     main.go    scale.go   term       version.go
git.go     linux.go   plugin.go  ssh.go     unix.go    windows.go
help.go    ls.go      run.go     tail.go    update.go
`[1:]},
	{20, 0, `
Hello
Γειά σου
안녕
今日は
`[1:], `
Hello    안녕
Γειά σου 今日は
`[1:]},
}

func TestWriter(t *testing.T) {
	for _, test := range tests {
		b := new(bytes.Buffer)
		w := NewWriter(b, test.wid, test.flag)
		if _, err := w.Write([]byte(test.src)); err != nil {
			t.Error(err)
		}
		if err := w.Flush(); err != nil {
			t.Error(err)
		}
		if g := b.String(); test.want != g {
			t.Log("\n" + test.want)
			t.Log("\n" + g)
			t.Errorf("%q != %q", test.want, g)
		}
	}
}
