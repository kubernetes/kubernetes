// Package command contains the set of Dockerfile commands.
package command

// Define constants for the command strings
const (
	Add         = "add"
	Arg         = "arg"
	Cmd         = "cmd"
	Copy        = "copy"
	Entrypoint  = "entrypoint"
	Env         = "env"
	Expose      = "expose"
	From        = "from"
	Healthcheck = "healthcheck"
	Label       = "label"
	Maintainer  = "maintainer"
	Onbuild     = "onbuild"
	Run         = "run"
	Shell       = "shell"
	StopSignal  = "stopsignal"
	User        = "user"
	Volume      = "volume"
	Workdir     = "workdir"
)

// Commands is list of all Dockerfile commands
var Commands = map[string]struct{}{
	Add:         {},
	Arg:         {},
	Cmd:         {},
	Copy:        {},
	Entrypoint:  {},
	Env:         {},
	Expose:      {},
	From:        {},
	Healthcheck: {},
	Label:       {},
	Maintainer:  {},
	Onbuild:     {},
	Run:         {},
	Shell:       {},
	StopSignal:  {},
	User:        {},
	Volume:      {},
	Workdir:     {},
}
