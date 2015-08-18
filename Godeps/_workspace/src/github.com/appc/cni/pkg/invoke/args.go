package invoke

import (
	"os"
	"strings"
)

type CNIArgs interface {
	// For use with os/exec; i.e., return nil to inherit the
	// environment from this process
	AsEnv() []string
}

type inherited struct{}

var inheritArgsFromEnv inherited

func (_ *inherited) AsEnv() []string {
	return nil
}

func ArgsFromEnv() CNIArgs {
	return &inheritArgsFromEnv
}

type Args struct {
	Command       string
	ContainerID   string
	NetNS         string
	PluginArgs    [][2]string
	PluginArgsStr string
	IfName        string
	Path          string
}

func (args *Args) AsEnv() []string {
	env := os.Environ()
	pluginArgsStr := args.PluginArgsStr
	if pluginArgsStr == "" {
		pluginArgsStr = stringify(args.PluginArgs)
	}

	env = append(env,
		"CNI_COMMAND="+args.Command,
		"CNI_CONTAINERID="+args.ContainerID,
		"CNI_NETNS="+args.NetNS,
		"CNI_ARGS="+pluginArgsStr,
		"CNI_IFNAME="+args.IfName,
		"CNI_PATH="+args.Path)
	return env
}

// taken from rkt/networking/net_plugin.go
func stringify(pluginArgs [][2]string) string {
	entries := make([]string, len(pluginArgs))

	for i, kv := range pluginArgs {
		entries[i] = strings.Join(kv[:], "=")
	}

	return strings.Join(entries, ";")
}
