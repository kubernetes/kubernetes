package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"

	"github.com/docker/docker/cliconfig"
	"github.com/docker/docker/opts"
	flag "github.com/docker/docker/pkg/mflag"
	"github.com/docker/docker/pkg/tlsconfig"
)

type command struct {
	name        string
	description string
}

type byName []command

func (a byName) Len() int           { return len(a) }
func (a byName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byName) Less(i, j int) bool { return a[i].name < a[j].name }

var (
	dockerCertPath  = os.Getenv("DOCKER_CERT_PATH")
	dockerTlSVerify = os.Getenv("DOCKER_TLS_VERIFY") != ""

	dockerCommands = []command{
		{"attach", "Attach to a running container"},
		{"build", "Build an image from a Dockerfile"},
		{"commit", "Create a new image from a container's changes"},
		{"cp", "Copy files/folders from a container to a HOSTDIR or to STDOUT"},
		{"create", "Create a new container"},
		{"diff", "Inspect changes on a container's filesystem"},
		{"events", "Get real time events from the server"},
		{"exec", "Run a command in a running container"},
		{"export", "Export a container's filesystem as a tar archive"},
		{"history", "Show the history of an image"},
		{"images", "List images"},
		{"import", "Import the contents from a tarball to create a filesystem image"},
		{"info", "Display system-wide information"},
		{"inspect", "Return low-level information on a container or image"},
		{"kill", "Kill a running container"},
		{"load", "Load an image from a tar archive or STDIN"},
		{"login", "Register or log in to a Docker registry"},
		{"logout", "Log out from a Docker registry"},
		{"logs", "Fetch the logs of a container"},
		{"port", "List port mappings or a specific mapping for the CONTAINER"},
		{"pause", "Pause all processes within a container"},
		{"ps", "List containers"},
		{"pull", "Pull an image or a repository from a registry"},
		{"push", "Push an image or a repository to a registry"},
		{"rename", "Rename a container"},
		{"restart", "Restart a running container"},
		{"rm", "Remove one or more containers"},
		{"rmi", "Remove one or more images"},
		{"run", "Run a command in a new container"},
		{"save", "Save an image(s) to a tar archive"},
		{"search", "Search the Docker Hub for images"},
		{"start", "Start one or more stopped containers"},
		{"stats", "Display a live stream of container(s) resource usage statistics"},
		{"stop", "Stop a running container"},
		{"tag", "Tag an image into a repository"},
		{"top", "Display the running processes of a container"},
		{"unpause", "Unpause all processes within a container"},
		{"version", "Show the Docker version information"},
		{"wait", "Block until a container stops, then print its exit code"},
	}
)

func init() {
	if dockerCertPath == "" {
		dockerCertPath = cliconfig.ConfigDir()
	}
}

func getDaemonConfDir() string {
	// TODO: update for Windows daemon
	if runtime.GOOS == "windows" {
		return cliconfig.ConfigDir()
	}
	return "/etc/docker"
}

var (
	flConfigDir = flag.String([]string{"-config"}, cliconfig.ConfigDir(), "Location of client config files")
	flVersion   = flag.Bool([]string{"v", "-version"}, false, "Print version information and quit")
	flDaemon    = flag.Bool([]string{"d", "-daemon"}, false, "Enable daemon mode")
	flDebug     = flag.Bool([]string{"D", "-debug"}, false, "Enable debug mode")
	flLogLevel  = flag.String([]string{"l", "-log-level"}, "info", "Set the logging level")
	flTLS       = flag.Bool([]string{"-tls"}, false, "Use TLS; implied by --tlsverify")
	flHelp      = flag.Bool([]string{"h", "-help"}, false, "Print usage")
	flTLSVerify = flag.Bool([]string{"-tlsverify"}, dockerTlSVerify, "Use TLS and verify the remote")

	// these are initialized in init() below since their default values depend on dockerCertPath which isn't fully initialized until init() runs
	tlsOptions tlsconfig.Options
	flTrustKey *string
	flHosts    []string
)

func setDefaultConfFlag(flag *string, def string) {
	if *flag == "" {
		if *flDaemon {
			*flag = filepath.Join(getDaemonConfDir(), def)
		} else {
			*flag = filepath.Join(cliconfig.ConfigDir(), def)
		}
	}
}

func init() {
	var placeholderTrustKey string
	// TODO use flag flag.String([]string{"i", "-identity"}, "", "Path to libtrust key file")
	flTrustKey = &placeholderTrustKey

	flag.StringVar(&tlsOptions.CAFile, []string{"-tlscacert"}, filepath.Join(dockerCertPath, defaultCaFile), "Trust certs signed only by this CA")
	flag.StringVar(&tlsOptions.CertFile, []string{"-tlscert"}, filepath.Join(dockerCertPath, defaultCertFile), "Path to TLS certificate file")
	flag.StringVar(&tlsOptions.KeyFile, []string{"-tlskey"}, filepath.Join(dockerCertPath, defaultKeyFile), "Path to TLS key file")
	opts.HostListVar(&flHosts, []string{"H", "-host"}, "Daemon socket(s) to connect to")

	flag.Usage = func() {
		fmt.Fprint(os.Stdout, "Usage: docker [OPTIONS] COMMAND [arg...]\n\nA self-sufficient runtime for containers.\n\nOptions:\n")

		flag.CommandLine.SetOutput(os.Stdout)
		flag.PrintDefaults()

		help := "\nCommands:\n"

		sort.Sort(byName(dockerCommands))

		for _, cmd := range dockerCommands {
			help += fmt.Sprintf("    %-10.10s%s\n", cmd.name, cmd.description)
		}

		help += "\nRun 'docker COMMAND --help' for more information on a command."
		fmt.Fprintf(os.Stdout, "%s\n", help)
	}
}
