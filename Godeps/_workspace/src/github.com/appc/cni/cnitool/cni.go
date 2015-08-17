package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/appc/cni"
)

const (
	EnvCNIPath = "CNI_PATH"
	EnvNetDir  = "NETCONFPATH"

	DefaultNetDir = "/etc/cni/net.d"

	CmdAdd = "add"
	CmdDel = "del"
)

func listConfFiles(dir string) ([]string, error) {
	// In part, from rkt/networking/podenv.go#listFiles
	files, err := ioutil.ReadDir(dir)
	switch {
	case err == nil: // break
	case os.IsNotExist(err):
		return nil, nil
	default:
		return nil, err
	}

	confFiles := []string{}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if filepath.Ext(f.Name()) == ".conf" {
			confFiles = append(confFiles, filepath.Join(dir, f.Name()))
		}
	}
	return confFiles, nil
}

func loadNetConf(dir, name string) (*cni.NetworkConfig, error) {
	files, err := listConfFiles(dir)
	switch {
	case err != nil:
		return nil, err
	case files == nil || len(files) == 0:
		return nil, fmt.Errorf("No net configurations found")
	}
	sort.Strings(files)

	for _, confFile := range files {
		conf, err := cni.ConfFromFile(confFile)
		if err != nil {
			return nil, err
		}
		if conf.Name == name {
			return conf, nil
		}
	}
	return nil, fmt.Errorf(`no net configuration with name "%s" in %s`, name, dir)
}

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}

	netdir := os.Getenv(EnvNetDir)
	if netdir == "" {
		netdir = DefaultNetDir
	}
	netconf, err := loadNetConf(netdir, os.Args[2])
	if err != nil {
		exit(err)
	}

	netns := os.Args[3]

	cninet := &cni.CNIConfig{
		Path: strings.Split(os.Getenv(EnvCNIPath), ":"),
	}

	rt := &cni.RuntimeConf{
		ContainerID: "cni",
		NetNS:       netns,
		IfName:      "eth0",
		Args:        "",
	}

	switch os.Args[1] {
	case CmdAdd:
		_, err := cninet.AddNetwork(netconf, rt)
		exit(err)
	case CmdDel:
		exit(cninet.DelNetwork(netconf, rt))
	}
}

func usage() {
	exe := filepath.Base(os.Args[0])

	fmt.Fprintf(os.Stderr, "%s: Add or remove network interfaces from a network namespace\n", exe)
	fmt.Fprintf(os.Stderr, "  %s %s <net> <netns>\n", exe, CmdAdd)
	fmt.Fprintf(os.Stderr, "  %s %s <net> <netns>\n", exe, CmdDel)
	os.Exit(1)
}

func exit(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
	os.Exit(0)
}
