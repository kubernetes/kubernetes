package containerd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

const (
	defaultAddress = `\\.\pipe\containerd-containerd-test`
	testImage      = "docker.io/library/go:nanoserver"
)

var (
	dockerLayerFolders []string

	defaultRoot  = filepath.Join(os.Getenv("programfiles"), "containerd", "root-test")
	defaultState = filepath.Join(os.Getenv("programfiles"), "containerd", "state-test")
)

func platformTestSetup(client *Client) error {
	var (
		roots       []string
		layerChains = make(map[string]string)
	)
	// Since we can't pull images yet, we'll piggyback on the default
	// docker's images
	wfPath := `C:\ProgramData\docker\windowsfilter`
	wf, err := os.Open(wfPath)
	if err != nil {
		return errors.Wrapf(err, "failed to access docker layers @ %s", wfPath)
	}
	defer wf.Close()
	entries, err := wf.Readdirnames(0)
	if err != nil {
		return errors.Wrapf(err, "failed to read %s entries", wfPath)
	}

	for _, fn := range entries {
		layerChainPath := filepath.Join(wfPath, fn, "layerchain.json")
		lfi, err := os.Stat(layerChainPath)
		switch {
		case err == nil && lfi.Mode().IsRegular():
			f, err := os.OpenFile(layerChainPath, os.O_RDONLY, 0660)
			if err != nil {
				fmt.Fprintln(os.Stderr,
					errors.Wrapf(err, "failed to open %s", layerChainPath))
				continue
			}
			defer f.Close()
			l := make([]string, 0)
			if err := json.NewDecoder(f).Decode(&l); err != nil {
				fmt.Fprintln(os.Stderr,
					errors.Wrapf(err, "failed to decode %s", layerChainPath))
				continue
			}
			switch {
			case len(l) == 1:
				layerChains[l[0]] = filepath.Join(wfPath, fn)
			case len(l) > 1:
				fmt.Fprintf(os.Stderr, "Too many entries in %s: %d", layerChainPath, len(l))
			case len(l) == 0:
				roots = append(roots, filepath.Join(wfPath, fn))
			}
		case os.IsNotExist(err):
			// keep on going
		default:
			return errors.Wrapf(err, "error trying to access %s", layerChainPath)
		}
	}

	// They'll be 2 roots, just take the first one
	l := roots[0]
	dockerLayerFolders = append(dockerLayerFolders, l)
	for {
		l = layerChains[l]
		if l == "" {
			break
		}

		dockerLayerFolders = append([]string{l}, dockerLayerFolders...)
	}

	return nil
}
