package gomoddirectives

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os/exec"

	"golang.org/x/mod/modfile"
)

type modInfo struct {
	Path      string `json:"Path"`
	Dir       string `json:"Dir"`
	GoMod     string `json:"GoMod"`
	GoVersion string `json:"GoVersion"`
	Main      bool   `json:"Main"`
}

// GetModuleFile gets module file.
func GetModuleFile() (*modfile.File, error) {
	// https://github.com/golang/go/issues/44753#issuecomment-790089020
	cmd := exec.Command("go", "list", "-m", "-json")

	raw, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("command go list: %w: %s", err, string(raw))
	}

	var v modInfo
	err = json.Unmarshal(raw, &v)
	if err != nil {
		return nil, fmt.Errorf("unmarshaling error: %w: %s", err, string(raw))
	}

	if v.GoMod == "" {
		return nil, errors.New("working directory is not part of a module")
	}

	raw, err = ioutil.ReadFile(v.GoMod)
	if err != nil {
		return nil, fmt.Errorf("reading go.mod file: %w", err)
	}

	return modfile.Parse("go.mod", raw, nil)
}
