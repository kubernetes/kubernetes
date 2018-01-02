// +build linux

package apparmor

import (
	"bufio"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/docker/docker/pkg/aaparser"
	"github.com/docker/docker/pkg/templates"
)

var (
	// profileDirectory is the file store for apparmor profiles and macros.
	profileDirectory = "/etc/apparmor.d"
)

// profileData holds information about the given profile for generation.
type profileData struct {
	// Name is profile name.
	Name string
	// Imports defines the apparmor functions to import, before defining the profile.
	Imports []string
	// InnerImports defines the apparmor functions to import in the profile.
	InnerImports []string
	// Version is the {major, minor, patch} version of apparmor_parser as a single number.
	Version int
}

// generateDefault creates an apparmor profile from ProfileData.
func (p *profileData) generateDefault(out io.Writer) error {
	compiled, err := templates.NewParse("apparmor_profile", baseTemplate)
	if err != nil {
		return err
	}

	if macroExists("tunables/global") {
		p.Imports = append(p.Imports, "#include <tunables/global>")
	} else {
		p.Imports = append(p.Imports, "@{PROC}=/proc/")
	}

	if macroExists("abstractions/base") {
		p.InnerImports = append(p.InnerImports, "#include <abstractions/base>")
	}

	ver, err := aaparser.GetVersion()
	if err != nil {
		return err
	}
	p.Version = ver

	return compiled.Execute(out, p)
}

// macrosExists checks if the passed macro exists.
func macroExists(m string) bool {
	_, err := os.Stat(path.Join(profileDirectory, m))
	return err == nil
}

// InstallDefault generates a default profile in a temp directory determined by
// os.TempDir(), then loads the profile into the kernel using 'apparmor_parser'.
func InstallDefault(name string) error {
	p := profileData{
		Name: name,
	}

	// Install to a temporary directory.
	f, err := ioutil.TempFile("", name)
	if err != nil {
		return err
	}
	profilePath := f.Name()

	defer f.Close()
	defer os.Remove(profilePath)

	if err := p.generateDefault(f); err != nil {
		return err
	}

	return aaparser.LoadProfile(profilePath)
}

// IsLoaded checks if a profile with the given name has been loaded into the
// kernel.
func IsLoaded(name string) (bool, error) {
	file, err := os.Open("/sys/kernel/security/apparmor/profiles")
	if err != nil {
		return false, err
	}
	defer file.Close()

	r := bufio.NewReader(file)
	for {
		p, err := r.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return false, err
		}
		if strings.HasPrefix(p, name+" ") {
			return true, nil
		}
	}

	return false, nil
}
