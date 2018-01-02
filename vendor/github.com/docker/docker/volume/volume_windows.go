package volume

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// read-write modes
var rwModes = map[string]bool{
	"rw": true,
}

// read-only modes
var roModes = map[string]bool{
	"ro": true,
}

var platformRawValidationOpts = []func(*validateOpts){
	// filepath.IsAbs is weird on Windows:
	//	`c:` is not considered an absolute path
	//	`c:\` is considered an absolute path
	// In any case, the regex matching below ensures absolute paths
	// TODO: consider this a bug with filepath.IsAbs (?)
	func(o *validateOpts) { o.skipAbsolutePathCheck = true },
}

const (
	// Spec should be in the format [source:]destination[:mode]
	//
	// Examples: c:\foo bar:d:rw
	//           c:\foo:d:\bar
	//           myname:d:
	//           d:\
	//
	// Explanation of this regex! Thanks @thaJeztah on IRC and gist for help. See
	// https://gist.github.com/thaJeztah/6185659e4978789fb2b2. A good place to
	// test is https://regex-golang.appspot.com/assets/html/index.html
	//
	// Useful link for referencing named capturing groups:
	// http://stackoverflow.com/questions/20750843/using-named-matches-from-go-regex
	//
	// There are three match groups: source, destination and mode.
	//

	// RXHostDir is the first option of a source
	RXHostDir = `[a-z]:\\(?:[^\\/:*?"<>|\r\n]+\\?)*`
	// RXName is the second option of a source
	RXName = `[^\\/:*?"<>|\r\n]+`
	// RXReservedNames are reserved names not possible on Windows
	RXReservedNames = `(con)|(prn)|(nul)|(aux)|(com[1-9])|(lpt[1-9])`

	// RXSource is the combined possibilities for a source
	RXSource = `((?P<source>((` + RXHostDir + `)|(` + RXName + `))):)?`

	// Source. Can be either a host directory, a name, or omitted:
	//  HostDir:
	//    -  Essentially using the folder solution from
	//       https://www.safaribooksonline.com/library/view/regular-expressions-cookbook/9781449327453/ch08s18.html
	//       but adding case insensitivity.
	//    -  Must be an absolute path such as c:\path
	//    -  Can include spaces such as `c:\program files`
	//    -  And then followed by a colon which is not in the capture group
	//    -  And can be optional
	//  Name:
	//    -  Must not contain invalid NTFS filename characters (https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx)
	//    -  And then followed by a colon which is not in the capture group
	//    -  And can be optional

	// RXDestination is the regex expression for the mount destination
	RXDestination = `(?P<destination>([a-z]):((?:\\[^\\/:*?"<>\r\n]+)*\\?))`
	// Destination (aka container path):
	//    -  Variation on hostdir but can be a drive followed by colon as well
	//    -  If a path, must be absolute. Can include spaces
	//    -  Drive cannot be c: (explicitly checked in code, not RegEx)

	// RXMode is the regex expression for the mode of the mount
	// Mode (optional):
	//    -  Hopefully self explanatory in comparison to above regex's.
	//    -  Colon is not in the capture group
	RXMode = `(:(?P<mode>(?i)ro|rw))?`
)

// BackwardsCompatible decides whether this mount point can be
// used in old versions of Docker or not.
// Windows volumes are never backwards compatible.
func (m *MountPoint) BackwardsCompatible() bool {
	return false
}

func splitRawSpec(raw string) ([]string, error) {
	specExp := regexp.MustCompile(`^` + RXSource + RXDestination + RXMode + `$`)
	match := specExp.FindStringSubmatch(strings.ToLower(raw))

	// Must have something back
	if len(match) == 0 {
		return nil, errInvalidSpec(raw)
	}

	var split []string
	matchgroups := make(map[string]string)
	// Pull out the sub expressions from the named capture groups
	for i, name := range specExp.SubexpNames() {
		matchgroups[name] = strings.ToLower(match[i])
	}
	if source, exists := matchgroups["source"]; exists {
		if source != "" {
			split = append(split, source)
		}
	}
	if destination, exists := matchgroups["destination"]; exists {
		if destination != "" {
			split = append(split, destination)
		}
	}
	if mode, exists := matchgroups["mode"]; exists {
		if mode != "" {
			split = append(split, mode)
		}
	}
	// Fix #26329. If the destination appears to be a file, and the source is null,
	// it may be because we've fallen through the possible naming regex and hit a
	// situation where the user intention was to map a file into a container through
	// a local volume, but this is not supported by the platform.
	if matchgroups["source"] == "" && matchgroups["destination"] != "" {
		validName, err := IsVolumeNameValid(matchgroups["destination"])
		if err != nil {
			return nil, err
		}
		if !validName {
			if fi, err := os.Stat(matchgroups["destination"]); err == nil {
				if !fi.IsDir() {
					return nil, fmt.Errorf("file '%s' cannot be mapped. Only directories can be mapped on this platform", matchgroups["destination"])
				}
			}
		}
	}
	return split, nil
}

// IsVolumeNameValid checks a volume name in a platform specific manner.
func IsVolumeNameValid(name string) (bool, error) {
	nameExp := regexp.MustCompile(`^` + RXName + `$`)
	if !nameExp.MatchString(name) {
		return false, nil
	}
	nameExp = regexp.MustCompile(`^` + RXReservedNames + `$`)
	if nameExp.MatchString(name) {
		return false, fmt.Errorf("volume name %q cannot be a reserved word for Windows filenames", name)
	}
	return true, nil
}

// ValidMountMode will make sure the mount mode is valid.
// returns if it's a valid mount mode or not.
func ValidMountMode(mode string) bool {
	if mode == "" {
		return true
	}
	return roModes[strings.ToLower(mode)] || rwModes[strings.ToLower(mode)]
}

// ReadWrite tells you if a mode string is a valid read-write mode or not.
func ReadWrite(mode string) bool {
	return rwModes[strings.ToLower(mode)] || mode == ""
}

func validateNotRoot(p string) error {
	p = strings.ToLower(convertSlash(p))
	if p == "c:" || p == `c:\` {
		return fmt.Errorf("destination path cannot be `c:` or `c:\\`: %v", p)
	}
	return nil
}

func validateCopyMode(mode bool) error {
	if mode {
		return fmt.Errorf("Windows does not support copying image path content")
	}
	return nil
}

func convertSlash(p string) string {
	return filepath.FromSlash(p)
}

func clean(p string) string {
	if match, _ := regexp.MatchString("^[a-z]:$", p); match {
		return p
	}
	return filepath.Clean(p)
}

func validateStat(fi os.FileInfo) error {
	if !fi.IsDir() {
		return fmt.Errorf("source path must be a directory")
	}
	return nil
}
