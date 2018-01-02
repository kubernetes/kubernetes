package main

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/BurntSushi/toml"
	"github.com/urfave/cli"
)

func loadRelease(path string) (*release, error) {
	var r release
	if _, err := toml.DecodeFile(path, &r); err != nil {
		if os.IsNotExist(err) {
			return nil, errors.New("please specify the release file as the first argument")
		}
		return nil, err
	}
	return &r, nil
}

func parseTag(path string) string {
	return strings.TrimSuffix(filepath.Base(path), ".toml")
}

func parseDependencies(r io.Reader) ([]dependency, error) {
	var deps []dependency
	s := bufio.NewScanner(r)
	for s.Scan() {
		ln := strings.TrimSpace(s.Text())
		if strings.HasPrefix(ln, "#") || ln == "" {
			continue
		}
		cidx := strings.Index(ln, "#")
		if cidx > 0 {
			ln = ln[:cidx]
		}
		ln = strings.TrimSpace(ln)
		parts := strings.Fields(ln)
		if len(parts) != 2 && len(parts) != 3 {
			return nil, fmt.Errorf("invalid config format: %s", ln)
		}
		deps = append(deps, dependency{
			Name:   parts[0],
			Commit: parts[1],
		})
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return deps, nil
}

func getPreviousDeps(previous string) ([]dependency, error) {
	r, err := fileFromRev(previous, vendorConf)
	if err != nil {
		return nil, err
	}
	return parseDependencies(r)
}

func changelog(previous, commit string) ([]change, error) {
	raw, err := getChangelog(previous, commit)
	if err != nil {
		return nil, err
	}
	return parseChangelog(raw)
}

func getChangelog(previous, commit string) ([]byte, error) {
	return git("log", "--oneline", fmt.Sprintf("%s..%s", previous, commit))
}

func parseChangelog(changelog []byte) ([]change, error) {
	var (
		changes []change
		s       = bufio.NewScanner(bytes.NewReader(changelog))
	)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		changes = append(changes, change{
			Commit:      fields[0],
			Description: strings.Join(fields[1:], " "),
		})
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return changes, nil
}

func fileFromRev(rev, file string) (io.Reader, error) {
	p, err := git("show", fmt.Sprintf("%s:%s", rev, file))
	if err != nil {
		return nil, err
	}

	return bytes.NewReader(p), nil
}

func git(args ...string) ([]byte, error) {
	o, err := exec.Command("git", args...).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s: %s", err, o)
	}
	return o, nil
}

func updatedDeps(previous, deps []dependency) []dependency {
	var updated []dependency
	pm, cm := toDepMap(previous), toDepMap(deps)
	for name, c := range cm {
		d, ok := pm[name]
		if !ok {
			// it is a new dep and should be noted
			updated = append(updated, c)
			continue
		}
		// it exists, see if its updated
		if d.Commit != c.Commit {
			// set the previous commit
			c.Previous = d.Commit
			updated = append(updated, c)
		}
	}
	return updated
}

func toDepMap(deps []dependency) map[string]dependency {
	out := make(map[string]dependency)
	for _, d := range deps {
		out[d.Name] = d
	}
	return out
}

func getContributors(previous, commit string) ([]string, error) {
	raw, err := git("log", "--format=%aN", fmt.Sprintf("%s..%s", previous, commit))
	if err != nil {
		return nil, err
	}
	var (
		set = make(map[string]struct{})
		s   = bufio.NewScanner(bytes.NewReader(raw))
		out []string
	)
	for s.Scan() {
		set[s.Text()] = struct{}{}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	for name := range set {
		out = append(out, name)
	}
	sort.Strings(out)
	return out, nil
}

// getTemplate will use a builtin template if the template is not specified on the cli
func getTemplate(context *cli.Context) (string, error) {
	path := context.GlobalString("template")
	f, err := os.Open(path)
	if err != nil {
		// if the template file does not exist and the path is for the default template then
		// return the compiled in template
		if os.IsNotExist(err) && path == defaultTemplateFile {
			return releaseNotes, nil
		}
		return "", err
	}
	defer f.Close()
	data, err := ioutil.ReadAll(f)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
