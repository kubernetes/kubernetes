// Copyright © 2015 Steve Francia <spf@spf13.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/spf13/viper"
)

// var BaseDir = ""
// var AppName = ""
// var CommandDir = ""

var funcMap template.FuncMap
var projectPath = ""
var inputPath = ""
var projectBase = ""

// for testing only
var testWd = ""

var cmdDirs = []string{"cmd", "cmds", "command", "commands"}

func init() {
	funcMap = template.FuncMap{
		"comment": commentifyString,
	}
}

func er(msg interface{}) {
	fmt.Println("Error:", msg)
	os.Exit(-1)
}

// Check if a file or directory exists.
func exists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func ProjectPath() string {
	if projectPath == "" {
		guessProjectPath()
	}

	return projectPath
}

// wrapper of the os package so we can test better
func getWd() (string, error) {
	if testWd == "" {
		return os.Getwd()
	}
	return testWd, nil
}

func guessCmdDir() string {
	guessProjectPath()
	if b, _ := isEmpty(projectPath); b {
		return "cmd"
	}

	files, _ := filepath.Glob(projectPath + string(os.PathSeparator) + "c*")
	for _, f := range files {
		for _, c := range cmdDirs {
			if f == c {
				return c
			}
		}
	}

	return "cmd"
}

func guessImportPath() string {
	guessProjectPath()

	if !strings.HasPrefix(projectPath, getSrcPath()) {
		er("Cobra only supports project within $GOPATH")
	}

	return filepath.ToSlash(filepath.Clean(strings.TrimPrefix(projectPath, getSrcPath())))
}

func getSrcPath() string {
	return filepath.Join(os.Getenv("GOPATH"), "src") + string(os.PathSeparator)
}

func projectName() string {
	return filepath.Base(ProjectPath())
}

func guessProjectPath() {
	// if no path is provided... assume CWD.
	if inputPath == "" {
		x, err := getWd()
		if err != nil {
			er(err)
		}

		// inspect CWD
		base := filepath.Base(x)

		// if we are in the cmd directory.. back up
		for _, c := range cmdDirs {
			if base == c {
				projectPath = filepath.Dir(x)
				return
			}
		}

		if projectPath == "" {
			projectPath = filepath.Clean(x)
			return
		}
	}

	srcPath := getSrcPath()
	// if provided, inspect for logical locations
	if strings.ContainsRune(inputPath, os.PathSeparator) {
		if filepath.IsAbs(inputPath) || filepath.HasPrefix(inputPath, string(os.PathSeparator)) {
			// if Absolute, use it
			projectPath = filepath.Clean(inputPath)
			return
		}
		// If not absolute but contains slashes,
		// assuming it means create it from $GOPATH
		count := strings.Count(inputPath, string(os.PathSeparator))

		switch count {
		// If only one directory deep, assume "github.com"
		case 1:
			projectPath = filepath.Join(srcPath, "github.com", inputPath)
			return
		case 2:
			projectPath = filepath.Join(srcPath, inputPath)
			return
		default:
			er("Unknown directory")
		}
	} else {
		// hardest case.. just a word.
		if projectBase == "" {
			x, err := getWd()
			if err == nil {
				projectPath = filepath.Join(x, inputPath)
				return
			}
			er(err)
		} else {
			projectPath = filepath.Join(srcPath, projectBase, inputPath)
			return
		}
	}
}

// isEmpty checks if a given path is empty.
func isEmpty(path string) (bool, error) {
	if b, _ := exists(path); !b {
		return false, fmt.Errorf("%q path does not exist", path)
	}
	fi, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	if fi.IsDir() {
		f, err := os.Open(path)
		// FIX: Resource leak - f.close() should be called here by defer or is missed
		// if the err != nil branch is taken.
		defer f.Close()
		if err != nil {
			return false, err
		}
		list, _ := f.Readdir(-1)
		// f.Close() - see bug fix above
		return len(list) == 0, nil
	}
	return fi.Size() == 0, nil
}

// isDir checks if a given path is a directory.
func isDir(path string) (bool, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	return fi.IsDir(), nil
}

// dirExists checks if a path exists and is a directory.
func dirExists(path string) (bool, error) {
	fi, err := os.Stat(path)
	if err == nil && fi.IsDir() {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func writeTemplateToFile(path string, file string, template string, data interface{}) error {
	filename := filepath.Join(path, file)

	r, err := templateToReader(template, data)

	if err != nil {
		return err
	}

	err = safeWriteToDisk(filename, r)

	if err != nil {
		return err
	}
	return nil
}

func writeStringToFile(path, file, text string) error {
	filename := filepath.Join(path, file)

	r := strings.NewReader(text)
	err := safeWriteToDisk(filename, r)

	if err != nil {
		return err
	}
	return nil
}

func templateToReader(tpl string, data interface{}) (io.Reader, error) {
	tmpl := template.New("")
	tmpl.Funcs(funcMap)
	tmpl, err := tmpl.Parse(tpl)

	if err != nil {
		return nil, err
	}
	buf := new(bytes.Buffer)
	err = tmpl.Execute(buf, data)

	return buf, err
}

// Same as WriteToDisk but checks to see if file/directory already exists.
func safeWriteToDisk(inpath string, r io.Reader) (err error) {
	dir, _ := filepath.Split(inpath)
	ospath := filepath.FromSlash(dir)

	if ospath != "" {
		err = os.MkdirAll(ospath, 0777) // rwx, rw, r
		if err != nil {
			return
		}
	}

	ex, err := exists(inpath)
	if err != nil {
		return
	}
	if ex {
		return fmt.Errorf("%v already exists", inpath)
	}

	file, err := os.Create(inpath)
	if err != nil {
		return
	}
	defer file.Close()

	_, err = io.Copy(file, r)
	return
}

func getLicense() License {
	l := whichLicense()
	if l != "" {
		if x, ok := Licenses[l]; ok {
			return x
		}
	}

	return Licenses["apache"]
}

func whichLicense() string {
	// if explicitly flagged, use that
	if userLicense != "" {
		return matchLicense(userLicense)
	}

	// if already present in the project, use that
	// TODO: Inspect project for existing license

	// default to viper's setting

	if viper.IsSet("license.header") || viper.IsSet("license.text") {
		if custom, ok := Licenses["custom"]; ok {
			custom.Header = viper.GetString("license.header")
			custom.Text = viper.GetString("license.text")
			Licenses["custom"] = custom
			return "custom"
		}
	}

	return matchLicense(viper.GetString("license"))
}

func copyrightLine() string {
	author := viper.GetString("author")
	year := time.Now().Format("2006")

	return "Copyright © " + year + " " + author
}

func commentifyString(in string) string {
	var newlines []string
	lines := strings.Split(in, "\n")
	for _, x := range lines {
		if !strings.HasPrefix(x, "//") {
			if x != "" {
				newlines = append(newlines, "// "+x)
			} else {
				newlines = append(newlines, "//")
			}
		} else {
			newlines = append(newlines, x)
		}
	}
	return strings.Join(newlines, "\n")
}
