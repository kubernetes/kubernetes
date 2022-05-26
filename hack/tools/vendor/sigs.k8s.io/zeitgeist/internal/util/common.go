/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"

	"github.com/blang/semver"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"

	"sigs.k8s.io/zeitgeist/internal/command"
)

const (
	TagPrefix = "v"
)

var (
	regexpCRLF       *regexp.Regexp = regexp.MustCompile(`\015$`)
	regexpCtrlChar   *regexp.Regexp = regexp.MustCompile(`\x1B[\[(]([0-9]{1,2}(;[0-9]{1,2})?)?[mKB]`)
	regexpOauthToken *regexp.Regexp = regexp.MustCompile(`[a-f0-9]{40}:x-oauth-basic`)
	regexpGitToken   *regexp.Regexp = regexp.MustCompile(`git:[a-f0-9]{35,40}@github.com`)
)

// UserInputError a custom error to handle more user input info
type UserInputError struct {
	ErrorString string
	isCtrlC     bool
}

// Error return the error string
func (e UserInputError) Error() string {
	return e.ErrorString
}

// IsCtrlC return true if the user has hit Ctrl+C
func (e UserInputError) IsCtrlC() bool {
	return e.isCtrlC
}

// NewUserInputError creates a new UserInputError
func NewUserInputError(message string, ctrlC bool) UserInputError {
	return UserInputError{
		ErrorString: message,
		isCtrlC:     ctrlC,
	}
}

// PackagesAvailable takes a slice of packages and determines if they are installed
// on the host OS. Replaces common::check_packages.
func PackagesAvailable(packages ...string) (bool, error) {
	type packageVerifier struct {
		cmd  string
		args []string
	}
	type packageChecker struct {
		manager  string
		verifier *packageVerifier
	}
	var checker *packageChecker

	for _, x := range []struct {
		possiblePackageManagers []string
		verifierCmd             string
		verifierArgs            []string
	}{
		{ // Debian, Ubuntu and similar
			[]string{"apt"},
			"dpkg",
			[]string{"-l"},
		},
		{ // Fedora, openSUSE and similar
			[]string{"dnf", "yum", "zypper"},
			"rpm",
			[]string{"--quiet", "-q"},
		},
		{ // ArchLinux and similar
			[]string{"yay", "pacaur", "pacman"},
			"pacman",
			[]string{"-Qs"},
		},
	} {
		// Find a working package verifier
		if !command.Available(x.verifierCmd) {
			logrus.Debugf("Skipping not available package verifier %s",
				x.verifierCmd)
			continue
		}

		// Find a working package manager
		packageManager := ""
		for _, mgr := range x.possiblePackageManagers {
			if command.Available(mgr) {
				packageManager = mgr
				break
			}
			logrus.Debugf("Skipping not available package manager %s", mgr)
		}
		if packageManager == "" {
			return false, errors.Errorf(
				"unable to find working package manager for verifier `%s`",
				x.verifierCmd,
			)
		}

		checker = &packageChecker{
			manager:  packageManager,
			verifier: &packageVerifier{x.verifierCmd, x.verifierArgs},
		}
		break
	}
	if checker == nil {
		return false, errors.New("unable to find working package manager")
	}
	logrus.Infof("Assuming %q as package manager", checker.manager)

	missingPkgs := []string{}
	for _, pkg := range packages {
		logrus.Infof("Checking if %q has been installed", pkg)

		args := append(checker.verifier.args, pkg)
		if err := command.New(checker.verifier.cmd, args...).
			RunSilentSuccess(); err != nil {
			logrus.Infof("Adding %s to missing packages", pkg)
			missingPkgs = append(missingPkgs, pkg)
		}
	}

	if len(missingPkgs) > 0 {
		logrus.Warnf("The following packages are not installed via %s: %s",
			checker.manager, strings.Join(missingPkgs, ", "))

		// TODO: `install` might not be the install command for every package
		// manager
		logrus.Infof("Install them with: sudo %s install %s",
			checker.manager, strings.Join(missingPkgs, " "))
		return false, nil
	}

	return true, nil
}

/*
#############################################################################
# Simple yes/no prompt
#
# @optparam default -n(default)/-y/-e (default to n, y or make (e)xplicit)
# @param message
common::askyorn () {
  local yorn
  local def=n
  local msg="y/N"

  case $1 in
  -y) # yes default
      def="y" msg="Y/n"
      shift
      ;;
  -e) # Explicit
      def="" msg="y/n"
      shift
      ;;
  -n) shift
      ;;
  esac

  while [[ $yorn != [yYnN] ]]; do
    logecho -n "$*? ($msg): "
    read yorn
    : ${yorn:=$def}
  done

  # Final test to set return code
  [[ $yorn == [yY] ]]
}
*/

// readInput prints a question and then reads an answer from the user
//
// If the user presses Ctrl+C instead of answering, this funtcion will
// return an error crafted with UserInputError. This error can be queried
// to find out if the user canceled the input using its method IsCtrlC:
//
//     if err.(util.UserInputError).IsCtrlC() {}
//
// Note that in case of cancelling input, the user will still have to press
// enter to finish the scan.
func readInput(question string) (string, error) {
	fmt.Print(question)

	// Trap Ctrl+C if a user wishes to cancel the input
	inputChannel := make(chan string, 1)
	signalChannel := make(chan os.Signal, 1)
	signal.Notify(signalChannel, syscall.SIGINT, syscall.SIGTERM)
	defer func() {
		signal.Stop(signalChannel)
		close(signalChannel)
	}()
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		scanner.Scan()
		response := scanner.Text()
		inputChannel <- response
		close(inputChannel)
	}()

	select {
	case <-signalChannel:
		return "", NewUserInputError("Input canceled", true)
	case response := <-inputChannel:
		return response, nil
	}
}

// Ask asks the user a question, expecting a known response expectedResponse
//
// You may specify a single response as a string or a series
// of valid/invalid responses with an optional default.
//
// To specify the valid responses, either pass a string or craft a series
// of answers using the following format:
//
//      "|successAnswers|nonSuccessAnswers|defaultAnswer"
//
// The successAnswers and nonSuccessAnswers can be either a string or a
// series os responses like:
//
//       "|opt1a:opt1b|opt2a:opt2b|defaultAnswer"
//
// This example will accept opt1a and opt1b as successful answers, opt2a and
// opt2b as unsuccessful answers and in case of an empty answer, it will
// return "defaultAnswer" as success.
//
// To consider the default as a success, simply list them with the rest of the
// non successfule answers.
func Ask(question, expectedResponse string, retries int) (answer string, success bool, err error) {
	attempts := 1

	if retries < 0 {
		fmt.Printf("Retries was set to a number less than zero (%d). Please specify a positive number of retries or zero, if you want to ask unconditionally.\n", retries)
	}

	const (
		partsSeparator string = "|"
		optsSeparator  string = ":"
	)

	successAnswers := make([]string, 0)
	nonSuccessAnswers := make([]string, 0)
	defaultAnswer := ""

	// Check out if string has several options
	if strings.Contains(expectedResponse, partsSeparator) {
		parts := strings.Split(expectedResponse, partsSeparator)
		if len(parts) > 3 {
			return "", false, errors.New("answer spec malformed")
		}
		// The first part has the answers to consider a success
		if strings.Contains(expectedResponse, parts[0]) {
			successAnswers = strings.Split(parts[0], optsSeparator)
		}
		// If there is a seconf part, its non success, but expected responses
		if len(parts) >= 2 {
			if strings.Contains(parts[1], optsSeparator) {
				nonSuccessAnswers = strings.Split(parts[1], optsSeparator)
			} else {
				nonSuccessAnswers = append(nonSuccessAnswers, parts[1])
			}
		}
		// If we have a fourth part, its the default answer
		if len(parts) == 3 {
			defaultAnswer = parts[2]
		}
	}

	for attempts <= retries {
		// Read the input from the user
		answer, err = readInput(fmt.Sprintf("%s (%d/%d) \n", question, attempts, retries))
		if err != nil {
			return answer, false, err
		}

		// if we have multiple options, use those and ignore the expected string
		if len(successAnswers) > 0 {
			// check the right answers
			for _, testResponse := range successAnswers {
				if answer == testResponse {
					return answer, true, nil
				}
			}

			// if we have wrong, but accepted answers, try those
			for _, testResponse := range nonSuccessAnswers {
				if answer == testResponse {
					return answer, false, nil
				}

				// If answer is the default, and it is a nonSuccess, return it
				if answer == "" && defaultAnswer == testResponse {
					return defaultAnswer, false, nil
				}
			}
		} else if answer == expectedResponse {
			return answer, true, nil
		}

		if answer == "" && defaultAnswer != "" {
			return defaultAnswer, true, nil
		}

		fmt.Printf("Expected '%s', but got '%s'\n", expectedResponse, answer)

		attempts++
	}

	return answer, false, NewUserInputError("expected response was not input. Retries exceeded", false)
}

// MoreRecent determines if file at path a was modified more recently than file
// at path b. If one file does not exist, the other will be treated as most
// recent. If both files do not exist or an error occurs, an error is returned.
func MoreRecent(a, b string) (bool, error) {
	fileA, errA := os.Stat(a)
	if errA != nil && !os.IsNotExist(errA) {
		return false, errA
	}

	fileB, errB := os.Stat(b)
	if errB != nil && !os.IsNotExist(errB) {
		return false, errB
	}

	switch {
	case os.IsNotExist(errA) && os.IsNotExist(errB):
		return false, errors.New("neither file exists")
	case os.IsNotExist(errA):
		return false, nil
	case os.IsNotExist(errB):
		return true, nil
	}

	return (fileA.ModTime().Unix() >= fileB.ModTime().Unix()), nil
}

func AddTagPrefix(tag string) string {
	if strings.HasPrefix(tag, TagPrefix) {
		return tag
	}
	return TagPrefix + tag
}

func TrimTagPrefix(tag string) string {
	return strings.TrimPrefix(tag, TagPrefix)
}

func TagStringToSemver(tag string) (semver.Version, error) {
	return semver.Make(TrimTagPrefix(tag))
}

func SemverToTagString(tag semver.Version) string {
	return AddTagPrefix(tag.String())
}

// CopyFileLocal copies a local file from one local location to another.
func CopyFileLocal(src, dst string, required bool) error {
	logrus.Infof("Trying to copy file %s to %s (required: %v)", src, dst, required)
	srcStat, err := os.Stat(src)
	if err != nil && required {
		return errors.Wrapf(
			err, "source %s is required but does not exist", src,
		)
	}
	if os.IsNotExist(err) && !required {
		logrus.Infof(
			"File %s does not exist but is also not required",
			filepath.Base(src),
		)
		return nil
	}

	if !srcStat.Mode().IsRegular() {
		return errors.New("cannot copy non-regular file: IsRegular reports " +
			"whether m describes a regular file. That is, it tests that no " +
			"mode type bits are set")
	}

	source, err := os.Open(src)
	if err != nil {
		return errors.Wrapf(err, "open source file %s", src)
	}
	defer source.Close()

	destination, err := os.Create(dst)
	if err != nil {
		return errors.Wrapf(err, "create destination file %s", dst)
	}
	defer destination.Close()
	if _, err := io.Copy(destination, source); err != nil {
		return errors.Wrapf(err, "copy source %s to destination %s", src, dst)
	}
	logrus.Infof("Copied %s", filepath.Base(dst))
	return nil
}

// CopyDirContentsLocal copies local directory contents from one local location
// to another.
func CopyDirContentsLocal(src, dst string) error {
	logrus.Infof("Trying to copy dir %s to %s", src, dst)
	// If initial destination does not exist create it.
	if _, err := os.Stat(dst); err != nil {
		if err := os.MkdirAll(dst, os.FileMode(0o755)); err != nil {
			return errors.Wrapf(err, "create destination directory %s", dst)
		}
	}
	files, err := ioutil.ReadDir(src)
	if err != nil {
		return errors.Wrapf(err, "reading source dir %s", src)
	}
	for _, file := range files {
		srcPath := filepath.Join(src, file.Name())
		dstPath := filepath.Join(dst, file.Name())

		fileInfo, err := os.Stat(srcPath)
		if err != nil {
			return errors.Wrapf(err, "stat source path %s", srcPath)
		}

		switch fileInfo.Mode() & os.ModeType {
		case os.ModeDir:
			if !Exists(dstPath) {
				if err := os.MkdirAll(dstPath, os.FileMode(0o755)); err != nil {
					return errors.Wrapf(err, "creating destination dir %s", dstPath)
				}
			}
			if err := CopyDirContentsLocal(srcPath, dstPath); err != nil {
				return errors.Wrapf(err, "copy %s to %s", srcPath, dstPath)
			}
		default:
			if err := CopyFileLocal(srcPath, dstPath, false); err != nil {
				return errors.Wrapf(err, "copy %s to %s", srcPath, dstPath)
			}
		}
	}
	return nil
}

// RemoveAndReplaceDir removes a directory and its contents then recreates it.
func RemoveAndReplaceDir(path string) error {
	logrus.Infof("Removing %s", path)
	if err := os.RemoveAll(path); err != nil {
		return errors.Wrapf(err, "remove %s", path)
	}
	logrus.Infof("Creating %s", path)
	if err := os.MkdirAll(path, os.FileMode(0o755)); err != nil {
		return errors.Wrapf(err, "create %s", path)
	}
	return nil
}

// Exists indicates whether a file exists.
func Exists(path string) bool {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return false
	}

	return true
}

// WrapText wraps a text
func WrapText(originalText string, lineSize int) (wrappedText string) {
	words := strings.Fields(strings.TrimSpace(originalText))
	wrappedText = words[0]
	spaceLeft := lineSize - len(wrappedText)
	for _, word := range words[1:] {
		if len(word)+1 > spaceLeft {
			wrappedText += "\n" + word
			spaceLeft = lineSize - len(word)
		} else {
			wrappedText += " " + word
			spaceLeft -= 1 + len(word)
		}
	}

	return wrappedText
}

// StripControlCharacters takes a slice of bytes and removes control
// characters and bare line feeds (ported from the original bash anago)
func StripControlCharacters(logData []byte) []byte {
	return regexpCRLF.ReplaceAllLiteral(
		regexpCtrlChar.ReplaceAllLiteral(logData, []byte{}), []byte{},
	)
}

// StripSensitiveData removes data deemed sensitive or non public
// from a byte slice (ported from the original bash anago)
func StripSensitiveData(logData []byte) []byte {
	// Remove OAuth tokens
	logData = regexpOauthToken.ReplaceAllLiteral(logData, []byte("__SANITIZED__:x-oauth-basic"))
	// Remove GitHub tokens
	logData = regexpGitToken.ReplaceAllLiteral(logData, []byte("//git:__SANITIZED__:@github.com"))
	return logData
}

// CleanLogFile cleans control characters and sensitive data from a file
func CleanLogFile(logPath string) (err error) {
	logrus.Debugf("Sanitizing logfile %s", logPath)

	// Open a tempfile to write sanitized log
	tempFile, err := ioutil.TempFile(os.TempDir(), "temp-release-log-")
	if err != nil {
		return errors.Wrap(err, "creating temp file for sanitizing log")
	}
	defer func() {
		err = tempFile.Close()
		os.Remove(tempFile.Name())
	}()

	// Open the new logfile for reading
	logFile, err := os.Open(logPath)
	if err != nil {
		return errors.Wrapf(err, "while opening %s ", logPath)
	}
	// Scan the log and pass it through the cleaning funcs
	scanner := bufio.NewScanner(logFile)
	for scanner.Scan() {
		chunk := scanner.Bytes()
		chunk = StripControlCharacters(
			StripSensitiveData(chunk),
		)
		chunk = append(chunk, []byte{10}...)
		_, err := tempFile.Write(chunk)
		if err != nil {
			return errors.Wrap(err, "while writing buffer to file")
		}
	}
	if err := logFile.Close(); err != nil {
		return errors.Wrap(err, "closing log file")
	}

	if err := CopyFileLocal(tempFile.Name(), logPath, true); err != nil {
		return errors.Wrap(err, "writing clean logfile")
	}

	return err
}
