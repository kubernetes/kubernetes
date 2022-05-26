/*
Copyright 2020 The Kubernetes Authors.

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

// Package dependencies checks dependencies, locally or remotely
package dependencies

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go/service/ec2/ec2iface"
	"github.com/mitchellh/mapstructure"
	"github.com/pkg/errors"
	log "github.com/sirupsen/logrus"
	"gopkg.in/yaml.v3"

	"sigs.k8s.io/zeitgeist/upstreams"
)

// Client holds any client that is needed
type Client struct {
	AWSEC2Client ec2iface.EC2API
}

// Dependencies is used to deserialise the configuration file
type Dependencies struct {
	Dependencies []*Dependency `yaml:"dependencies"`
}

// Dependency is the internal representation of a dependency
type Dependency struct {
	Name string `yaml:"name"`
	// Version of the dependency that should be present throughout your code
	Version string `yaml:"version"`
	// Scheme for versioning this dependency
	Scheme VersionScheme `yaml:"scheme"`
	// Optional: sensitivity, to alert e.g. on new major versions
	Sensitivity VersionSensitivity `yaml:"sensitivity"`
	// Optional: upstream
	Upstream map[string]string `yaml:"upstream"`
	// List of references to this dependency in local files
	RefPaths []*RefPath `yaml:"refPaths"`
}

// RefPath represents a file to check for a reference to the version
type RefPath struct {
	// Path of the file to test
	Path string `yaml:"path"`
	// Match expression for the line that should contain the dependency's version. Regexp is supported.
	Match string `yaml:"match"`
}

// NewClient returns all clients that can be used to the validation
func NewClient() *Client {
	return &Client{
		AWSEC2Client: upstreams.NewAWSClient(),
	}
}

// UnmarshalYAML implements custom unmarshalling of Dependency with validation
func (decoded *Dependency) UnmarshalYAML(unmarshal func(interface{}) error) error {
	// Use a different type to prevent infinite loop in unmarshalling
	type DependencyYAML Dependency

	d := (*DependencyYAML)(decoded)

	if err := unmarshal(&d); err != nil {
		return err
	}

	// Custom validation for the Dependency type
	if d.Name == "" {
		return errors.Errorf("Dependency has no `name`: %v", d)
	}

	if d.Version == "" {
		return errors.Errorf("Dependency has no `version`: %v", d)
	}

	// Default scheme to Semver if unset
	if d.Scheme == "" {
		d.Scheme = Semver
	}

	// Validate Scheme and return
	switch d.Scheme {
	case Semver, Alpha, Random:
		// All good!
	default:
		return errors.Errorf("unknown version scheme: %s", d.Scheme)
	}

	log.Debugf("Deserialised Dependency %v: %v", d.Name, d)

	return nil
}

func fromFile(dependencyFilePath string) (*Dependencies, error) {
	depFile, err := ioutil.ReadFile(dependencyFilePath)
	if err != nil {
		return nil, err
	}

	dependencies := &Dependencies{}

	err = yaml.Unmarshal(depFile, dependencies)
	if err != nil {
		return nil, err
	}

	return dependencies, nil
}

// LocalCheck checks whether dependencies are in-sync locally
//
// Will return an error if the dependency cannot be found in the files it has defined, or if the version does not match
func (c *Client) LocalCheck(dependencyFilePath, basePath string) error {
	log.Debugf("Base path %s", basePath)
	externalDeps, err := fromFile(dependencyFilePath)
	if err != nil {
		return err
	}

	var nonMatchingPaths []string
	for _, dep := range externalDeps.Dependencies {
		log.Debugf("Examining dependency: %v", dep.Name)

		for _, refPath := range dep.RefPaths {
			filePath := filepath.Join(basePath, refPath.Path)

			file, err := os.Open(filePath)
			if err != nil {
				log.Errorf("Error opening %v: %v", filePath, err)
				return err
			}

			log.Debugf("Examining file: %v", filePath)

			match := refPath.Match
			matcher := regexp.MustCompile(match)
			scanner := bufio.NewScanner(file)

			var found bool

			var lineNumber int
			for scanner.Scan() {
				lineNumber++

				line := scanner.Text()
				if matcher.MatchString(line) {
					if strings.Contains(line, dep.Version) {
						log.Debugf(
							"Line %v matches expected regexp '%v' and version '%v': %v",
							lineNumber,
							match,
							dep.Version,
							line,
						)

						found = true
						break
					}
				}
			}

			if !found {
				log.Debugf("Finished reading file %v, no match found.", filePath)

				nonMatchingPaths = append(nonMatchingPaths, refPath.Path)
			}
		}

		if len(nonMatchingPaths) > 0 {
			log.Errorf(
				"%v indicates that %v should be at version %v, but the following files didn't match: %v",
				dependencyFilePath,
				dep.Name,
				dep.Version,
				strings.Join(nonMatchingPaths, ", "),
			)

			return errors.New("Dependencies are not in sync")
		}
	}

	return nil
}

// RemoteCheck checks whether dependencies are up to date with upstream
//
// Will return an error if checking the versions upstream fails.
//
// Out-of-date dependencies will be printed out on stdout at the INFO level.
func (c *Client) RemoteCheck(dependencyFilePath string) ([]string, error) {
	externalDeps, err := fromFile(dependencyFilePath)
	if err != nil {
		return nil, err
	}

	updates := make([]string, 0)

	for _, dep := range externalDeps.Dependencies {
		log.Debugf("Examining dependency: %v", dep.Name)

		if dep.Upstream == nil {
			continue
		}

		upstream := dep.Upstream
		latestVersion := Version{dep.Version, dep.Scheme}
		currentVersion := Version{dep.Version, dep.Scheme}

		var err error

		// Cast the flavour from the currently unknown upstream type
		flavour := upstreams.UpstreamFlavour(upstream["flavour"])
		switch flavour {
		case upstreams.DummyFlavour:
			var d upstreams.Dummy

			decodeErr := mapstructure.Decode(upstream, &d)
			if decodeErr != nil {
				return nil, decodeErr
			}

			latestVersion.Version, err = d.LatestVersion()
		case upstreams.GithubFlavour:
			var gh upstreams.Github

			decodeErr := mapstructure.Decode(upstream, &gh)
			if decodeErr != nil {
				return nil, decodeErr
			}

			latestVersion.Version, err = gh.LatestVersion()
		case upstreams.GitLabFlavour:
			var gl upstreams.GitLab

			decodeErr := mapstructure.Decode(upstream, &gl)
			if decodeErr != nil {
				log.Debug("errr decoding")
				return nil, decodeErr
			}

			latestVersion.Version, err = gl.LatestVersion()
		case upstreams.AMIFlavour:
			var ami upstreams.AMI

			decodeErr := mapstructure.Decode(upstream, &ami)
			if decodeErr != nil {
				return nil, decodeErr
			}

			ami.ServiceClient = c.AWSEC2Client

			latestVersion.Version, err = ami.LatestVersion()
		default:
			return nil, errors.Errorf("unknown upstream flavour '%v' for dependency %v", flavour, dep.Name)
		}

		if err != nil {
			return nil, err
		}

		updateAvailable, err := latestVersion.MoreSensitivelyRecentThan(currentVersion, dep.Sensitivity)
		if err != nil {
			return nil, err
		}

		if updateAvailable {
			updates = append(
				updates,
				fmt.Sprintf(
					"Update available for dependency %v: %v (current: %v)",
					dep.Name,
					latestVersion.Version,
					currentVersion.Version,
				),
			)
		} else {
			log.Debugf(
				"No update available for dependency %v: %v (latest: %v)\n",
				dep.Name,
				currentVersion.Version,
				latestVersion.Version,
			)
		}
	}

	return updates, nil
}
