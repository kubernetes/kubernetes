/*
Copyright 2017 The Kubernetes Authors.

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

package env

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

var argumentEnvironment = regexp.MustCompile("(?ms)^(.+)\\=(.*)$")
var validArgumentEnvironment = regexp.MustCompile("(?ms)^(\\w+)\\=(.*)$")

// IsEnvironmentArgument checks whether a string is an environment argument, that is, whether it matches the "anycharacters=anycharacters" pattern.
func IsEnvironmentArgument(s string) bool {
	return argumentEnvironment.MatchString(s)
}

// IsValidEnvironmentArgument checks whether a string is a valid environment argument, that is, whether it matches the "wordcharacters=anycharacters" pattern. Word characters can be letters, numbers, and underscores.
func IsValidEnvironmentArgument(s string) bool {
	return validArgumentEnvironment.MatchString(s)
}

// SplitEnvironmentFromResources separates resources from environment arguments.
// Resources must come first. Arguments may have the "DASH-" syntax.
func SplitEnvironmentFromResources(args []string) (resources, envArgs []string, ok bool) {
	first := true
	for _, s := range args {
		// this method also has to understand env removal syntax, i.e. KEY-
		isEnv := IsEnvironmentArgument(s) || strings.HasSuffix(s, "-")
		switch {
		case first && isEnv:
			first = false
			fallthrough
		case !first && isEnv:
			envArgs = append(envArgs, s)
		case first && !isEnv:
			resources = append(resources, s)
		case !first && !isEnv:
			return nil, nil, false
		}
	}
	return resources, envArgs, true
}

// parseIntoEnvVar parses the list of key-value pairs into kubernetes EnvVar.
// envVarType is for making errors more specific to user intentions.
func parseIntoEnvVar(spec []string, defaultReader io.Reader, envVarType string) ([]v1.EnvVar, []string, error) {
	env := []v1.EnvVar{}
	exists := sets.NewString()
	var remove []string
	for _, envSpec := range spec {
		switch {
		case !IsValidEnvironmentArgument(envSpec) && !strings.HasSuffix(envSpec, "-"):
			return nil, nil, fmt.Errorf("%ss must be of the form key=value and can only contain letters, numbers, and underscores", envVarType)
		case envSpec == "-":
			if defaultReader == nil {
				return nil, nil, fmt.Errorf("when '-' is used, STDIN must be open")
			}
			fileEnv, err := readEnv(defaultReader, envVarType)
			if err != nil {
				return nil, nil, err
			}
			env = append(env, fileEnv...)
		case strings.Contains(envSpec, "="):
			parts := strings.SplitN(envSpec, "=", 2)
			if len(parts) != 2 {
				return nil, nil, fmt.Errorf("invalid %s: %v", envVarType, envSpec)
			}
			exists.Insert(parts[0])
			env = append(env, v1.EnvVar{
				Name:  parts[0],
				Value: parts[1],
			})
		case strings.HasSuffix(envSpec, "-"):
			remove = append(remove, envSpec[:len(envSpec)-1])
		default:
			return nil, nil, fmt.Errorf("unknown %s: %v", envVarType, envSpec)
		}
	}
	for _, removeLabel := range remove {
		if _, found := exists[removeLabel]; found {
			return nil, nil, fmt.Errorf("can not both modify and remove the same %s in the same command", envVarType)
		}
	}
	return env, remove, nil
}

// ParseEnv parses the elements of the first argument looking for environment variables in key=value form and, if one of those values is "-", it also scans the reader.
// The same environment variable cannot be both modified and removed in the same command.
func ParseEnv(spec []string, defaultReader io.Reader) ([]v1.EnvVar, []string, error) {
	return parseIntoEnvVar(spec, defaultReader, "environment variable")
}

func readEnv(r io.Reader, envVarType string) ([]v1.EnvVar, error) {
	env := []v1.EnvVar{}
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		envSpec := scanner.Text()
		if pos := strings.Index(envSpec, "#"); pos != -1 {
			envSpec = envSpec[:pos]
		}

		if strings.Contains(envSpec, "=") {
			parts := strings.SplitN(envSpec, "=", 2)
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid %s: %v", envVarType, envSpec)
			}
			env = append(env, v1.EnvVar{
				Name:  parts[0],
				Value: parts[1],
			})
		}
	}

	if err := scanner.Err(); err != nil && err != io.EOF {
		return nil, err
	}

	return env, nil
}
