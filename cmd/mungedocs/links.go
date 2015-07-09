/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"fmt"
	"net/url"
	"os"
	"path"
	"regexp"
	"strings"
)

var (
	// Finds markdown links of the form [foo](bar "alt-text").
	linkRE = regexp.MustCompile(`\[([^]]*)\]\(([^)]*)\)`)
	// Splits the link target into link target and alt-text.
	altTextRE = regexp.MustCompile(`(.*)( ".*")`)
)

// checkLinks assumes fileBytes has links in markdown syntax, and verifies that
// any relative links actually point to files that exist.
func checkLinks(filePath string, fileBytes []byte) ([]byte, error) {
	dir := path.Dir(filePath)
	errors := []string{}

	output := linkRE.ReplaceAllFunc(fileBytes, func(in []byte) (out []byte) {
		match := linkRE.FindSubmatch(in)
		// match[0] is the entire expression; [1] is the visible text and [2] is the link text.
		visibleText := string(match[1])
		linkText := string(match[2])
		altText := ""
		if parts := altTextRE.FindStringSubmatch(linkText); parts != nil {
			linkText = parts[1]
			altText = parts[2]
		}

		// clean up some random garbage I found in our docs.
		linkText = strings.Trim(linkText, " ")
		linkText = strings.Trim(linkText, "\n")
		linkText = strings.Trim(linkText, " ")

		u, err := url.Parse(linkText)
		if err != nil {
			errors = append(
				errors,
				fmt.Sprintf("%v, link %q is unparsable: %v", filePath, linkText, err),
			)
			return in
		}

		if u.Host != "" {
			// We only care about relative links.
			return in
		}

		suggestedVisibleText := visibleText
		if u.Path != "" && !strings.HasPrefix(linkText, "TODO:") {
			newPath, targetExists := checkPath(filePath, path.Clean(u.Path))
			if !targetExists {
				errors = append(
					errors,
					fmt.Sprintf("%v, %q: target not found\n", filePath, linkText),
				)
			}
			u.Path = newPath
			// Make the visible text show the absolute path if it's
			// not nested in or beneath the current directory.
			if strings.HasPrefix(u.Path, "..") {
				suggestedVisibleText = makeRepoRelative(path.Join(dir, u.Path))
			} else {
				suggestedVisibleText = u.Path
			}
			if unescaped, err := url.QueryUnescape(u.String()); err != nil {
				// Remove %28 type stuff, be nice to humans.
				// And don't fight with the toc generator.
				linkText = unescaped
			} else {
				linkText = u.String()
			}
		}
		// If the current visible text is trying to be a file name, use
		// the correct file name.
		if (strings.Contains(visibleText, ".md") || strings.Contains(visibleText, "/")) && !strings.ContainsAny(visibleText, ` '"`+"`") {
			visibleText = suggestedVisibleText
		}

		return []byte(fmt.Sprintf("[%s](%s)", visibleText, linkText+altText))
	})
	err := error(nil)
	if len(errors) != 0 {
		err = fmt.Errorf("%s", strings.Join(errors, "\n"))
	}
	return output, err
}

func makeRepoRelative(path string) string {
	parts := strings.Split(path, "github.com/GoogleCloudPlatform/kubernetes/")
	if len(parts) > 1 {
		// Take out anything that is specific to the local filesystem.
		return parts[1]
	}
	return path
}

func checkPath(filePath, linkPath string) (newPath string, ok bool) {
	dir := path.Dir(filePath)
	if strings.HasPrefix(linkPath, "/") {
		if !strings.HasPrefix(linkPath, "/GoogleCloudPlatform") {
			// Any absolute paths that aren't relative to github.com are wrong.
			// Try to fix.
			linkPath = linkPath[1:]
		}
	}

	newPath = linkPath
	for i := 0; i < 5; i++ {
		// The file must exist.
		target := path.Join(dir, newPath)
		if info, err := os.Stat(target); err == nil {
			if info.IsDir() {
				return newPath + "/", true
			}
			return newPath, true
		}
		newPath = path.Join("..", newPath)
	}
	return linkPath, false
}
