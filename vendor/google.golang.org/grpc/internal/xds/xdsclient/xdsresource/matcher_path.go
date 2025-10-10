/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package xdsresource

import (
	"regexp"
	"strings"

	"google.golang.org/grpc/internal/grpcutil"
)

type pathMatcher interface {
	match(path string) bool
	String() string
}

type pathExactMatcher struct {
	// fullPath is all upper case if caseInsensitive is true.
	fullPath        string
	caseInsensitive bool
}

func newPathExactMatcher(p string, caseInsensitive bool) *pathExactMatcher {
	ret := &pathExactMatcher{
		fullPath:        p,
		caseInsensitive: caseInsensitive,
	}
	if caseInsensitive {
		ret.fullPath = strings.ToUpper(p)
	}
	return ret
}

func (pem *pathExactMatcher) match(path string) bool {
	if pem.caseInsensitive {
		return pem.fullPath == strings.ToUpper(path)
	}
	return pem.fullPath == path
}

func (pem *pathExactMatcher) String() string {
	return "pathExact:" + pem.fullPath
}

type pathPrefixMatcher struct {
	// prefix is all upper case if caseInsensitive is true.
	prefix          string
	caseInsensitive bool
}

func newPathPrefixMatcher(p string, caseInsensitive bool) *pathPrefixMatcher {
	ret := &pathPrefixMatcher{
		prefix:          p,
		caseInsensitive: caseInsensitive,
	}
	if caseInsensitive {
		ret.prefix = strings.ToUpper(p)
	}
	return ret
}

func (ppm *pathPrefixMatcher) match(path string) bool {
	if ppm.caseInsensitive {
		return strings.HasPrefix(strings.ToUpper(path), ppm.prefix)
	}
	return strings.HasPrefix(path, ppm.prefix)
}

func (ppm *pathPrefixMatcher) String() string {
	return "pathPrefix:" + ppm.prefix
}

type pathRegexMatcher struct {
	re *regexp.Regexp
}

func newPathRegexMatcher(re *regexp.Regexp) *pathRegexMatcher {
	return &pathRegexMatcher{re: re}
}

func (prm *pathRegexMatcher) match(path string) bool {
	return grpcutil.FullMatchWithRegex(prm.re, path)
}

func (prm *pathRegexMatcher) String() string {
	return "pathRegex:" + prm.re.String()
}
