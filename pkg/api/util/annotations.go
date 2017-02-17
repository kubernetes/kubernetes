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

package util

import (
	"fmt"
	"regexp"
)

const (
	metadataAnnotationPattern string = "^metadata.annotations\\['((.)+)'\\]$"
)

type AnnotationFieldPathParser struct {
	annotationRegexp *regexp.Regexp
}

func NewAnnotationFieldPathParser() (*AnnotationFieldPathParser, error) {
	annotationRegexp, err := regexp.Compile(metadataAnnotationPattern)
	if err != nil {
		return nil, err
	}

	return &AnnotationFieldPathParser{annotationRegexp: annotationRegexp}, nil
}

func (p *AnnotationFieldPathParser) Validate(s string) bool {
	if !p.annotationRegexp.MatchString(s) {
		return false
	}

	key, err := p.ParseKey(s)
	if err != nil {
		return false
	}

	var previous rune
	for _, c := range key {
		if (c == '\'' || c == '[' || c == ']') && previous != '\\' {
			return false
		}
		previous = c
	}

	return true
}

func (p *AnnotationFieldPathParser) ParseKey(s string) (string, error) {
	submatch := p.annotationRegexp.FindStringSubmatch(s)

	if len(submatch) < 2 {
		return "", fmt.Errorf("Couldn't extract annotation label")
	}

	return submatch[1], nil
}
