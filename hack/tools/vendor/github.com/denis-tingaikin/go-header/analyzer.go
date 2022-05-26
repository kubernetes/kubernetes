// Copyright (c) 2020-2022 Denis Tingaikin
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package goheader

import (
	"fmt"
	"go/ast"
	"os"
	"os/exec"
	"strings"
	"time"
)

type Target struct {
	Path string
	File *ast.File
}

const iso = "2006-01-02 15:04:05 -0700"

func (t *Target) ModTime() (time.Time, error) {
	diff, err := exec.Command("git", "diff", t.Path).CombinedOutput()
	if err == nil && len(diff) == 0 {
		line, err := exec.Command("git", "log", "-1", "--pretty=format:%cd", "--date=iso", "--", t.Path).CombinedOutput()
		if err == nil {
			return time.Parse(iso, string(line))
		}
	}
	info, err := os.Stat(t.Path)
	if err != nil {
		return time.Time{}, err
	}
	return info.ModTime(), nil
}

type Analyzer struct {
	values   map[string]Value
	template string
}

func (a *Analyzer) Analyze(target *Target) Issue {
	if a.template == "" {
		return NewIssue("Missed template for check")
	}
	if t, err := target.ModTime(); err == nil {
		if t.Year() != time.Now().Year() {
			return nil
		}
	}
	file := target.File
	var header string
	var offset = Location{
		Position: 1,
	}
	if len(file.Comments) > 0 && file.Comments[0].Pos() < file.Package {
		if strings.HasPrefix(file.Comments[0].List[0].Text, "/*") {
			header = (&ast.CommentGroup{List: []*ast.Comment{file.Comments[0].List[0]}}).Text()
		} else {
			header = file.Comments[0].Text()
			offset.Position += 3
		}
	}
	header = strings.TrimSpace(header)
	if header == "" {
		return NewIssue("Missed header for check")
	}
	s := NewReader(header)
	s.SetOffset(offset)
	t := NewReader(a.template)
	for !s.Done() && !t.Done() {
		templateCh := t.Peek()
		if templateCh == '{' {
			name := a.readField(t)
			if a.values[name] == nil {
				return NewIssue(fmt.Sprintf("Template has unknown value: %v", name))
			}
			if i := a.values[name].Read(s); i != nil {
				return i
			}
			continue
		}
		sourceCh := s.Peek()
		if sourceCh != templateCh {
			l := s.Location()
			notNextLine := func(r rune) bool {
				return r != '\n'
			}
			actual := s.ReadWhile(notNextLine)
			expected := t.ReadWhile(notNextLine)
			return NewIssueWithLocation(fmt.Sprintf("Actual: %v\nExpected:%v", actual, expected), l)
		}
		s.Next()
		t.Next()
	}
	if !s.Done() {
		l := s.Location()
		return NewIssueWithLocation(fmt.Sprintf("Unexpected string: %v", s.Finish()), l)
	}
	if !t.Done() {
		l := s.Location()
		return NewIssueWithLocation(fmt.Sprintf("Missed string: %v", t.Finish()), l)
	}
	return nil
}

func (a *Analyzer) readField(reader *Reader) string {
	_ = reader.Next()
	_ = reader.Next()

	r := reader.ReadWhile(func(r rune) bool {
		return r != '}'
	})

	_ = reader.Next()
	_ = reader.Next()

	return strings.ToLower(strings.TrimSpace(r))
}

func New(options ...Option) *Analyzer {
	a := &Analyzer{}
	for _, o := range options {
		o.apply(a)
	}
	for _, v := range a.values {
		err := v.Calculate(a.values)
		if err != nil {
			panic(err.Error())
		}
	}
	return a
}
