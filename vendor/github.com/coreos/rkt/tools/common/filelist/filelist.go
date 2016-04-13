// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package filelist

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/hashicorp/errwrap"
)

// Lists is a structure holding relative paths to files, symlinks and
// directories. The members of this structure can be later combined
// with common.MapFilesToDirectories to get some meaningful paths.
type Lists struct {
	Files    []string
	Symlinks []string
	Dirs     []string
}

type pair struct {
	kind string
	data *[]string
}

// ParseFilelist parses a given filelist. The filelist format is
// rather simple:
// <BLOCK>
// <BLOCK>
// ...
//
// Where "<BLOCK>" is as follows:
// <HEADER>
// <LIST>
//
// Where "<HEADER>" is as follows:
// <KIND>
// (<COUNT>)
//
// Where "<KIND>" is either "files", "symlinks" or "dirs" and
// "<COUNT>" tells how many items are in the following list. "<LIST>"
// is as follows:
// <1st ITEM>
// <2nd ITEM>
// ...
// <COUNTth ITEM>
// <EMPTY LINE>
func (list *Lists) ParseFilelist(filelist io.Reader) error {
	scanner := bufio.NewScanner(filelist)
	for {
		kind, count, err := parseHeader(scanner)
		if err != nil {
			return errwrap.Wrap(errors.New("failed to parse filelist"), err)
		}
		if kind == "" {
			break
		}
		data := list.getDataForKind(kind)
		if data == nil {
			return fmt.Errorf("failed to parse filelist: unknown kind %q, expected 'files', 'symlinks' or 'dirs'", kind)
		}
		items, err := parseList(scanner, count)
		if err != nil {
			return errwrap.Wrap(errors.New("failed to parse filelist"), err)
		}
		*data = items
	}
	return nil
}

// parseList parses the list part of a block. It makes sure that there
// is an exactly expected count of items.
func parseList(scanner *bufio.Scanner, count int) ([]string, error) {
	got := 0
	items := make([]string, 0, count)
	for {
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return nil, err
			}
			return nil, fmt.Errorf("expected either an empty line or a line with an item, unexpected EOF?")
		}
		line := scanner.Text()
		if line == "" {
			if got < count {
				return nil, fmt.Errorf("too few items (declared %d, got %d)", count, got)
			}
			break
		}
		got++
		if got > count {
			return nil, fmt.Errorf("too many items (declared %d)", count)
		}
		items = append(items, line)
	}
	return items, nil
}

// parseHeader parses the first two lines of a block described in
// ParseFilelist docs. So it returns a data kind (files, symlinks or
// dirs) and a count of elements in the following list. If the
// returned kind is empty then it means that there is no more entries
// (provided that there is no error either).
func parseHeader(scanner *bufio.Scanner) (string, int, error) {
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return "", 0, err
		}
		// no more entries in the file, just return empty kind
		return "", 0, nil
	}
	kind := scanner.Text()
	if kind == "" {
		return "", 0, fmt.Errorf("got an empty kind, expected 'files', 'symlinks' or 'dirs'")
	}
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return "", 0, err
		} else {
			return "", 0, fmt.Errorf("expected a line with a count, unexpected EOF?")
		}
	}
	countReader := strings.NewReader(scanner.Text())
	count := 0
	n, err := fmt.Fscanf(countReader, "(%d)", &count)
	if err != nil {
		return "", 0, err
	}
	if n != 1 {
		return "", 0, fmt.Errorf("incorrectly formatted line with number of %s", kind)
	}
	return kind, count, nil
}

func (list *Lists) getDataForKind(kind string) *[]string {
	switch kind {
	case "files":
		return &list.Files
	case "symlinks":
		return &list.Symlinks
	case "dirs":
		return &list.Dirs
	}
	return nil
}

// GenerateFilelist generates a filelist, duh. And writes it to a
// given writer. The format of generated file is described in
// filelist.ParseFilelist.
func (list *Lists) GenerateFilelist(out io.Writer) error {
	w := bufio.NewWriter(out)
	for _, pair := range list.getPairs() {
		dLen := len(*pair.data)
		toWrite := []string{
			pair.kind,
			"\n(",
			strconv.Itoa(dLen),
			")\n",
		}
		if dLen > 0 {
			toWrite = append(toWrite,
				strings.Join(*pair.data, "\n"),
				"\n")
		}
		toWrite = append(toWrite, "\n")
		for _, str := range toWrite {
			if _, err := w.WriteString(str); err != nil {
				return err
			}
		}
	}
	w.Flush()
	return nil
}

func (list *Lists) getPairs() []pair {
	return []pair{
		{kind: "files", data: &list.Files},
		{kind: "symlinks", data: &list.Symlinks},
		{kind: "dirs", data: &list.Dirs},
	}
}
