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

package mem

import "sort"

type DirMap map[string]*FileData

func (m DirMap) Len() int           { return len(m) }
func (m DirMap) Add(f *FileData)    { m[f.name] = f }
func (m DirMap) Remove(f *FileData) { delete(m, f.name) }
func (m DirMap) Files() (files []*FileData) {
	for _, f := range m {
		files = append(files, f)
	}
	sort.Sort(filesSorter(files))
	return files
}

// implement sort.Interface for []*FileData
type filesSorter []*FileData

func (s filesSorter) Len() int           { return len(s) }
func (s filesSorter) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s filesSorter) Less(i, j int) bool { return s[i].name < s[j].name }

func (m DirMap) Names() (names []string) {
	for x := range m {
		names = append(names, x)
	}
	return names
}
