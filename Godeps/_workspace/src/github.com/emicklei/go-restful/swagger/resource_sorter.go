package swagger

// Copyright 2014 Ernest Micklei. All rights reserved.
// Use of this source code is governed by a license
// that can be found in the LICENSE file.

type ResourceSorter []Resource

func (s ResourceSorter) Len() int {
	return len(s)
}

func (s ResourceSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s ResourceSorter) Less(i, j int) bool {
	return s[i].Path < s[j].Path
}
