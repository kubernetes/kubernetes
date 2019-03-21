package rice

import "os"

// SortByName allows an array of os.FileInfo objects
// to be easily sorted by filename using sort.Sort(SortByName(array))
type SortByName []os.FileInfo

func (f SortByName) Len() int           { return len(f) }
func (f SortByName) Less(i, j int) bool { return f[i].Name() < f[j].Name() }
func (f SortByName) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }

// SortByModified allows an array of os.FileInfo objects
// to be easily sorted by modified date using sort.Sort(SortByModified(array))
type SortByModified []os.FileInfo

func (f SortByModified) Len() int           { return len(f) }
func (f SortByModified) Less(i, j int) bool { return f[i].ModTime().Unix() > f[j].ModTime().Unix() }
func (f SortByModified) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }
