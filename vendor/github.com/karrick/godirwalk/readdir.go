package godirwalk

// ReadDirents returns a sortable slice of pointers to Dirent structures, each
// representing the file system name and mode type for one of the immediate
// descendant of the specified directory. If the specified directory is a
// symbolic link, it will be resolved.
//
// If an optional scratch buffer is provided that is at least one page of
// memory, it will be used when reading directory entries from the file
// system. If you plan on calling this function in a loop, you will have
// significantly better performance if you allocate a scratch buffer and use it
// each time you call this function.
//
//    children, err := godirwalk.ReadDirents(osDirname, nil)
//    if err != nil {
//        return nil, errors.Wrap(err, "cannot get list of directory children")
//    }
//    sort.Sort(children)
//    for _, child := range children {
//        fmt.Printf("%s %s\n", child.ModeType, child.Name)
//    }
func ReadDirents(osDirname string, scratchBuffer []byte) (Dirents, error) {
	return readDirents(osDirname, scratchBuffer)
}

// ReadDirnames returns a slice of strings, representing the immediate
// descendants of the specified directory. If the specified directory is a
// symbolic link, it will be resolved.
//
// If an optional scratch buffer is provided that is at least one page of
// memory, it will be used when reading directory entries from the file
// system. If you plan on calling this function in a loop, you will have
// significantly better performance if you allocate a scratch buffer and use it
// each time you call this function.
//
// Note that this function, depending on operating system, may or may not invoke
// the ReadDirents function, in order to prepare the list of immediate
// descendants. Therefore, if your program needs both the names and the file
// system mode types of descendants, it will always be faster to invoke
// ReadDirents directly, rather than calling this function, then looping over
// the results and calling os.Stat or os.LStat for each entry.
//
//    children, err := godirwalk.ReadDirnames(osDirname, nil)
//    if err != nil {
//        return nil, errors.Wrap(err, "cannot get list of directory children")
//    }
//    sort.Strings(children)
//    for _, child := range children {
//        fmt.Printf("%s\n", child)
//    }
func ReadDirnames(osDirname string, scratchBuffer []byte) ([]string, error) {
	return readDirnames(osDirname, scratchBuffer)
}
