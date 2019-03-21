/*
Package godirwalk provides functions to read and traverse directory trees.

In short, why do I use this library?

* It's faster than `filepath.Walk`.

* It's more correct on Windows than `filepath.Walk`.

* It's more easy to use than `filepath.Walk`.

* It's more flexible than `filepath.Walk`.

USAGE

This library will normalize the provided top level directory name based on the
os-specific path separator by calling `filepath.Clean` on its first
argument. However it always provides the pathname created by using the correct
os-specific path separator when invoking the provided callback function.

    dirname := "some/directory/root"
    err := godirwalk.Walk(dirname, &godirwalk.Options{
        Callback: func(osPathname string, de *godirwalk.Dirent) error {
            fmt.Printf("%s %s\n", de.ModeType(), osPathname)
            return nil
        },
    })

This library not only provides functions for traversing a file system directory
tree, but also for obtaining a list of immediate descendants of a particular
directory, typically much more quickly than using `os.ReadDir` or
`os.ReadDirnames`.
*/
package godirwalk
