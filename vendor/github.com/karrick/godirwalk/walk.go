package godirwalk

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Options provide parameters for how the Walk function operates.
type Options struct {
	// ErrorCallback specifies a function to be invoked in the case of an error
	// that could potentially be ignored while walking a file system
	// hierarchy. When set to nil or left as its zero-value, any error condition
	// causes Walk to immediately return the error describing what took
	// place. When non-nil, this user supplied function is invoked with the OS
	// pathname of the file system object that caused the error along with the
	// error that took place. The return value of the supplied ErrorCallback
	// function determines whether the error will cause Walk to halt immediately
	// as it would were no ErrorCallback value provided, or skip this file
	// system node yet continue on with the remaining nodes in the file system
	// hierarchy.
	//
	// ErrorCallback is invoked both for errors that are returned by the
	// runtime, and for errors returned by other user supplied callback
	// functions.
	ErrorCallback func(string, error) ErrorAction

	// FollowSymbolicLinks specifies whether Walk will follow symbolic links
	// that refer to directories. When set to false or left as its zero-value,
	// Walk will still invoke the callback function with symbolic link nodes,
	// but if the symbolic link refers to a directory, it will not recurse on
	// that directory. When set to true, Walk will recurse on symbolic links
	// that refer to a directory.
	FollowSymbolicLinks bool

	// Unsorted controls whether or not Walk will sort the immediate descendants
	// of a directory by their relative names prior to visiting each of those
	// entries.
	//
	// When set to false or left at its zero-value, Walk will get the list of
	// immediate descendants of a particular directory, sort that list by
	// lexical order of their names, and then visit each node in the list in
	// sorted order. This will cause Walk to always traverse the same directory
	// tree in the same order, however may be inefficient for directories with
	// many immediate descendants.
	//
	// When set to true, Walk skips sorting the list of immediate descendants
	// for a directory, and simply visits each node in the order the operating
	// system enumerated them. This will be more fast, but with the side effect
	// that the traversal order may be different from one invocation to the
	// next.
	Unsorted bool

	// Callback is a required function that Walk will invoke for every file
	// system node it encounters.
	Callback WalkFunc

	// PostChildrenCallback is an option function that Walk will invoke for
	// every file system directory it encounters after its children have been
	// processed.
	PostChildrenCallback WalkFunc

	// ScratchBuffer is an optional byte slice to use as a scratch buffer for
	// Walk to use when reading directory entries, to reduce amount of garbage
	// generation. Not all architectures take advantage of the scratch
	// buffer. If omitted or the provided buffer has fewer bytes than
	// MinimumScratchBufferSize, then a buffer with MinimumScratchBufferSize
	// bytes will be created and used once per Walk invocation.
	ScratchBuffer []byte

	// AllowNonDirectory causes Walk to bypass the check that ensures it is
	// being called on a directory node, or when FollowSymbolicLinks is true, a
	// symbolic link that points to a directory. Leave this value false to have
	// Walk return an error when called on a non-directory. Set this true to
	// have Walk run even when called on a non-directory node.
	AllowNonDirectory bool
}

// ErrorAction defines a set of actions the Walk function could take based on
// the occurrence of an error while walking the file system. See the
// documentation for the ErrorCallback field of the Options structure for more
// information.
type ErrorAction int

const (
	// Halt is the ErrorAction return value when the upstream code wants to halt
	// the walk process when a runtime error takes place. It matches the default
	// action the Walk function would take were no ErrorCallback provided.
	Halt ErrorAction = iota

	// SkipNode is the ErrorAction return value when the upstream code wants to
	// ignore the runtime error for the current file system node, skip
	// processing of the node that caused the error, and continue walking the
	// file system hierarchy with the remaining nodes.
	SkipNode
)

// SkipThis is used as a return value from WalkFuncs to indicate that the file
// system entry named in the call is to be skipped. It is not returned as an
// error by any function.
var SkipThis = errors.New("skip this directory entry")

// WalkFunc is the type of the function called for each file system node visited
// by Walk. The pathname argument will contain the argument to Walk as a prefix;
// that is, if Walk is called with "dir", which is a directory containing the
// file "a", the provided WalkFunc will be invoked with the argument "dir/a",
// using the correct os.PathSeparator for the Go Operating System architecture,
// GOOS. The directory entry argument is a pointer to a Dirent for the node,
// providing access to both the basename and the mode type of the file system
// node.
//
// If an error is returned by the Callback or PostChildrenCallback functions,
// and no ErrorCallback function is provided, processing stops. If an
// ErrorCallback function is provided, then it is invoked with the OS pathname
// of the node that caused the error along along with the error. The return
// value of the ErrorCallback function determines whether to halt processing, or
// skip this node and continue processing remaining file system nodes.
//
// The exception is when the function returns the special value
// filepath.SkipDir. If the function returns filepath.SkipDir when invoked on a
// directory, Walk skips the directory's contents entirely. If the function
// returns filepath.SkipDir when invoked on a non-directory file system node,
// Walk skips the remaining files in the containing directory. Note that any
// supplied ErrorCallback function is not invoked with filepath.SkipDir when the
// Callback or PostChildrenCallback functions return that special value.
//
// One arguably confusing aspect of the filepath.WalkFunc API that this library
// must emulate is how a caller tells Walk to skip file system entries or
// directories. With both filepath.Walk and this Walk, when a callback function
// wants to skip a directory and not descend into its children, it returns
// filepath.SkipDir. If the callback function returns filepath.SkipDir for a
// non-directory, filepath.Walk and this library will stop processing any more
// entries in the current directory, which is what many people do not want. If
// you want to simply skip a particular non-directory entry but continue
// processing entries in the directory, a callback function must return nil. The
// implications of this API is when you want to walk a file system hierarchy and
// skip an entry, when the entry is a directory, you must return one value,
// namely filepath.SkipDir, but when the entry is a non-directory, you must
// return a different value, namely nil. In other words, to get identical
// behavior for two file system entry types you need to send different token
// values.
//
// Here is an example callback function that adheres to filepath.Walk API to
// have it skip any file system entry whose full pathname includes a particular
// substring, optSkip:
//
//     func callback1(osPathname string, de *godirwalk.Dirent) error {
//         if optSkip != "" && strings.Contains(osPathname, optSkip) {
//             if b, err := de.IsDirOrSymlinkToDir(); b == true && err == nil {
//                 return filepath.SkipDir
//             }
//             return nil
//         }
//         // Process file like normal...
//         return nil
//     }
//
// This library attempts to eliminate some of that logic boilerplate by
// providing a new token error value, SkipThis, which a callback function may
// return to skip the current file system entry regardless of what type of entry
// it is. If the current entry is a directory, its children will not be
// enumerated, exactly as if the callback returned filepath.SkipDir. If the
// current entry is a non-directory, the next file system entry in the current
// directory will be enumerated, exactly as if the callback returned nil. The
// following example callback function has identical behavior as the previous,
// but has less boilerplate, and admittedly more simple logic.
//
//     func callback2(osPathname string, de *godirwalk.Dirent) error {
//         if optSkip != "" && strings.Contains(osPathname, optSkip) {
//             return godirwalk.SkipThis
//         }
//         // Process file like normal...
//         return nil
//     }
type WalkFunc func(osPathname string, directoryEntry *Dirent) error

// Walk walks the file tree rooted at the specified directory, calling the
// specified callback function for each file system node in the tree, including
// root, symbolic links, and other node types.
//
// This function is often much faster than filepath.Walk because it does not
// invoke os.Stat for every node it encounters, but rather obtains the file
// system node type when it reads the parent directory.
//
// If a runtime error occurs, either from the operating system or from the
// upstream Callback or PostChildrenCallback functions, processing typically
// halts. However, when an ErrorCallback function is provided in the provided
// Options structure, that function is invoked with the error along with the OS
// pathname of the file system node that caused the error. The ErrorCallback
// function's return value determines the action that Walk will then take.
//
//    func main() {
//        dirname := "."
//        if len(os.Args) > 1 {
//            dirname = os.Args[1]
//        }
//        err := godirwalk.Walk(dirname, &godirwalk.Options{
//            Callback: func(osPathname string, de *godirwalk.Dirent) error {
//                fmt.Printf("%s %s\n", de.ModeType(), osPathname)
//                return nil
//            },
//            ErrorCallback: func(osPathname string, err error) godirwalk.ErrorAction {
//            	// Your program may want to log the error somehow.
//            	fmt.Fprintf(os.Stderr, "ERROR: %s\n", err)
//
//            	// For the purposes of this example, a simple SkipNode will suffice,
//            	// although in reality perhaps additional logic might be called for.
//            	return godirwalk.SkipNode
//            },
//        })
//        if err != nil {
//            fmt.Fprintf(os.Stderr, "%s\n", err)
//            os.Exit(1)
//        }
//    }
func Walk(pathname string, options *Options) error {
	if options == nil || options.Callback == nil {
		return errors.New("cannot walk without non-nil options and Callback function")
	}

	pathname = filepath.Clean(pathname)

	var fi os.FileInfo
	var err error

	if options.FollowSymbolicLinks {
		fi, err = os.Stat(pathname)
	} else {
		fi, err = os.Lstat(pathname)
	}
	if err != nil {
		return err
	}

	mode := fi.Mode()
	if !options.AllowNonDirectory && mode&os.ModeDir == 0 {
		return fmt.Errorf("cannot Walk non-directory: %s", pathname)
	}

	dirent := &Dirent{
		name:     filepath.Base(pathname),
		path:     filepath.Dir(pathname),
		modeType: mode & os.ModeType,
	}

	if len(options.ScratchBuffer) < MinimumScratchBufferSize {
		options.ScratchBuffer = newScratchBuffer()
	}

	// If ErrorCallback is nil, set to a default value that halts the walk
	// process on all operating system errors. This is done to allow error
	// handling to be more succinct in the walk code.
	if options.ErrorCallback == nil {
		options.ErrorCallback = defaultErrorCallback
	}

	err = walk(pathname, dirent, options)
	switch err {
	case nil, SkipThis, filepath.SkipDir:
		// silence SkipThis and filepath.SkipDir for top level
		debug("no error of significance: %v\n", err)
		return nil
	default:
		return err
	}
}

// defaultErrorCallback always returns Halt because if the upstream code did not
// provide an ErrorCallback function, walking the file system hierarchy ought to
// halt upon any operating system error.
func defaultErrorCallback(_ string, _ error) ErrorAction { return Halt }

// walk recursively traverses the file system node specified by pathname and the
// Dirent.
func walk(osPathname string, dirent *Dirent, options *Options) error {
	err := options.Callback(osPathname, dirent)
	if err != nil {
		if err == SkipThis || err == filepath.SkipDir {
			return err
		}
		if action := options.ErrorCallback(osPathname, err); action == SkipNode {
			return nil
		}
		return err
	}

	if dirent.IsSymlink() {
		if !options.FollowSymbolicLinks {
			return nil
		}
		// Does this symlink point to a directory?
		info, err := os.Stat(osPathname)
		if err != nil {
			if action := options.ErrorCallback(osPathname, err); action == SkipNode {
				return nil
			}
			return err
		}
		if !info.IsDir() {
			return nil
		}
	} else if !dirent.IsDir() {
		return nil
	}

	// If get here, then specified pathname refers to a directory or a
	// symbolic link to a directory.

	var ds scanner

	if options.Unsorted {
		// When upstream does not request a sorted iteration, it's more memory
		// efficient to read a single child at a time from the file system.
		ds, err = NewScanner(osPathname)
	} else {
		// When upstream wants a sorted iteration, we must read the entire
		// directory and sort through the child names, and then iterate on each
		// child.
		ds, err = newSortedScanner(osPathname, options.ScratchBuffer)
	}
	if err != nil {
		if action := options.ErrorCallback(osPathname, err); action == SkipNode {
			return nil
		}
		return err
	}

	for ds.Scan() {
		deChild, err := ds.Dirent()
		osChildname := filepath.Join(osPathname, deChild.name)
		if err != nil {
			if action := options.ErrorCallback(osChildname, err); action == SkipNode {
				return nil
			}
			return err
		}
		err = walk(osChildname, deChild, options)
		debug("osChildname: %q; error: %v\n", osChildname, err)
		if err == nil || err == SkipThis {
			continue
		}
		if err != filepath.SkipDir {
			return err
		}
		// When received SkipDir on a directory or a symbolic link to a
		// directory, stop processing that directory but continue processing
		// siblings.  When received on a non-directory, stop processing
		// remaining siblings.
		isDir, err := deChild.IsDirOrSymlinkToDir()
		if err != nil {
			if action := options.ErrorCallback(osChildname, err); action == SkipNode {
				continue // ignore and continue with next sibling
			}
			return err // caller does not approve of this error
		}
		if !isDir {
			break // stop processing remaining siblings, but allow post children callback
		}
		// continue processing remaining siblings
	}
	if err = ds.Err(); err != nil {
		return err
	}

	if options.PostChildrenCallback == nil {
		return nil
	}

	err = options.PostChildrenCallback(osPathname, dirent)
	if err == nil || err == filepath.SkipDir {
		return err
	}

	if action := options.ErrorCallback(osPathname, err); action == SkipNode {
		return nil
	}
	return err
}
