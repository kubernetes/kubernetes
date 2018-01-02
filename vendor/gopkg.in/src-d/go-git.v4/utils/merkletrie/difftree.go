package merkletrie

// The focus of this difftree implementation is to save time by
// skipping whole directories if their hash is the same in both
// trees.
//
// The diff algorithm implemented here is based on the doubleiter
// type defined in this same package; we will iterate over both
// trees at the same time, while comparing the current noders in
// each iterator.  Depending on how they differ we will output the
// corresponding chages and move the iterators further over both
// trees.
//
// The table bellow show all the possible comparison results, along
// with what changes should we produce and how to advance the
// iterators.
//
// The table is implemented by the switches in this function,
// diffTwoNodes, diffTwoNodesSameName and diffTwoDirs.
//
// Many Bothans died to bring us this information, make sure you
// understand the table before modifying this code.

// # Cases
//
// When comparing noders in both trees you will found yourself in
// one of 169 possible cases, but if we ignore moves, we can
// simplify a lot the search space into the following table:
//
// - "-": nothing, no file or directory
// - a<>: an empty file named "a".
// - a<1>: a file named "a", with "1" as its contents.
// - a<2>: a file named "a", with "2" as its contents.
// - a(): an empty dir named "a".
// - a(...): a dir named "a", with some files and/or dirs inside (possibly
//   empty).
// - a(;;;): a dir named "a", with some other files and/or dirs inside
//   (possibly empty), which different from the ones in "a(...)".
//
//     \ to     -   a<>  a<1>  a<2>  a()  a(...)  a(;;;)
// from \
// -           00    01    02    03   04     05      06
// a<>         10    11    12    13   14     15      16
// a<1>        20    21    22    23   24     25      26
// a<2>        30    31    32    33   34     35      36
// a()         40    41    42    43   44     45      46
// a(...)      50    51    52    53   54     55      56
// a(;;;)      60    61    62    63   64     65      66
//
// Every (from, to) combination in the table is a special case, but
// some of them can be merged into some more general cases, for
// instance 11 and 22 can be merged into the general case: both
// noders are equal.
//
// Here is a full list of all the cases that are similar and how to
// merge them together into more general cases.  Each general case
// is labeled wiht an uppercase letter for further reference, and it
// is followed by the pseudocode of the checks you have to perfrom
// on both noders to see if you are in such a case, the actions to
// perform (i.e. what changes to output) and how to advance the
// iterators of each tree to continue the comparison process.
//
// ## A. Impossible: 00
//
// ## B. Same thing on both sides: 11, 22, 33, 44, 55, 66
//   - check: `SameName() && SameHash()`
//   - action: do nothing.
//   - advance: `FromNext(); ToNext()`
//
// ### C. To was created: 01, 02, 03, 04, 05, 06
//   - check: `DifferentName() && ToBeforeFrom()`
//   - action: inserRecursively(to)
//   - advance: `ToNext()`
//
// ### D. From was deleted: 10, 20, 30, 40, 50, 60
//   - check: `DifferentName() && FromBeforeTo()`
//   - action: `DeleteRecursively(from)`
//   - advance: `FromNext()`
//
// ### E. Empty file to file with contents: 12, 13
//   - check: `SameName() && DifferentHash() && FromIsFile() &&
//             ToIsFile() && FromIsEmpty()`
//   - action: `modifyFile(from, to)`
//   - advance: `FromNext()` or `FromStep()`
//
// ### E'. file with contents to empty file: 21, 31
//   - check: `SameName() && DifferentHash() && FromIsFile() &&
//             ToIsFile() && ToIsEmpty()`
//   - action: `modifyFile(from, to)`
//   - advance: `FromNext()` or `FromStep()`
//
// ### F. empty file to empty dir with the same name: 14
//   - check: `SameName() && FromIsFile() && FromIsEmpty() &&
//             ToIsDir() && ToIsEmpty()`
//   - action: `DeleteFile(from); InsertEmptyDir(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### F'. empty dir to empty file of the same name: 41
//   - check: `SameName() && FromIsDir() && FromIsEmpty &&
//             ToIsFile() && ToIsEmpty()`
//   - action: `DeleteEmptyDir(from); InsertFile(to)`
//   - advance: `FromNext(); ToNext()` or step for any of them.
//
// ### G. empty file to non-empty dir of the same name: 15, 16
//   - check: `SameName() && FromIsFile() && ToIsDir() &&
//             FromIsEmpty() && ToIsNotEmpty()`
//   - action: `DeleteFile(from); InsertDirRecursively(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### G'. non-empty dir to empty file of the same name: 51, 61
//   - check: `SameName() && FromIsDir() && FromIsNotEmpty() &&
//             ToIsFile() && FromIsEmpty()`
//   - action: `DeleteDirRecursively(from); InsertFile(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### H. modify file contents: 23, 32
//   - check: `SameName() && FromIsFile() && ToIsFile() &&
//             FromIsNotEmpty() && ToIsNotEmpty()`
//   - action: `ModifyFile(from, to)`
//   - advance: `FromNext(); ToNext()`
//
// ### I. file with contents to empty dir: 24, 34
//   - check: `SameName() && DifferentHash() && FromIsFile() &&
//             FromIsNotEmpty() && ToIsDir() && ToIsEmpty()`
//   - action: `DeleteFile(from); InsertEmptyDir(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### I'. empty dir to file with contents: 42, 43
//   - check: `SameName() && DifferentHash() && FromIsDir() &&
//             FromIsEmpty() && ToIsFile() && ToIsEmpty()`
//   - action: `DeleteDir(from); InsertFile(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### J. file with contents to dir with contetns: 25, 26, 35, 36
//   - check: `SameName() && DifferentHash() && FromIsFile() &&
//             FromIsNotEmpty() && ToIsDir() && ToIsNotEmpty()`
//   - action: `DeleteFile(from); InsertDirRecursively(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### J'. dir with contetns to file with contents: 52, 62, 53, 63
//   - check: `SameName() && DifferentHash() && FromIsDir() &&
//             FromIsNotEmpty() && ToIsFile() && ToIsNotEmpty()`
//   - action: `DeleteDirRecursively(from); InsertFile(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### K. empty dir to dir with contents: 45, 46
//   - check: `SameName() && DifferentHash() && FromIsDir() &&
//             FromIsEmpty() && ToIsDir() && ToIsNotEmpty()`
//   - action: `InsertChildrenRecursively(to)`
//   - advance: `FromNext(); ToNext()`
//
// ### K'. dir with contents to empty dir: 54, 64
//   - check: `SameName() && DifferentHash() && FromIsDir() &&
//             FromIsEmpty() && ToIsDir() && ToIsNotEmpty()`
//   - action: `DeleteChildrenRecursively(from)`
//   - advance: `FromNext(); ToNext()`
//
// ### L. dir with contents to dir with different contents: 56, 65
//   - check: `SameName() && DifferentHash() && FromIsDir() &&
//             FromIsNotEmpty() && ToIsDir() && ToIsNotEmpty()`
//   - action: nothing
//   - advance: `FromStep(); ToStep()`
//
//

// All these cases can be further simplified by a truth table
// reduction process, in which we gather similar checks together to
// make the final code easier to read and understand.
//
// The first 6 columns are the outputs of the checks to perform on
// both noders.  I have labeled them 1 to 6, this is what they mean:
//
// 1: SameName()
// 2: SameHash()
// 3: FromIsDir()
// 4: ToIsDir()
// 5: FromIsEmpty()
// 6: ToIsEmpty()
//
// The from and to columns are a fsnoder example of the elements
// that you will find on each tree under the specified comparison
// results (columns 1 to 6).
//
// The type column identifies the case we are into, from the list above.
//
// The type' column identifies the new set of reduced cases, using
// lowercase letters, and they are explained after the table.
//
// The last column is the set of actions and advances for each case.
//
// "---" means impossible except in case of hash collision.
//
// advance meaning:
// - NN: from.Next(); to.Next()
// - SS: from.Step(); to.Step()
//
// 1 2 3 4 5 6 | from   |  to    |type|type'|action ; advance
// ------------+--------+--------+----+------------------------------------
// 0 0 0 0 0 0 |        |        |    |     | if !SameName() {
//     .       |        |        |    |     |    if FromBeforeTo() {
//     .       |        |        | D  |  d  |       delete(from); from.Next()
//     .       |        |        |    |     |    } else {
//     .       |        |        | C  |  c  |       insert(to); to.Next()
//     .       |        |        |    |     |    }
// 0 1 1 1 1 1 |        |        |    |     | }
// 1 0 0 0 0 0 |  a<1>  |  a<2>  | H  |  e  | modify(from, to); NN
// 1 0 0 0 0 1 |  a<1>  |   a<>  | E' |  e  | modify(from, to); NN
// 1 0 0 0 1 0 |   a<>  |  a<1>  | E  |  e  | modify(from, to); NN
// 1 0 0 0 1 1 |  ----  |  ----  |    |  e  |
// 1 0 0 1 0 0 |  a<1>  | a(...) | J  |  f  | delete(from); insert(to); NN
// 1 0 0 1 0 1 |  a<1>  |    a() | I  |  f  | delete(from); insert(to); NN
// 1 0 0 1 1 0 |   a<>  | a(...) | G  |  f  | delete(from); insert(to); NN
// 1 0 0 1 1 1 |   a<>  |    a() | F  |  f  | delete(from); insert(to); NN
// 1 0 1 0 0 0 | a(...) |  a<1>  | J' |  f  | delete(from); insert(to); NN
// 1 0 1 0 0 1 | a(...) |   a<>  | G' |  f  | delete(from); insert(to); NN
// 1 0 1 0 1 0 |    a() |  a<1>  | I' |  f  | delete(from); insert(to); NN
// 1 0 1 0 1 1 |    a() |   a<>  | F' |  f  | delete(from); insert(to); NN
// 1 0 1 1 0 0 | a(...) | a(;;;) | L  |  g  | nothing; SS
// 1 0 1 1 0 1 | a(...) |    a() | K' |  h  | deleteChidren(from); NN
// 1 0 1 1 1 0 |    a() | a(...) | K  |  i  | insertChildren(to); NN
// 1 0 1 1 1 1 |  ----  |  ----  |    |     |
// 1 1 0 0 0 0 |  a<1>  |  a<1>  | B  |  b  | nothing; NN
// 1 1 0 0 0 1 |  ----  |  ----  |    |  b  |
// 1 1 0 0 1 0 |  ----  |  ----  |    |  b  |
// 1 1 0 0 1 1 |   a<>  |   a<>  | B  |  b  | nothing; NN
// 1 1 0 1 0 0 |  ----  |  ----  |    |  b  |
// 1 1 0 1 0 1 |  ----  |  ----  |    |  b  |
// 1 1 0 1 1 0 |  ----  |  ----  |    |  b  |
// 1 1 0 1 1 1 |  ----  |  ----  |    |  b  |
// 1 1 1 0 0 0 |  ----  |  ----  |    |  b  |
// 1 1 1 0 0 1 |  ----  |  ----  |    |  b  |
// 1 1 1 0 1 0 |  ----  |  ----  |    |  b  |
// 1 1 1 0 1 1 |  ----  |  ----  |    |  b  |
// 1 1 1 1 0 0 | a(...) | a(...) | B  |  b  | nothing; NN
// 1 1 1 1 0 1 |  ----  |  ----  |    |  b  |
// 1 1 1 1 1 0 |  ----  |  ----  |    |  b  |
// 1 1 1 1 1 1 |   a()  |   a()  | B  |  b  | nothing; NN
//
// c and d:
//     if !SameName()
//         d if FromBeforeTo()
//         c else
// b: SameName) && sameHash()
// e: SameName() && !sameHash() && BothAreFiles()
// f: SameName() && !sameHash() && FileAndDir()
// g: SameName() && !sameHash() && BothAreDirs() && NoneIsEmpty
// i: SameName() && !sameHash() && BothAreDirs() && FromIsEmpty
// h: else of i

import (
	"fmt"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"
)

// DiffTree calculates the list of changes between two merkletries.  It
// uses the provided hashEqual callback to compare noders.
func DiffTree(fromTree, toTree noder.Noder,
	hashEqual noder.Equal) (Changes, error) {
	ret := NewChanges()

	ii, err := newDoubleIter(fromTree, toTree, hashEqual)
	if err != nil {
		return nil, err
	}

	for {
		from := ii.from.current
		to := ii.to.current

		switch r := ii.remaining(); r {
		case noMoreNoders:
			return ret, nil
		case onlyFromRemains:
			if err = ret.AddRecursiveDelete(from); err != nil {
				return nil, err
			}
			if err = ii.nextFrom(); err != nil {
				return nil, err
			}
		case onlyToRemains:
			if err = ret.AddRecursiveInsert(to); err != nil {
				return nil, err
			}
			if err = ii.nextTo(); err != nil {
				return nil, err
			}
		case bothHaveNodes:
			if err = diffNodes(&ret, ii); err != nil {
				return nil, err
			}
		default:
			panic(fmt.Sprintf("unknown remaining value: %d", r))
		}
	}
}

func diffNodes(changes *Changes, ii *doubleIter) error {
	from := ii.from.current
	to := ii.to.current
	var err error

	// compare their full paths as strings
	switch from.Compare(to) {
	case -1:
		if err = changes.AddRecursiveDelete(from); err != nil {
			return err
		}
		if err = ii.nextFrom(); err != nil {
			return err
		}
	case 1:
		if err = changes.AddRecursiveInsert(to); err != nil {
			return err
		}
		if err = ii.nextTo(); err != nil {
			return err
		}
	default:
		if err := diffNodesSameName(changes, ii); err != nil {
			return err
		}
	}

	return nil
}

func diffNodesSameName(changes *Changes, ii *doubleIter) error {
	from := ii.from.current
	to := ii.to.current

	status, err := ii.compare()
	if err != nil {
		return err
	}

	switch {
	case status.sameHash:
		// do nothing
		if err = ii.nextBoth(); err != nil {
			return err
		}
	case status.bothAreFiles:
		changes.Add(NewModify(from, to))
		if err = ii.nextBoth(); err != nil {
			return err
		}
	case status.fileAndDir:
		if err = changes.AddRecursiveDelete(from); err != nil {
			return err
		}
		if err = changes.AddRecursiveInsert(to); err != nil {
			return err
		}
		if err = ii.nextBoth(); err != nil {
			return err
		}
	case status.bothAreDirs:
		if err = diffDirs(changes, ii); err != nil {
			return err
		}
	default:
		return fmt.Errorf("bad status from double iterator")
	}

	return nil
}

func diffDirs(changes *Changes, ii *doubleIter) error {
	from := ii.from.current
	to := ii.to.current

	status, err := ii.compare()
	if err != nil {
		return err
	}

	switch {
	case status.fromIsEmptyDir:
		if err = changes.AddRecursiveInsert(to); err != nil {
			return err
		}
		if err = ii.nextBoth(); err != nil {
			return err
		}
	case status.toIsEmptyDir:
		if err = changes.AddRecursiveDelete(from); err != nil {
			return err
		}
		if err = ii.nextBoth(); err != nil {
			return err
		}
	case !status.fromIsEmptyDir && !status.toIsEmptyDir:
		// do nothing
		if err = ii.stepBoth(); err != nil {
			return err
		}
	default:
		return fmt.Errorf("both dirs are empty but has different hash")
	}

	return nil
}
