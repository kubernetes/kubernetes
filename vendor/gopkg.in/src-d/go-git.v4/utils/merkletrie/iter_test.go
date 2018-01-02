package merkletrie_test

import (
	"fmt"
	"io"
	"strings"

	"gopkg.in/src-d/go-git.v4/utils/merkletrie"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/internal/fsnoder"
	"gopkg.in/src-d/go-git.v4/utils/merkletrie/noder"

	. "gopkg.in/check.v1"
)

type IterSuite struct{}

var _ = Suite(&IterSuite{})

// A test is a list of operations we want to perform on an iterator and
// their expected results.
//
// The operations are expressed as a sequence of `n` and `s`,
// representing the amount of next and step operations we want to call
// on the iterator and their order.  For example, an operations value of
// "nns" means: call a `n`ext, then another `n`ext and finish with a
// `s`tep.
//
// The expected is the full path of the noders returned by the
// operations, separated by spaces.
//
// For instance:
//
//     t := test{
//         operations: "ns",
//         expected:   "a a/b"
//     }
//
// means:
//
// - the first iterator operation has to be Next, and it must return a
// node called "a" with no ancestors.
//
// - the second operation has to be Step, and it must return a node
// called "b" with a single ancestor called "a".
type test struct {
	operations string
	expected   string
}

// Runs a test on the provided iterator, checking that the names of the
// returned values are correct.  If not, the treeDescription value is
// printed along with information about mismatch.
func (t test) run(c *C, iter *merkletrie.Iter,
	treeDescription string, testNumber int) {

	expectedChunks := strings.Split(t.expected, " ")
	if t.expected == "" {
		expectedChunks = []string{}
	}

	if len(t.operations) < len(expectedChunks) {
		c.Fatalf("malformed test %d: not enough operations", testNumber)
		return
	}

	var obtained noder.Path
	var err error
	for i, b := range t.operations {
		comment := Commentf("\ntree: %q\ntest #%d (%q)\noperation #%d (%q)",
			treeDescription, testNumber, t.operations, i, t.operations[i])

		switch t.operations[i] {
		case 'n':
			obtained, err = iter.Next()
			if err != io.EOF {
				c.Assert(err, IsNil)
			}
		case 's':
			obtained, err = iter.Step()
			if err != io.EOF {
				c.Assert(err, IsNil)
			}
		default:
			c.Fatalf("unknown operation at test %d, operation %d (%c)\n",
				testNumber, i, b)
		}
		if i >= len(expectedChunks) {
			c.Assert(err, Equals, io.EOF, comment)
			continue
		}

		c.Assert(err, IsNil, comment)
		c.Assert(obtained.String(), Equals, expectedChunks[i], comment)
	}
}

// A testsCollection value represents a tree and a collection of tests
// we want to perfrom on iterators of that tree.
//
// Example:
//
//            .
//            |
//        ---------
//        |   |   |
//        a   b   c
//            |
//            z
//
//    var foo testCollection = {
// 	      tree: "(a<> b(z<>) c<>)"
//     	  tests: []test{
//            {operations: "nns", expected: "a b b/z"},
//            {operations: "nnn", expected: "a b c"},
//		  },
//    }
//
// A new iterator will be build for each test.
type testsCollection struct {
	tree  string // a fsnoder description of a tree.
	tests []test // the collection of tests we want to run
}

// Executes all the tests in a testsCollection.
func (tc testsCollection) run(c *C) {
	root, err := fsnoder.New(tc.tree)
	c.Assert(err, IsNil)

	for i, t := range tc.tests {
		iter, err := merkletrie.NewIter(root)
		c.Assert(err, IsNil)
		t.run(c, iter, root.String(), i)
	}
}

func (s *IterSuite) TestEmptyNamedDir(c *C) {
	tc := testsCollection{
		tree: "A()",
		tests: []test{
			{operations: "n", expected: ""},
			{operations: "nn", expected: ""},
			{operations: "nnn", expected: ""},
			{operations: "nnns", expected: ""},
			{operations: "nnnssnsnns", expected: ""},
			{operations: "s", expected: ""},
			{operations: "ss", expected: ""},
			{operations: "sss", expected: ""},
			{operations: "sssn", expected: ""},
			{operations: "sssnnsnssn", expected: ""},
		},
	}
	tc.run(c)
}

func (s *IterSuite) TestEmptyUnnamedDir(c *C) {
	tc := testsCollection{
		tree: "()",
		tests: []test{
			{operations: "n", expected: ""},
			{operations: "nn", expected: ""},
			{operations: "nnn", expected: ""},
			{operations: "nnns", expected: ""},
			{operations: "nnnssnsnns", expected: ""},
			{operations: "s", expected: ""},
			{operations: "ss", expected: ""},
			{operations: "sss", expected: ""},
			{operations: "sssn", expected: ""},
			{operations: "sssnnsnssn", expected: ""},
		},
	}
	tc.run(c)
}

func (s *IterSuite) TestOneFile(c *C) {
	tc := testsCollection{
		tree: "(a<>)",
		tests: []test{
			{operations: "n", expected: "a"},
			{operations: "nn", expected: "a"},
			{operations: "nnn", expected: "a"},
			{operations: "nnns", expected: "a"},
			{operations: "nnnssnsnns", expected: "a"},
			{operations: "s", expected: "a"},
			{operations: "ss", expected: "a"},
			{operations: "sss", expected: "a"},
			{operations: "sssn", expected: "a"},
			{operations: "sssnnsnssn", expected: "a"},
		},
	}
	tc.run(c)
}

//     root
//      / \
//     a   b
func (s *IterSuite) TestTwoFiles(c *C) {
	tc := testsCollection{
		tree: "(a<> b<>)",
		tests: []test{
			{operations: "nnn", expected: "a b"},
			{operations: "nns", expected: "a b"},
			{operations: "nsn", expected: "a b"},
			{operations: "nss", expected: "a b"},
			{operations: "snn", expected: "a b"},
			{operations: "sns", expected: "a b"},
			{operations: "ssn", expected: "a b"},
			{operations: "sss", expected: "a b"},
		},
	}
	tc.run(c)
}

//     root
//      |
//      a
//      |
//      b
func (s *IterSuite) TestDirWithFile(c *C) {
	tc := testsCollection{
		tree: "(a(b<>))",
		tests: []test{
			{operations: "nnn", expected: "a"},
			{operations: "nns", expected: "a"},
			{operations: "nsn", expected: "a a/b"},
			{operations: "nss", expected: "a a/b"},
			{operations: "snn", expected: "a"},
			{operations: "sns", expected: "a"},
			{operations: "ssn", expected: "a a/b"},
			{operations: "sss", expected: "a a/b"},
		},
	}
	tc.run(c)
}

//     root
//      /|\
//     c a b
func (s *IterSuite) TestThreeSiblings(c *C) {
	tc := testsCollection{
		tree: "(c<> a<> b<>)",
		tests: []test{
			{operations: "nnnn", expected: "a b c"},
			{operations: "nnns", expected: "a b c"},
			{operations: "nnsn", expected: "a b c"},
			{operations: "nnss", expected: "a b c"},
			{operations: "nsnn", expected: "a b c"},
			{operations: "nsns", expected: "a b c"},
			{operations: "nssn", expected: "a b c"},
			{operations: "nsss", expected: "a b c"},
			{operations: "snnn", expected: "a b c"},
			{operations: "snns", expected: "a b c"},
			{operations: "snsn", expected: "a b c"},
			{operations: "snss", expected: "a b c"},
			{operations: "ssnn", expected: "a b c"},
			{operations: "ssns", expected: "a b c"},
			{operations: "sssn", expected: "a b c"},
			{operations: "ssss", expected: "a b c"},
		},
	}
	tc.run(c)
}

//     root
//      |
//      b
//      |
//      c
//      |
//      a
func (s *IterSuite) TestThreeVertical(c *C) {
	tc := testsCollection{
		tree: "(b(c(a())))",
		tests: []test{
			{operations: "nnnn", expected: "b"},
			{operations: "nnns", expected: "b"},
			{operations: "nnsn", expected: "b"},
			{operations: "nnss", expected: "b"},
			{operations: "nsnn", expected: "b b/c"},
			{operations: "nsns", expected: "b b/c"},
			{operations: "nssn", expected: "b b/c b/c/a"},
			{operations: "nsss", expected: "b b/c b/c/a"},
			{operations: "snnn", expected: "b"},
			{operations: "snns", expected: "b"},
			{operations: "snsn", expected: "b"},
			{operations: "snss", expected: "b"},
			{operations: "ssnn", expected: "b b/c"},
			{operations: "ssns", expected: "b b/c"},
			{operations: "sssn", expected: "b b/c b/c/a"},
			{operations: "ssss", expected: "b b/c b/c/a"},
		},
	}
	tc.run(c)
}

//     root
//      / \
//     c   a
//     |
//     b
func (s *IterSuite) TestThreeMix1(c *C) {
	tc := testsCollection{
		tree: "(c(b<>) a<>)",
		tests: []test{
			{operations: "nnnn", expected: "a c"},
			{operations: "nnns", expected: "a c"},
			{operations: "nnsn", expected: "a c c/b"},
			{operations: "nnss", expected: "a c c/b"},
			{operations: "nsnn", expected: "a c"},
			{operations: "nsns", expected: "a c"},
			{operations: "nssn", expected: "a c c/b"},
			{operations: "nsss", expected: "a c c/b"},
			{operations: "snnn", expected: "a c"},
			{operations: "snns", expected: "a c"},
			{operations: "snsn", expected: "a c c/b"},
			{operations: "snss", expected: "a c c/b"},
			{operations: "ssnn", expected: "a c"},
			{operations: "ssns", expected: "a c"},
			{operations: "sssn", expected: "a c c/b"},
			{operations: "ssss", expected: "a c c/b"},
		},
	}
	tc.run(c)
}

//     root
//      / \
//     b   a
//         |
//         c
func (s *IterSuite) TestThreeMix2(c *C) {
	tc := testsCollection{
		tree: "(b() a(c<>))",
		tests: []test{
			{operations: "nnnn", expected: "a b"},
			{operations: "nnns", expected: "a b"},
			{operations: "nnsn", expected: "a b"},
			{operations: "nnss", expected: "a b"},
			{operations: "nsnn", expected: "a a/c b"},
			{operations: "nsns", expected: "a a/c b"},
			{operations: "nssn", expected: "a a/c b"},
			{operations: "nsss", expected: "a a/c b"},
			{operations: "snnn", expected: "a b"},
			{operations: "snns", expected: "a b"},
			{operations: "snsn", expected: "a b"},
			{operations: "snss", expected: "a b"},
			{operations: "ssnn", expected: "a a/c b"},
			{operations: "ssns", expected: "a a/c b"},
			{operations: "sssn", expected: "a a/c b"},
			{operations: "ssss", expected: "a a/c b"},
		},
	}
	tc.run(c)
}

//      root
//      / | \
//     /  |  ----
//    f   d      h --------
//   /\         /  \      |
//  e   a      j   b/      g
//  |  / \     |
//  l  n  k    icm
//     |
//     o
//     |
//     p/
func (s *IterSuite) TestCrazy(c *C) {
	tc := testsCollection{
		tree: "(f(e(l<>) a(n(o(p())) k<>)) d<> h(j(i<> c<> m<>) b() g<>))",
		tests: []test{
			{operations: "nnnnn", expected: "d f h"},
			{operations: "nnnns", expected: "d f h"},
			{operations: "nnnsn", expected: "d f h h/b h/g"},
			{operations: "nnnss", expected: "d f h h/b h/g"},
			{operations: "nnsnn", expected: "d f f/a f/e h"},
			{operations: "nnsns", expected: "d f f/a f/e f/e/l"},
			{operations: "nnssn", expected: "d f f/a f/a/k f/a/n"},
			{operations: "nnsss", expected: "d f f/a f/a/k f/a/n"},
			{operations: "nsnnn", expected: "d f h"},
			{operations: "nsnns", expected: "d f h"},
			{operations: "nsnsn", expected: "d f h h/b h/g"},
			{operations: "nsnss", expected: "d f h h/b h/g"},
			{operations: "nssnn", expected: "d f f/a f/e h"},
		},
	}
	tc.run(c)
}

//      .
//      |
//      a
//      |
//      b
//     / \
//    z   h
//   / \
//  d   e
//      |
//      f
func (s *IterSuite) TestNewIterFromPath(c *C) {
	tree, err := fsnoder.New("(a(b(z(d<> e(f<>)) h<>)))")
	c.Assert(err, IsNil)

	z := find(c, tree, "z")

	iter, err := merkletrie.NewIterFromPath(z)
	c.Assert(err, IsNil)

	n, err := iter.Next()
	c.Assert(err, IsNil)
	c.Assert(n.String(), Equals, "a/b/z/d")

	n, err = iter.Next()
	c.Assert(err, IsNil)
	c.Assert(n.String(), Equals, "a/b/z/e")

	n, err = iter.Step()
	c.Assert(err, IsNil)
	c.Assert(n.String(), Equals, "a/b/z/e/f")

	_, err = iter.Step()
	c.Assert(err, Equals, io.EOF)
}

func find(c *C, tree noder.Noder, name string) noder.Path {
	iter, err := merkletrie.NewIter(tree)
	c.Assert(err, IsNil)

	for {
		current, err := iter.Step()
		if err != io.EOF {
			c.Assert(err, IsNil)
		} else {
			c.Fatalf("node %s not found in tree %s", name, tree)
		}

		if current.Name() == name {
			return current
		}
	}
}

type errorNoder struct{ noder.Noder }

func (e *errorNoder) Children() ([]noder.Noder, error) {
	return nil, fmt.Errorf("mock error")
}

func (s *IterSuite) TestNewIterNil(c *C) {
	i, err := merkletrie.NewIter(nil)
	c.Assert(err, IsNil)
	_, err = i.Next()
	c.Assert(err, Equals, io.EOF)
}

func (s *IterSuite) TestNewIterFailsOnChildrenErrors(c *C) {
	_, err := merkletrie.NewIter(&errorNoder{})
	c.Assert(err, ErrorMatches, "mock error")
}
