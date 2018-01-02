/*
Package fsnoder allows to create merkletrie noders that resemble file
systems, from human readable string descriptions.  Its intended use is
generating noders in tests in a readable way.

For example:

	root, _ = New("(a<1> b<2>, B(c<3> d()))")

will create a noder as follows:

       root        - "root" is an unnamed dir containing "a", "b" and "B".
       / | \       - "a" is a file containing the string "1".
      /  |  \      - "b" is a file containing the string "2".
     a   b   B     - "B" is a directory containing "c" and "d".
            / \    - "c" is a file containing the string "3".
           c   d   - "D" is an empty directory.

Files are expressed as:

- one or more letters and dots for the name of the file

- a single number, between angle brackets, for the contents of the file.

- examples: a<1>, foo.go<2>.

Directories are expressed as:

- one or more letters for the name of the directory.

- its elements between parents, separated with spaces, in any order.

- (optionally) the root directory can be unnamed, by skiping its name.

Examples:

- D(a<1> b<2>) : two files, "a" and "b", having "1" and "2" as their
respective contents, inside a directory called "D".

- A() : An empty directory called "A".

- A(b<>) : An directory called "A", with an empty file inside called "b":

- (b(c<1> d(e<2>)) f<>) : an unamed directory containing:

    ├── b              --> directory
    │   ├── c          --> file containing "1"
    │   └── d          --> directory
    │       └── e      --> file containing "2"
    └── f              --> empty file
*/
package fsnoder
