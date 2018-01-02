package diff

import (
	"bytes"
	"testing"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/filemode"

	. "gopkg.in/check.v1"
)

func Test(t *testing.T) { TestingT(t) }

type UnifiedEncoderTestSuite struct{}

var _ = Suite(&UnifiedEncoderTestSuite{})

func (s *UnifiedEncoderTestSuite) TestBothFilesEmpty(c *C) {
	buffer := bytes.NewBuffer(nil)
	e := NewUnifiedEncoder(buffer, 1)
	err := e.Encode(testPatch{filePatches: []testFilePatch{{}}})
	c.Assert(err, Equals, ErrBothFilesEmpty)
}

func (s *UnifiedEncoderTestSuite) TestBinaryFile(c *C) {
	buffer := bytes.NewBuffer(nil)
	e := NewUnifiedEncoder(buffer, 1)
	p := testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "binary",
				seed: "something",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "binary",
				seed: "otherthing",
			},
		}},
	}

	err := e.Encode(p)
	c.Assert(err, IsNil)

	c.Assert(buffer.String(), Equals, `diff --git a/binary b/binary
index a459bc245bdbc45e1bca99e7fe61731da5c48da4..6879395eacf3cc7e5634064ccb617ac7aa62be7d 100644
Binary files a/binary and b/binary differ
`)
}

func (s *UnifiedEncoderTestSuite) TestEncode(c *C) {
	for _, f := range fixtures {
		c.Log("executing: ", f.desc)

		buffer := bytes.NewBuffer(nil)
		e := NewUnifiedEncoder(buffer, f.context)

		err := e.Encode(f.patch)
		c.Assert(err, IsNil)

		c.Assert(buffer.String(), Equals, f.diff)
	}
}

var oneChunkPatch Patch = testPatch{
	message: "",
	filePatches: []testFilePatch{{
		from: &testFile{
			mode: filemode.Regular,
			path: "onechunk.txt",
			seed: "A\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nÑ\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nX\nY\nZ",
		},
		to: &testFile{
			mode: filemode.Regular,
			path: "onechunk.txt",
			seed: "B\nC\nD\nE\nF\nG\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nV\nW\nX\nY\nZ",
		},

		chunks: []testChunk{{
			content: "A\n",
			op:      Delete,
		}, {
			content: "B\nC\nD\nE\nF\nG",
			op:      Equal,
		}, {
			content: "H\n",
			op:      Delete,
		}, {
			content: "I\nJ\nK\nL\nM\nN\n",
			op:      Equal,
		}, {
			content: "Ñ\n",
			op:      Delete,
		}, {
			content: "O\nP\nQ\nR\nS\nT\n",
			op:      Equal,
		}, {
			content: "U\n",
			op:      Delete,
		}, {
			content: "V\nW\nX\nY\nZ",
			op:      Equal,
		}},
	}},
}

var oneChunkPatchInverted Patch = testPatch{
	message: "",
	filePatches: []testFilePatch{{
		to: &testFile{
			mode: filemode.Regular,
			path: "onechunk.txt",
			seed: "A\nB\nC\nD\nE\nF\nG\nH\nI\nJ\nK\nL\nM\nN\nÑ\nO\nP\nQ\nR\nS\nT\nU\nV\nW\nX\nY\nZ",
		},
		from: &testFile{
			mode: filemode.Regular,
			path: "onechunk.txt",
			seed: "B\nC\nD\nE\nF\nG\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nV\nW\nX\nY\nZ",
		},

		chunks: []testChunk{{
			content: "A\n",
			op:      Add,
		}, {
			content: "B\nC\nD\nE\nF\nG",
			op:      Equal,
		}, {
			content: "H\n",
			op:      Add,
		}, {
			content: "I\nJ\nK\nL\nM\nN\n",
			op:      Equal,
		}, {
			content: "Ñ\n",
			op:      Add,
		}, {
			content: "O\nP\nQ\nR\nS\nT\n",
			op:      Equal,
		}, {
			content: "U\n",
			op:      Add,
		}, {
			content: "V\nW\nX\nY\nZ",
			op:      Equal,
		}},
	}},
}

var fixtures []*fixture = []*fixture{{
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Executable,
				path: "test.txt",
				seed: "test",
			},
			chunks: nil,
		}},
	},
	desc:    "make executable",
	context: 1,
	diff: `diff --git a/test.txt b/test.txt
old mode 100644
new mode 100755
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "test1.txt",
				seed: "test",
			},
			chunks: nil,
		}},
	},
	desc:    "rename file",
	context: 1,
	diff: `diff --git a/test.txt b/test1.txt
rename from test.txt
rename to test1.txt
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "test1.txt",
				seed: "test1",
			},
			chunks: []testChunk{{
				content: "test",
				op:      Delete,
			}, {
				content: "test1",
				op:      Add,
			}},
		}},
	},
	desc:    "rename file with changes",
	context: 1,
	diff: `diff --git a/test.txt b/test1.txt
rename from test.txt
rename to test1.txt
index 30d74d258442c7c65512eafab474568dd706c430..f079749c42ffdcc5f52ed2d3a6f15b09307e975e 100644
--- a/test.txt
+++ b/test1.txt
@@ -1 +1 @@
-test
+test1
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Executable,
				path: "test1.txt",
				seed: "test",
			},
			chunks: nil,
		}},
	},
	desc:    "rename with file mode change",
	context: 1,
	diff: `diff --git a/test.txt b/test1.txt
old mode 100644
new mode 100755
rename from test.txt
rename to test1.txt
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test2",
			},

			chunks: []testChunk{{
				content: "test",
				op:      Delete,
			}, {
				content: "test2",
				op:      Add,
			}},
		}},
	},

	desc:    "one line change",
	context: 1,
	diff: `diff --git a/test.txt b/test.txt
index 30d74d258442c7c65512eafab474568dd706c430..d606037cb232bfda7788a8322492312d55b2ae9d 100644
--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-test
+test2
`,
}, {
	patch: testPatch{
		message: "this is the message\n",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test2",
			},

			chunks: []testChunk{{
				content: "test",
				op:      Delete,
			}, {
				content: "test2",
				op:      Add,
			}},
		}},
	},

	desc:    "one line change with message",
	context: 1,
	diff: `this is the message
diff --git a/test.txt b/test.txt
index 30d74d258442c7c65512eafab474568dd706c430..d606037cb232bfda7788a8322492312d55b2ae9d 100644
--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-test
+test2
`,
}, {
	patch: testPatch{
		message: "this is the message",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "test.txt",
				seed: "test2",
			},

			chunks: []testChunk{{
				content: "test",
				op:      Delete,
			}, {
				content: "test2",
				op:      Add,
			}},
		}},
	},

	desc:    "one line change with message and no end of line",
	context: 1,
	diff: `this is the message
diff --git a/test.txt b/test.txt
index 30d74d258442c7c65512eafab474568dd706c430..d606037cb232bfda7788a8322492312d55b2ae9d 100644
--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-test
+test2
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: nil,
			to: &testFile{
				mode: filemode.Regular,
				path: "new.txt",
				seed: "test\ntest2\test3",
			},

			chunks: []testChunk{{
				content: "test\ntest2\ntest3",
				op:      Add,
			}},
		}},
	},

	desc:    "new file",
	context: 1,
	diff: `diff --git a/new.txt b/new.txt
new file mode 100644
index 0000000000000000000000000000000000000000..65c8dd02a42273038658a22b1cb29c8d9457ca12
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,3 @@
+test
+test2
+test3
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "old.txt",
				seed: "test",
			},
			to: nil,

			chunks: []testChunk{{
				content: "test",
				op:      Delete,
			}},
		}},
	},

	desc:    "delete file",
	context: 1,
	diff: `diff --git a/old.txt b/old.txt
deleted file mode 100644
index 30d74d258442c7c65512eafab474568dd706c430..0000000000000000000000000000000000000000
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-test
`,
}, {
	patch:   oneChunkPatch,
	desc:    "modified deleting lines file with context to 1",
	context: 1,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index ab5eed5d4a2c33aeef67e0188ee79bed666bde6f..0adddcde4fd38042c354518351820eb06c417c82 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,2 +1 @@
-A
 B
@@ -7,3 +6,2 @@ F
 G
-H
 I
@@ -14,3 +12,2 @@ M
 N
-Ñ
 O
@@ -21,3 +18,2 @@ S
 T
-U
 V
`,
}, {
	patch:   oneChunkPatch,
	desc:    "modified deleting lines file with context to 2",
	context: 2,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index ab5eed5d4a2c33aeef67e0188ee79bed666bde6f..0adddcde4fd38042c354518351820eb06c417c82 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,3 +1,2 @@
-A
 B
 C
@@ -6,5 +5,4 @@ E
 F
 G
-H
 I
 J
@@ -13,5 +11,4 @@ L
 M
 N
-Ñ
 O
 P
@@ -20,5 +17,4 @@ R
 S
 T
-U
 V
 W
`,
}, {
	patch: oneChunkPatch,

	desc:    "modified deleting lines file with context to 3",
	context: 3,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index ab5eed5d4a2c33aeef67e0188ee79bed666bde6f..0adddcde4fd38042c354518351820eb06c417c82 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,25 +1,21 @@
-A
 B
 C
 D
 E
 F
 G
-H
 I
 J
 K
 L
 M
 N
-Ñ
 O
 P
 Q
 R
 S
 T
-U
 V
 W
 X
`,
}, {
	patch:   oneChunkPatch,
	desc:    "modified deleting lines file with context to 4",
	context: 4,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index ab5eed5d4a2c33aeef67e0188ee79bed666bde6f..0adddcde4fd38042c354518351820eb06c417c82 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,26 +1,22 @@
-A
 B
 C
 D
 E
 F
 G
-H
 I
 J
 K
 L
 M
 N
-Ñ
 O
 P
 Q
 R
 S
 T
-U
 V
 W
 X
 Y
`,
}, {
	patch:   oneChunkPatch,
	desc:    "modified deleting lines file with context to 0",
	context: 0,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index ab5eed5d4a2c33aeef67e0188ee79bed666bde6f..0adddcde4fd38042c354518351820eb06c417c82 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1 +0,0 @@
-A
@@ -8 +6,0 @@ G
-H
@@ -15 +12,0 @@ N
-Ñ
@@ -22 +18,0 @@ T
-U
`,
}, {
	patch:   oneChunkPatchInverted,
	desc:    "modified adding lines file with context to 1",
	context: 1,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index 0adddcde4fd38042c354518351820eb06c417c82..ab5eed5d4a2c33aeef67e0188ee79bed666bde6f 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1 +1,2 @@
+A
 B
@@ -6,2 +7,3 @@ F
 G
+H
 I
@@ -12,2 +14,3 @@ M
 N
+Ñ
 O
@@ -18,2 +21,3 @@ S
 T
+U
 V
`,
}, {
	patch:   oneChunkPatchInverted,
	desc:    "modified adding lines file with context to 2",
	context: 2,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index 0adddcde4fd38042c354518351820eb06c417c82..ab5eed5d4a2c33aeef67e0188ee79bed666bde6f 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,2 +1,3 @@
+A
 B
 C
@@ -5,4 +6,5 @@ E
 F
 G
+H
 I
 J
@@ -11,4 +13,5 @@ L
 M
 N
+Ñ
 O
 P
@@ -17,4 +20,5 @@ R
 S
 T
+U
 V
 W
`,
}, {
	patch:   oneChunkPatchInverted,
	desc:    "modified adding lines file with context to 3",
	context: 3,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index 0adddcde4fd38042c354518351820eb06c417c82..ab5eed5d4a2c33aeef67e0188ee79bed666bde6f 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,21 +1,25 @@
+A
 B
 C
 D
 E
 F
 G
+H
 I
 J
 K
 L
 M
 N
+Ñ
 O
 P
 Q
 R
 S
 T
+U
 V
 W
 X
`,
}, {
	patch:   oneChunkPatchInverted,
	desc:    "modified adding lines file with context to 4",
	context: 4,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index 0adddcde4fd38042c354518351820eb06c417c82..ab5eed5d4a2c33aeef67e0188ee79bed666bde6f 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -1,22 +1,26 @@
+A
 B
 C
 D
 E
 F
 G
+H
 I
 J
 K
 L
 M
 N
+Ñ
 O
 P
 Q
 R
 S
 T
+U
 V
 W
 X
 Y
`,
}, {
	patch:   oneChunkPatchInverted,
	desc:    "modified adding lines file with context to 0",
	context: 0,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index 0adddcde4fd38042c354518351820eb06c417c82..ab5eed5d4a2c33aeef67e0188ee79bed666bde6f 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -0,0 +1 @@
+A
@@ -6,0 +8 @@ G
+H
@@ -12,0 +15 @@ N
+Ñ
@@ -18,0 +22 @@ T
+U
`,
}, {
	patch: testPatch{
		message: "",
		filePatches: []testFilePatch{{
			from: &testFile{
				mode: filemode.Regular,
				path: "onechunk.txt",
				seed: "B\nC\nD\nE\nF\nG\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nV\nW\nX\nY\nZ",
			},
			to: &testFile{
				mode: filemode.Regular,
				path: "onechunk.txt",
				seed: "B\nC\nD\nE\nF\nG\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nV\nW\nX\nY\n",
			},

			chunks: []testChunk{{
				content: "B\nC\nD\nE\nF\nG\nI\nJ\nK\nL\nM\nN\nO\nP\nQ\nR\nS\nT\nV\nW\nX\nY\n",
				op:      Equal,
			}, {
				content: "Z",
				op:      Delete,
			}},
		}},
	},
	desc:    "remove last letter",
	context: 0,
	diff: `diff --git a/onechunk.txt b/onechunk.txt
index 0adddcde4fd38042c354518351820eb06c417c82..553ae669c7a9303cf848fcc749a2569228ac5309 100644
--- a/onechunk.txt
+++ b/onechunk.txt
@@ -23 +22,0 @@ Y
-Z
`,
}}

type testPatch struct {
	message     string
	filePatches []testFilePatch
}

func (t testPatch) FilePatches() []FilePatch {
	var result []FilePatch
	for _, f := range t.filePatches {
		result = append(result, f)
	}

	return result
}

func (t testPatch) Message() string {
	return t.message
}

type testFilePatch struct {
	from, to *testFile
	chunks   []testChunk
}

func (t testFilePatch) IsBinary() bool {
	return len(t.chunks) == 0
}
func (t testFilePatch) Files() (File, File) {
	// Go is amazing
	switch {
	case t.from == nil && t.to == nil:
		return nil, nil
	case t.from == nil:
		return nil, t.to
	case t.to == nil:
		return t.from, nil
	}

	return t.from, t.to
}

func (t testFilePatch) Chunks() []Chunk {
	var result []Chunk
	for _, c := range t.chunks {
		result = append(result, c)
	}
	return result
}

type testFile struct {
	path string
	mode filemode.FileMode
	seed string
}

func (t testFile) Hash() plumbing.Hash {
	return plumbing.ComputeHash(plumbing.BlobObject, []byte(t.seed))
}

func (t testFile) Mode() filemode.FileMode {
	return t.mode
}

func (t testFile) Path() string {
	return t.path
}

type testChunk struct {
	content string
	op      Operation
}

func (t testChunk) Content() string {
	return t.content
}

func (t testChunk) Type() Operation {
	return t.op
}

type fixture struct {
	desc    string
	context int
	diff    string
	patch   Patch
}
