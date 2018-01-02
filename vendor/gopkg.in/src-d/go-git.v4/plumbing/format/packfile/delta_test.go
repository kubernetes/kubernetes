package packfile

import (
	"math/rand"

	. "gopkg.in/check.v1"
)

type DeltaSuite struct {
	testCases []deltaTest
}

var _ = Suite(&DeltaSuite{})

type deltaTest struct {
	description string
	base        []piece
	target      []piece
}

func (s *DeltaSuite) SetUpSuite(c *C) {
	s.testCases = []deltaTest{{
		description: "distinct file",
		base:        []piece{{"0", 300}},
		target:      []piece{{"2", 200}},
	}, {
		description: "same file",
		base:        []piece{{"1", 3000}},
		target:      []piece{{"1", 3000}},
	}, {
		description: "small file",
		base:        []piece{{"1", 3}},
		target:      []piece{{"1", 3}, {"0", 1}},
	}, {
		description: "big file",
		base:        []piece{{"1", 300000}},
		target:      []piece{{"1", 30000}, {"0", 1000000}},
	}, {
		description: "add elements before",
		base:        []piece{{"0", 200}},
		target:      []piece{{"1", 300}, {"0", 200}},
	}, {
		description: "add 10 times more elements at the end",
		base:        []piece{{"1", 300}, {"0", 200}},
		target:      []piece{{"0", 2000}},
	}, {
		description: "add elements between",
		base:        []piece{{"0", 400}},
		target:      []piece{{"0", 200}, {"1", 200}, {"0", 200}},
	}, {
		description: "add elements after",
		base:        []piece{{"0", 200}},
		target:      []piece{{"0", 200}, {"1", 200}},
	}, {
		description: "modify elements at the end",
		base:        []piece{{"1", 300}, {"0", 200}},
		target:      []piece{{"0", 100}},
	}, {
		description: "complex modification",
		base: []piece{{"0", 3}, {"1", 40}, {"2", 30}, {"3", 2},
			{"4", 400}, {"5", 23}},
		target: []piece{{"1", 30}, {"2", 20}, {"7", 40}, {"4", 400},
			{"5", 10}},
	}, {
		description: "A copy operation bigger tan 64kb",
		base:        []piece{{bigRandStr, 1}, {"1", 200}},
		target:      []piece{{bigRandStr, 1}},
	}}
}

var bigRandStr = randStringBytes(100 * 1024)

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return string(b)
}

func (s *DeltaSuite) TestAddDelta(c *C) {
	for _, t := range s.testCases {
		baseBuf := genBytes(t.base)
		targetBuf := genBytes(t.target)
		delta := DiffDelta(baseBuf, targetBuf)
		result, err := PatchDelta(baseBuf, delta)

		c.Log("Executing test case:", t.description)
		c.Assert(err, IsNil)
		c.Assert(result, DeepEquals, targetBuf)
	}
}

func (s *DeltaSuite) TestIncompleteDelta(c *C) {
	for _, t := range s.testCases {
		c.Log("Incomplete delta on:", t.description)
		baseBuf := genBytes(t.base)
		targetBuf := genBytes(t.target)
		delta := DiffDelta(baseBuf, targetBuf)
		delta = delta[:len(delta)-2]
		result, err := PatchDelta(baseBuf, delta)
		c.Assert(err, NotNil)
		c.Assert(result, IsNil)
	}

	// check nil input too
	result, err := PatchDelta(nil, nil)
	c.Assert(err, NotNil)
	c.Assert(result, IsNil)
}
