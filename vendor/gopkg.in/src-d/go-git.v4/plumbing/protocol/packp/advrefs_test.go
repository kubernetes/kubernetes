package packp

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"

	. "gopkg.in/check.v1"
)

type AdvRefSuite struct{}

var _ = Suite(&AdvRefSuite{})

func (s *AdvRefSuite) TestAddReferenceSymbolic(c *C) {
	ref := plumbing.NewSymbolicReference("foo", "bar")

	a := NewAdvRefs()
	err := a.AddReference(ref)
	c.Assert(err, IsNil)

	values := a.Capabilities.Get(capability.SymRef)
	c.Assert(values, HasLen, 1)
	c.Assert(values[0], Equals, "foo:bar")
}

func (s *AdvRefSuite) TestAddReferenceHash(c *C) {
	ref := plumbing.NewHashReference("foo", plumbing.NewHash("5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c"))

	a := NewAdvRefs()
	err := a.AddReference(ref)
	c.Assert(err, IsNil)

	c.Assert(a.References, HasLen, 1)
	c.Assert(a.References["foo"].String(), Equals, "5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c")
}

func (s *AdvRefSuite) TestAllReferences(c *C) {
	hash := plumbing.NewHash("5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c")

	a := NewAdvRefs()
	err := a.AddReference(plumbing.NewSymbolicReference("foo", "bar"))
	c.Assert(err, IsNil)
	err = a.AddReference(plumbing.NewHashReference("bar", hash))
	c.Assert(err, IsNil)

	refs, err := a.AllReferences()
	c.Assert(err, IsNil)

	iter, err := refs.IterReferences()
	c.Assert(err, IsNil)

	var count int
	iter.ForEach(func(ref *plumbing.Reference) error {
		count++
		switch ref.Name() {
		case "bar":
			c.Assert(ref.Hash(), Equals, hash)
		case "foo":
			c.Assert(ref.Target().String(), Equals, "bar")
		}
		return nil
	})

	c.Assert(count, Equals, 2)
}

func (s *AdvRefSuite) TestAllReferencesBadSymref(c *C) {
	a := NewAdvRefs()
	err := a.Capabilities.Set(capability.SymRef, "foo")
	c.Assert(err, IsNil)

	_, err = a.AllReferences()
	c.Assert(err, NotNil)
}

type AdvRefsDecodeEncodeSuite struct{}

var _ = Suite(&AdvRefsDecodeEncodeSuite{})

func (s *AdvRefsDecodeEncodeSuite) test(c *C, in []string, exp []string) {
	var err error
	var input io.Reader
	{
		var buf bytes.Buffer
		p := pktline.NewEncoder(&buf)
		err = p.EncodeString(in...)
		c.Assert(err, IsNil)
		input = &buf
	}

	var expected []byte
	{
		var buf bytes.Buffer
		p := pktline.NewEncoder(&buf)
		err = p.EncodeString(exp...)
		c.Assert(err, IsNil)

		expected = buf.Bytes()
	}

	var obtained []byte
	{
		ar := NewAdvRefs()
		c.Assert(ar.Decode(input), IsNil)

		var buf bytes.Buffer
		c.Assert(ar.Encode(&buf), IsNil)

		obtained = buf.Bytes()
	}

	c.Assert(string(obtained), DeepEquals, string(expected))
}

func (s *AdvRefsDecodeEncodeSuite) TestNoHead(c *C) {
	input := []string{
		"0000000000000000000000000000000000000000 capabilities^{}\x00",
		pktline.FlushString,
	}

	expected := []string{
		"0000000000000000000000000000000000000000 capabilities^{}\x00\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestNoHeadSmart(c *C) {
	input := []string{
		"# service=git-upload-pack\n",
		"0000000000000000000000000000000000000000 capabilities^{}\x00",
		pktline.FlushString,
	}

	expected := []string{
		"# service=git-upload-pack\n",
		"0000000000000000000000000000000000000000 capabilities^{}\x00\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestNoHeadSmartBug(c *C) {
	input := []string{
		"# service=git-upload-pack\n",
		pktline.FlushString,
		"0000000000000000000000000000000000000000 capabilities^{}\x00\n",
		pktline.FlushString,
	}

	expected := []string{
		"# service=git-upload-pack\n",
		pktline.FlushString,
		"0000000000000000000000000000000000000000 capabilities^{}\x00\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestRefs(c *C) {
	input := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree",
		pktline.FlushString,
	}

	expected := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestPeeled(c *C) {
	input := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		pktline.FlushString,
	}

	expected := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestAll(c *C) {
	input := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}",
		"shallow 1111111111111111111111111111111111111111",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}

	expected := []string{
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestAllSmart(c *C) {
	input := []string{
		"# service=git-upload-pack\n",
		pktline.FlushString,
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}

	expected := []string{
		"# service=git-upload-pack\n",
		pktline.FlushString,
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func (s *AdvRefsDecodeEncodeSuite) TestAllSmartBug(c *C) {
	input := []string{
		"# service=git-upload-pack\n",
		pktline.FlushString,
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}

	expected := []string{
		"# service=git-upload-pack\n",
		pktline.FlushString,
		"6ecf0ef2c2dffb796033e5a02219af86ec6584e5 HEAD\x00symref=HEAD:/refs/heads/master ofs-delta multi_ack\n",
		"a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n",
		"5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c refs/tags/v2.6.11-tree\n",
		"c39ae07f393806ccf406ef966e9a15afc43cc36a refs/tags/v2.6.11-tree^{}\n",
		"7777777777777777777777777777777777777777 refs/tags/v2.6.12-tree\n",
		"8888888888888888888888888888888888888888 refs/tags/v2.6.12-tree^{}\n",
		"shallow 1111111111111111111111111111111111111111\n",
		"shallow 2222222222222222222222222222222222222222\n",
		pktline.FlushString,
	}

	s.test(c, input, expected)
}

func ExampleAdvRefs_Decode() {
	// Here is a raw advertised-ref message.
	raw := "" +
		"0065a6930aaee06755d1bdcfd943fbf614e4d92bb0c7 HEAD\x00multi_ack ofs-delta symref=HEAD:/refs/heads/master\n" +
		"003fa6930aaee06755d1bdcfd943fbf614e4d92bb0c7 refs/heads/master\n" +
		"00441111111111111111111111111111111111111111 refs/tags/v2.6.11-tree\n" +
		"00475555555555555555555555555555555555555555 refs/tags/v2.6.11-tree^{}\n" +
		"0035shallow 5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c\n" +
		"0000"

	// Use the raw message as our input.
	input := strings.NewReader(raw)

	// Decode the input into a newly allocated AdvRefs value.
	ar := NewAdvRefs()
	_ = ar.Decode(input) // error check ignored for brevity

	// Do something interesting with the AdvRefs, e.g. print its contents.
	fmt.Println("head =", ar.Head)
	fmt.Println("capabilities =", ar.Capabilities.String())
	fmt.Println("...")
	fmt.Println("shallows =", ar.Shallows)
	// Output: head = a6930aaee06755d1bdcfd943fbf614e4d92bb0c7
	// capabilities = multi_ack ofs-delta symref=HEAD:/refs/heads/master
	// ...
	// shallows = [5dc01c595e6c6ec9ccda4f6f69c131c0dd945f8c]
}

func ExampleAdvRefs_Encode() {
	// Create an AdvRefs with the contents you want...
	ar := NewAdvRefs()

	// ...add a hash for the HEAD...
	head := plumbing.NewHash("1111111111111111111111111111111111111111")
	ar.Head = &head

	// ...add some server capabilities...
	ar.Capabilities.Add(capability.MultiACK)
	ar.Capabilities.Add(capability.OFSDelta)
	ar.Capabilities.Add(capability.SymRef, "HEAD:/refs/heads/master")

	// ...add a couple of references...
	ar.References["refs/heads/master"] = plumbing.NewHash("2222222222222222222222222222222222222222")
	ar.References["refs/tags/v1"] = plumbing.NewHash("3333333333333333333333333333333333333333")

	// ...including a peeled ref...
	ar.Peeled["refs/tags/v1"] = plumbing.NewHash("4444444444444444444444444444444444444444")

	// ...and finally add a shallow
	ar.Shallows = append(ar.Shallows, plumbing.NewHash("5555555555555555555555555555555555555555"))

	// Encode the packpContents to a bytes.Buffer.
	// You can encode into stdout too, but you will not be able
	// see the '\x00' after "HEAD".
	var buf bytes.Buffer
	_ = ar.Encode(&buf) // error checks ignored for brevity

	// Print the contents of the buffer as a quoted string.
	// Printing is as a non-quoted string will be prettier but you
	// will miss the '\x00' after "HEAD".
	fmt.Printf("%q", buf.String())
	// Output:
	// "00651111111111111111111111111111111111111111 HEAD\x00multi_ack ofs-delta symref=HEAD:/refs/heads/master\n003f2222222222222222222222222222222222222222 refs/heads/master\n003a3333333333333333333333333333333333333333 refs/tags/v1\n003d4444444444444444444444444444444444444444 refs/tags/v1^{}\n0035shallow 5555555555555555555555555555555555555555\n0000"
}
