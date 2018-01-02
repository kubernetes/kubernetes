// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dnsmessage

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"
)

func mustNewName(name string) Name {
	n, err := NewName(name)
	if err != nil {
		panic(err)
	}
	return n
}

func (m *Message) String() string {
	s := fmt.Sprintf("Message: %#v\n", &m.Header)
	if len(m.Questions) > 0 {
		s += "-- Questions\n"
		for _, q := range m.Questions {
			s += fmt.Sprintf("%#v\n", q)
		}
	}
	if len(m.Answers) > 0 {
		s += "-- Answers\n"
		for _, a := range m.Answers {
			s += fmt.Sprintf("%#v\n", a)
		}
	}
	if len(m.Authorities) > 0 {
		s += "-- Authorities\n"
		for _, ns := range m.Authorities {
			s += fmt.Sprintf("%#v\n", ns)
		}
	}
	if len(m.Additionals) > 0 {
		s += "-- Additionals\n"
		for _, e := range m.Additionals {
			s += fmt.Sprintf("%#v\n", e)
		}
	}
	return s
}

func TestNameString(t *testing.T) {
	want := "foo"
	name := mustNewName(want)
	if got := fmt.Sprint(name); got != want {
		t.Errorf("got fmt.Sprint(%#v) = %s, want = %s", name, got, want)
	}
}

func TestQuestionPackUnpack(t *testing.T) {
	want := Question{
		Name:  mustNewName("."),
		Type:  TypeA,
		Class: ClassINET,
	}
	buf, err := want.pack(make([]byte, 1, 50), map[string]int{})
	if err != nil {
		t.Fatal("Packing failed:", err)
	}
	var p Parser
	p.msg = buf
	p.header.questions = 1
	p.section = sectionQuestions
	p.off = 1
	got, err := p.Question()
	if err != nil {
		t.Fatalf("Unpacking failed: %v\n%s", err, string(buf[1:]))
	}
	if p.off != len(buf) {
		t.Errorf("Unpacked different amount than packed: got n = %d, want = %d", p.off, len(buf))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got = %+v, want = %+v", got, want)
	}
}

func TestName(t *testing.T) {
	tests := []string{
		"",
		".",
		"google..com",
		"google.com",
		"google..com.",
		"google.com.",
		".google.com.",
		"www..google.com.",
		"www.google.com.",
	}

	for _, test := range tests {
		n, err := NewName(test)
		if err != nil {
			t.Errorf("Creating name for %q: %v", test, err)
			continue
		}
		if ns := n.String(); ns != test {
			t.Errorf("Got %#v.String() = %q, want = %q", n, ns, test)
			continue
		}
	}
}

func TestNamePackUnpack(t *testing.T) {
	tests := []struct {
		in   string
		want string
		err  error
	}{
		{"", "", errNonCanonicalName},
		{".", ".", nil},
		{"google..com", "", errNonCanonicalName},
		{"google.com", "", errNonCanonicalName},
		{"google..com.", "", errZeroSegLen},
		{"google.com.", "google.com.", nil},
		{".google.com.", "", errZeroSegLen},
		{"www..google.com.", "", errZeroSegLen},
		{"www.google.com.", "www.google.com.", nil},
	}

	for _, test := range tests {
		in := mustNewName(test.in)
		want := mustNewName(test.want)
		buf, err := in.pack(make([]byte, 0, 30), map[string]int{})
		if err != test.err {
			t.Errorf("Packing of %q: got err = %v, want err = %v", test.in, err, test.err)
			continue
		}
		if test.err != nil {
			continue
		}
		var got Name
		n, err := got.unpack(buf, 0)
		if err != nil {
			t.Errorf("Unpacking for %q failed: %v", test.in, err)
			continue
		}
		if n != len(buf) {
			t.Errorf(
				"Unpacked different amount than packed for %q: got n = %d, want = %d",
				test.in,
				n,
				len(buf),
			)
		}
		if got != want {
			t.Errorf("Unpacking packing of %q: got = %#v, want = %#v", test.in, got, want)
		}
	}
}

func checkErrorPrefix(err error, prefix string) bool {
	e, ok := err.(*nestedError)
	return ok && e.s == prefix
}

func TestHeaderUnpackError(t *testing.T) {
	wants := []string{
		"id",
		"bits",
		"questions",
		"answers",
		"authorities",
		"additionals",
	}
	var buf []byte
	var h header
	for _, want := range wants {
		n, err := h.unpack(buf, 0)
		if n != 0 || !checkErrorPrefix(err, want) {
			t.Errorf("got h.unpack([%d]byte, 0) = %d, %v, want = 0, %s", len(buf), n, err, want)
		}
		buf = append(buf, 0, 0)
	}
}

func TestParserStart(t *testing.T) {
	const want = "unpacking header"
	var p Parser
	for i := 0; i <= 1; i++ {
		_, err := p.Start([]byte{})
		if !checkErrorPrefix(err, want) {
			t.Errorf("got p.Start(nil) = _, %v, want = _, %s", err, want)
		}
	}
}

func TestResourceNotStarted(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*Parser) error
	}{
		{"CNAMEResource", func(p *Parser) error { _, err := p.CNAMEResource(); return err }},
		{"MXResource", func(p *Parser) error { _, err := p.MXResource(); return err }},
		{"NSResource", func(p *Parser) error { _, err := p.NSResource(); return err }},
		{"PTRResource", func(p *Parser) error { _, err := p.PTRResource(); return err }},
		{"SOAResource", func(p *Parser) error { _, err := p.SOAResource(); return err }},
		{"TXTResource", func(p *Parser) error { _, err := p.TXTResource(); return err }},
		{"SRVResource", func(p *Parser) error { _, err := p.SRVResource(); return err }},
		{"AResource", func(p *Parser) error { _, err := p.AResource(); return err }},
		{"AAAAResource", func(p *Parser) error { _, err := p.AAAAResource(); return err }},
	}

	for _, test := range tests {
		if err := test.fn(&Parser{}); err != ErrNotStarted {
			t.Errorf("got _, %v = p.%s(), want = _, %v", err, test.name, ErrNotStarted)
		}
	}
}

func TestDNSPackUnpack(t *testing.T) {
	wants := []Message{
		{
			Questions: []Question{
				{
					Name:  mustNewName("."),
					Type:  TypeAAAA,
					Class: ClassINET,
				},
			},
			Answers:     []Resource{},
			Authorities: []Resource{},
			Additionals: []Resource{},
		},
		largeTestMsg(),
	}
	for i, want := range wants {
		b, err := want.Pack()
		if err != nil {
			t.Fatalf("%d: packing failed: %v", i, err)
		}
		var got Message
		err = got.Unpack(b)
		if err != nil {
			t.Fatalf("%d: unpacking failed: %v", i, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%d: got = %+v, want = %+v", i, &got, &want)
		}
	}
}

func TestSkipAll(t *testing.T) {
	msg := largeTestMsg()
	buf, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing large test message:", err)
	}
	var p Parser
	if _, err := p.Start(buf); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		f    func() error
	}{
		{"SkipAllQuestions", p.SkipAllQuestions},
		{"SkipAllAnswers", p.SkipAllAnswers},
		{"SkipAllAuthorities", p.SkipAllAuthorities},
		{"SkipAllAdditionals", p.SkipAllAdditionals},
	}
	for _, test := range tests {
		for i := 1; i <= 3; i++ {
			if err := test.f(); err != nil {
				t.Errorf("Call #%d to %s(): %v", i, test.name, err)
			}
		}
	}
}

func TestSkipEach(t *testing.T) {
	msg := smallTestMsg()

	buf, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing test message:", err)
	}
	var p Parser
	if _, err := p.Start(buf); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		f    func() error
	}{
		{"SkipQuestion", p.SkipQuestion},
		{"SkipAnswer", p.SkipAnswer},
		{"SkipAuthority", p.SkipAuthority},
		{"SkipAdditional", p.SkipAdditional},
	}
	for _, test := range tests {
		if err := test.f(); err != nil {
			t.Errorf("First call: got %s() = %v, want = %v", test.name, err, nil)
		}
		if err := test.f(); err != ErrSectionDone {
			t.Errorf("Second call: got %s() = %v, want = %v", test.name, err, ErrSectionDone)
		}
	}
}

func TestSkipAfterRead(t *testing.T) {
	msg := smallTestMsg()

	buf, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing test message:", err)
	}
	var p Parser
	if _, err := p.Start(buf); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		skip func() error
		read func() error
	}{
		{"Question", p.SkipQuestion, func() error { _, err := p.Question(); return err }},
		{"Answer", p.SkipAnswer, func() error { _, err := p.Answer(); return err }},
		{"Authority", p.SkipAuthority, func() error { _, err := p.Authority(); return err }},
		{"Additional", p.SkipAdditional, func() error { _, err := p.Additional(); return err }},
	}
	for _, test := range tests {
		if err := test.read(); err != nil {
			t.Errorf("Got %s() = _, %v, want = _, %v", test.name, err, nil)
		}
		if err := test.skip(); err != ErrSectionDone {
			t.Errorf("Got Skip%s() = %v, want = %v", test.name, err, ErrSectionDone)
		}
	}
}

func TestSkipNotStarted(t *testing.T) {
	var p Parser

	tests := []struct {
		name string
		f    func() error
	}{
		{"SkipAllQuestions", p.SkipAllQuestions},
		{"SkipAllAnswers", p.SkipAllAnswers},
		{"SkipAllAuthorities", p.SkipAllAuthorities},
		{"SkipAllAdditionals", p.SkipAllAdditionals},
	}
	for _, test := range tests {
		if err := test.f(); err != ErrNotStarted {
			t.Errorf("Got %s() = %v, want = %v", test.name, err, ErrNotStarted)
		}
	}
}

func TestTooManyRecords(t *testing.T) {
	const recs = int(^uint16(0)) + 1
	tests := []struct {
		name string
		msg  Message
		want error
	}{
		{
			"Questions",
			Message{
				Questions: make([]Question, recs),
			},
			errTooManyQuestions,
		},
		{
			"Answers",
			Message{
				Answers: make([]Resource, recs),
			},
			errTooManyAnswers,
		},
		{
			"Authorities",
			Message{
				Authorities: make([]Resource, recs),
			},
			errTooManyAuthorities,
		},
		{
			"Additionals",
			Message{
				Additionals: make([]Resource, recs),
			},
			errTooManyAdditionals,
		},
	}

	for _, test := range tests {
		if _, got := test.msg.Pack(); got != test.want {
			t.Errorf("Packing %d %s: got = %v, want = %v", recs, test.name, got, test.want)
		}
	}
}

func TestVeryLongTxt(t *testing.T) {
	want := Resource{
		ResourceHeader{
			Name:  mustNewName("foo.bar.example.com."),
			Type:  TypeTXT,
			Class: ClassINET,
		},
		&TXTResource{loremIpsum},
	}
	buf, err := want.pack(make([]byte, 0, 8000), map[string]int{})
	if err != nil {
		t.Fatal("Packing failed:", err)
	}
	var got Resource
	off, err := got.Header.unpack(buf, 0)
	if err != nil {
		t.Fatal("Unpacking ResourceHeader failed:", err)
	}
	body, n, err := unpackResourceBody(buf, off, got.Header)
	if err != nil {
		t.Fatal("Unpacking failed:", err)
	}
	got.Body = body
	if n != len(buf) {
		t.Errorf("Unpacked different amount than packed: got n = %d, want = %d", n, len(buf))
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Got = %#v, want = %#v", got, want)
	}
}

func TestStartError(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*Builder) error
	}{
		{"Questions", func(b *Builder) error { return b.StartQuestions() }},
		{"Answers", func(b *Builder) error { return b.StartAnswers() }},
		{"Authorities", func(b *Builder) error { return b.StartAuthorities() }},
		{"Additionals", func(b *Builder) error { return b.StartAdditionals() }},
	}

	envs := []struct {
		name string
		fn   func() *Builder
		want error
	}{
		{"sectionNotStarted", func() *Builder { return &Builder{section: sectionNotStarted} }, ErrNotStarted},
		{"sectionDone", func() *Builder { return &Builder{section: sectionDone} }, ErrSectionDone},
	}

	for _, env := range envs {
		for _, test := range tests {
			if got := test.fn(env.fn()); got != env.want {
				t.Errorf("got Builder{%s}.Start%s = %v, want = %v", env.name, test.name, got, env.want)
			}
		}
	}
}

func TestBuilderResourceError(t *testing.T) {
	tests := []struct {
		name string
		fn   func(*Builder) error
	}{
		{"CNAMEResource", func(b *Builder) error { return b.CNAMEResource(ResourceHeader{}, CNAMEResource{}) }},
		{"MXResource", func(b *Builder) error { return b.MXResource(ResourceHeader{}, MXResource{}) }},
		{"NSResource", func(b *Builder) error { return b.NSResource(ResourceHeader{}, NSResource{}) }},
		{"PTRResource", func(b *Builder) error { return b.PTRResource(ResourceHeader{}, PTRResource{}) }},
		{"SOAResource", func(b *Builder) error { return b.SOAResource(ResourceHeader{}, SOAResource{}) }},
		{"TXTResource", func(b *Builder) error { return b.TXTResource(ResourceHeader{}, TXTResource{}) }},
		{"SRVResource", func(b *Builder) error { return b.SRVResource(ResourceHeader{}, SRVResource{}) }},
		{"AResource", func(b *Builder) error { return b.AResource(ResourceHeader{}, AResource{}) }},
		{"AAAAResource", func(b *Builder) error { return b.AAAAResource(ResourceHeader{}, AAAAResource{}) }},
	}

	envs := []struct {
		name string
		fn   func() *Builder
		want error
	}{
		{"sectionNotStarted", func() *Builder { return &Builder{section: sectionNotStarted} }, ErrNotStarted},
		{"sectionHeader", func() *Builder { return &Builder{section: sectionHeader} }, ErrNotStarted},
		{"sectionQuestions", func() *Builder { return &Builder{section: sectionQuestions} }, ErrNotStarted},
		{"sectionDone", func() *Builder { return &Builder{section: sectionDone} }, ErrSectionDone},
	}

	for _, env := range envs {
		for _, test := range tests {
			if got := test.fn(env.fn()); got != env.want {
				t.Errorf("got Builder{%s}.%s = %v, want = %v", env.name, test.name, got, env.want)
			}
		}
	}
}

func TestFinishError(t *testing.T) {
	var b Builder
	want := ErrNotStarted
	if _, got := b.Finish(); got != want {
		t.Errorf("got Builder{}.Finish() = %v, want = %v", got, want)
	}
}

func TestBuilder(t *testing.T) {
	msg := largeTestMsg()
	want, err := msg.Pack()
	if err != nil {
		t.Fatal("Packing without builder:", err)
	}

	var b Builder
	b.Start(nil, msg.Header)

	if err := b.StartQuestions(); err != nil {
		t.Fatal("b.StartQuestions():", err)
	}
	for _, q := range msg.Questions {
		if err := b.Question(q); err != nil {
			t.Fatalf("b.Question(%#v): %v", q, err)
		}
	}

	if err := b.StartAnswers(); err != nil {
		t.Fatal("b.StartAnswers():", err)
	}
	for _, a := range msg.Answers {
		switch a.Header.Type {
		case TypeA:
			if err := b.AResource(a.Header, *a.Body.(*AResource)); err != nil {
				t.Fatalf("b.AResource(%#v): %v", a, err)
			}
		case TypeNS:
			if err := b.NSResource(a.Header, *a.Body.(*NSResource)); err != nil {
				t.Fatalf("b.NSResource(%#v): %v", a, err)
			}
		case TypeCNAME:
			if err := b.CNAMEResource(a.Header, *a.Body.(*CNAMEResource)); err != nil {
				t.Fatalf("b.CNAMEResource(%#v): %v", a, err)
			}
		case TypeSOA:
			if err := b.SOAResource(a.Header, *a.Body.(*SOAResource)); err != nil {
				t.Fatalf("b.SOAResource(%#v): %v", a, err)
			}
		case TypePTR:
			if err := b.PTRResource(a.Header, *a.Body.(*PTRResource)); err != nil {
				t.Fatalf("b.PTRResource(%#v): %v", a, err)
			}
		case TypeMX:
			if err := b.MXResource(a.Header, *a.Body.(*MXResource)); err != nil {
				t.Fatalf("b.MXResource(%#v): %v", a, err)
			}
		case TypeTXT:
			if err := b.TXTResource(a.Header, *a.Body.(*TXTResource)); err != nil {
				t.Fatalf("b.TXTResource(%#v): %v", a, err)
			}
		case TypeAAAA:
			if err := b.AAAAResource(a.Header, *a.Body.(*AAAAResource)); err != nil {
				t.Fatalf("b.AAAAResource(%#v): %v", a, err)
			}
		case TypeSRV:
			if err := b.SRVResource(a.Header, *a.Body.(*SRVResource)); err != nil {
				t.Fatalf("b.SRVResource(%#v): %v", a, err)
			}
		}
	}

	if err := b.StartAuthorities(); err != nil {
		t.Fatal("b.StartAuthorities():", err)
	}
	for _, a := range msg.Authorities {
		if err := b.NSResource(a.Header, *a.Body.(*NSResource)); err != nil {
			t.Fatalf("b.NSResource(%#v): %v", a, err)
		}
	}

	if err := b.StartAdditionals(); err != nil {
		t.Fatal("b.StartAdditionals():", err)
	}
	for _, a := range msg.Additionals {
		if err := b.TXTResource(a.Header, *a.Body.(*TXTResource)); err != nil {
			t.Fatalf("b.TXTResource(%#v): %v", a, err)
		}
	}

	got, err := b.Finish()
	if err != nil {
		t.Fatal("b.Finish():", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("Got from Builder: %#v\nwant = %#v", got, want)
	}
}

func TestResourcePack(t *testing.T) {
	for _, tt := range []struct {
		m   Message
		err error
	}{
		{
			Message{
				Questions: []Question{
					{
						Name:  mustNewName("."),
						Type:  TypeAAAA,
						Class: ClassINET,
					},
				},
				Answers: []Resource{{ResourceHeader{}, nil}},
			},
			&nestedError{"packing Answer", errNilResouceBody},
		},
		{
			Message{
				Questions: []Question{
					{
						Name:  mustNewName("."),
						Type:  TypeAAAA,
						Class: ClassINET,
					},
				},
				Authorities: []Resource{{ResourceHeader{}, (*NSResource)(nil)}},
			},
			&nestedError{"packing Authority",
				&nestedError{"ResourceHeader",
					&nestedError{"Name", errNonCanonicalName},
				},
			},
		},
		{
			Message{
				Questions: []Question{
					{
						Name:  mustNewName("."),
						Type:  TypeA,
						Class: ClassINET,
					},
				},
				Additionals: []Resource{{ResourceHeader{}, nil}},
			},
			&nestedError{"packing Additional", errNilResouceBody},
		},
	} {
		_, err := tt.m.Pack()
		if !reflect.DeepEqual(err, tt.err) {
			t.Errorf("got %v for %v; want %v", err, tt.m, tt.err)
		}
	}
}

func BenchmarkParsing(b *testing.B) {
	b.ReportAllocs()

	name := mustNewName("foo.bar.example.com.")
	msg := Message{
		Header: Header{Response: true, Authoritative: true},
		Questions: []Question{
			{
				Name:  name,
				Type:  TypeA,
				Class: ClassINET,
			},
		},
		Answers: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&AResource{[4]byte{}},
			},
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&AAAAResource{[16]byte{}},
			},
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&CNAMEResource{name},
			},
			{
				ResourceHeader{
					Name:  name,
					Class: ClassINET,
				},
				&NSResource{name},
			},
		},
	}

	buf, err := msg.Pack()
	if err != nil {
		b.Fatal("msg.Pack():", err)
	}

	for i := 0; i < b.N; i++ {
		var p Parser
		if _, err := p.Start(buf); err != nil {
			b.Fatal("p.Start(buf):", err)
		}

		for {
			_, err := p.Question()
			if err == ErrSectionDone {
				break
			}
			if err != nil {
				b.Fatal("p.Question():", err)
			}
		}

		for {
			h, err := p.AnswerHeader()
			if err == ErrSectionDone {
				break
			}
			if err != nil {
				panic(err)
			}

			switch h.Type {
			case TypeA:
				if _, err := p.AResource(); err != nil {
					b.Fatal("p.AResource():", err)
				}
			case TypeAAAA:
				if _, err := p.AAAAResource(); err != nil {
					b.Fatal("p.AAAAResource():", err)
				}
			case TypeCNAME:
				if _, err := p.CNAMEResource(); err != nil {
					b.Fatal("p.CNAMEResource():", err)
				}
			case TypeNS:
				if _, err := p.NSResource(); err != nil {
					b.Fatal("p.NSResource():", err)
				}
			default:
				b.Fatalf("unknown type: %T", h)
			}
		}
	}
}

func BenchmarkBuilding(b *testing.B) {
	b.ReportAllocs()

	name := mustNewName("foo.bar.example.com.")
	buf := make([]byte, 0, packStartingCap)

	for i := 0; i < b.N; i++ {
		var bld Builder
		bld.StartWithoutCompression(buf, Header{Response: true, Authoritative: true})

		if err := bld.StartQuestions(); err != nil {
			b.Fatal("bld.StartQuestions():", err)
		}
		q := Question{
			Name:  name,
			Type:  TypeA,
			Class: ClassINET,
		}
		if err := bld.Question(q); err != nil {
			b.Fatalf("bld.Question(%+v): %v", q, err)
		}

		hdr := ResourceHeader{
			Name:  name,
			Class: ClassINET,
		}
		if err := bld.StartAnswers(); err != nil {
			b.Fatal("bld.StartQuestions():", err)
		}

		ar := AResource{[4]byte{}}
		if err := bld.AResource(hdr, ar); err != nil {
			b.Fatalf("bld.AResource(%+v, %+v): %v", hdr, ar, err)
		}

		aaar := AAAAResource{[16]byte{}}
		if err := bld.AAAAResource(hdr, aaar); err != nil {
			b.Fatalf("bld.AAAAResource(%+v, %+v): %v", hdr, aaar, err)
		}

		cnr := CNAMEResource{name}
		if err := bld.CNAMEResource(hdr, cnr); err != nil {
			b.Fatalf("bld.CNAMEResource(%+v, %+v): %v", hdr, cnr, err)
		}

		nsr := NSResource{name}
		if err := bld.NSResource(hdr, nsr); err != nil {
			b.Fatalf("bld.NSResource(%+v, %+v): %v", hdr, nsr, err)
		}

		if _, err := bld.Finish(); err != nil {
			b.Fatal("bld.Finish():", err)
		}
	}
}

func smallTestMsg() Message {
	name := mustNewName("example.com.")
	return Message{
		Header: Header{Response: true, Authoritative: true},
		Questions: []Question{
			{
				Name:  name,
				Type:  TypeA,
				Class: ClassINET,
			},
		},
		Answers: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
		},
		Authorities: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
		},
		Additionals: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
		},
	}
}

func largeTestMsg() Message {
	name := mustNewName("foo.bar.example.com.")
	return Message{
		Header: Header{Response: true, Authoritative: true},
		Questions: []Question{
			{
				Name:  name,
				Type:  TypeA,
				Class: ClassINET,
			},
		},
		Answers: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 1}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeA,
					Class: ClassINET,
				},
				&AResource{[4]byte{127, 0, 0, 2}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeAAAA,
					Class: ClassINET,
				},
				&AAAAResource{[16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeCNAME,
					Class: ClassINET,
				},
				&CNAMEResource{mustNewName("alias.example.com.")},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeSOA,
					Class: ClassINET,
				},
				&SOAResource{
					NS:      mustNewName("ns1.example.com."),
					MBox:    mustNewName("mb.example.com."),
					Serial:  1,
					Refresh: 2,
					Retry:   3,
					Expire:  4,
					MinTTL:  5,
				},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypePTR,
					Class: ClassINET,
				},
				&PTRResource{mustNewName("ptr.example.com.")},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeMX,
					Class: ClassINET,
				},
				&MXResource{
					7,
					mustNewName("mx.example.com."),
				},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeSRV,
					Class: ClassINET,
				},
				&SRVResource{
					8,
					9,
					11,
					mustNewName("srv.example.com."),
				},
			},
		},
		Authorities: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeNS,
					Class: ClassINET,
				},
				&NSResource{mustNewName("ns1.example.com.")},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeNS,
					Class: ClassINET,
				},
				&NSResource{mustNewName("ns2.example.com.")},
			},
		},
		Additionals: []Resource{
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeTXT,
					Class: ClassINET,
				},
				&TXTResource{"So Long, and Thanks for All the Fish"},
			},
			{
				ResourceHeader{
					Name:  name,
					Type:  TypeTXT,
					Class: ClassINET,
				},
				&TXTResource{"Hamster Huey and the Gooey Kablooie"},
			},
		},
	}
}

const loremIpsum = `
Lorem ipsum dolor sit amet, nec enim antiopam id, an ullum choro
nonumes qui, pro eu debet honestatis mediocritatem. No alia enim eos,
magna signiferumque ex vis. Mei no aperiri dissentias, cu vel quas
regione. Malorum quaeque vim ut, eum cu semper aliquid invidunt, ei
nam ipsum assentior.

Nostrum appellantur usu no, vis ex probatus adipiscing. Cu usu illum
facilis eleifend. Iusto conceptam complectitur vim id. Tale omnesque
no usu, ei oblique sadipscing vim. At nullam voluptua usu, mei laudem
reformidans et. Qui ei eros porro reformidans, ius suas veritus
torquatos ex. Mea te facer alterum consequat.

Soleat torquatos democritum sed et, no mea congue appareat, facer
aliquam nec in. Has te ipsum tritani. At justo dicta option nec, movet
phaedrum ad nam. Ea detracto verterem liberavisse has, delectus
suscipiantur in mei. Ex nam meliore complectitur. Ut nam omnis
honestatis quaerendum, ea mea nihil affert detracto, ad vix rebum
mollis.

Ut epicurei praesent neglegentur pri, prima fuisset intellegebat ad
vim. An habemus comprehensam usu, at enim dignissim pro. Eam reque
vivendum adipisci ea. Vel ne odio choro minimum. Sea admodum
dissentiet ex. Mundi tamquam evertitur ius cu. Homero postea iisque ut
pro, vel ne saepe senserit consetetur.

Nulla utamur facilisis ius ea, in viderer diceret pertinax eum. Mei no
enim quodsi facilisi, ex sed aeterno appareat mediocritatem, eum
sententiae deterruisset ut. At suas timeam euismod cum, offendit
appareat interpretaris ne vix. Vel ea civibus albucius, ex vim quidam
accusata intellegebat, noluisse instructior sea id. Nec te nonumes
habemus appellantur, quis dignissim vituperata eu nam.

At vix apeirian patrioque vituperatoribus, an usu agam assum. Debet
iisque an mea. Per eu dicant ponderum accommodare. Pri alienum
placerat senserit an, ne eum ferri abhorreant vituperatoribus. Ut mea
eligendi disputationi. Ius no tation everti impedit, ei magna quidam
mediocritatem pri.

Legendos perpetua iracundia ne usu, no ius ullum epicurei intellegam,
ad modus epicuri lucilius eam. In unum quaerendum usu. Ne diam paulo
has, ea veri virtute sed. Alia honestatis conclusionemque mea eu, ut
iudico albucius his.

Usu essent probatus eu, sed omnis dolor delicatissimi ex. No qui augue
dissentias dissentiet. Laudem recteque no usu, vel an velit noluisse,
an sed utinam eirmod appetere. Ne mea fuisset inimicus ocurreret. At
vis dicant abhorreant, utinam forensibus nec ne, mei te docendi
consequat. Brute inermis persecuti cum id. Ut ipsum munere propriae
usu, dicit graeco disputando id has.

Eros dolore quaerendum nam ei. Timeam ornatus inciderint pro id. Nec
torquatos sadipscing ei, ancillae molestie per in. Malis principes duo
ea, usu liber postulant ei.

Graece timeam voluptatibus eu eam. Alia probatus quo no, ea scripta
feugiat duo. Congue option meliore ex qui, noster invenire appellantur
ea vel. Eu exerci legendos vel. Consetetur repudiandae vim ut. Vix an
probo minimum, et nam illud falli tempor.

Cum dico signiferumque eu. Sed ut regione maiorum, id veritus insolens
tacimates vix. Eu mel sint tamquam lucilius, duo no oporteat
tacimates. Atqui augue concludaturque vix ei, id mel utroque menandri.

Ad oratio blandit aliquando pro. Vis et dolorum rationibus
philosophia, ad cum nulla molestie. Hinc fuisset adversarium eum et,
ne qui nisl verear saperet, vel te quaestio forensibus. Per odio
option delenit an. Alii placerat has no, in pri nihil platonem
cotidieque. Est ut elit copiosae scaevola, debet tollit maluisset sea
an.

Te sea hinc debet pericula, liber ridens fabulas cu sed, quem mutat
accusam mea et. Elitr labitur albucius et pri, an labore feugait mel.
Velit zril melius usu ea. Ad stet putent interpretaris qui. Mel no
error volumus scripserit. In pro paulo iudico, quo ei dolorem
verterem, affert fabellas dissentiet ea vix.

Vis quot deserunt te. Error aliquid detraxit eu usu, vis alia eruditi
salutatus cu. Est nostrud bonorum an, ei usu alii salutatus. Vel at
nisl primis, eum ex aperiri noluisse reformidans. Ad veri velit
utroque vis, ex equidem detraxit temporibus has.

Inermis appareat usu ne. Eros placerat periculis mea ad, in dictas
pericula pro. Errem postulant at usu, ea nec amet ornatus mentitum. Ad
mazim graeco eum, vel ex percipit volutpat iudicabit, sit ne delicata
interesset. Mel sapientem prodesset abhorreant et, oblique suscipit
eam id.

An maluisset disputando mea, vidit mnesarchum pri et. Malis insolens
inciderint no sea. Ea persius maluisset vix, ne vim appellantur
instructior, consul quidam definiebas pri id. Cum integre feugiat
pericula in, ex sed persius similique, mel ne natum dicit percipitur.

Primis discere ne pri, errem putent definitionem at vis. Ei mel dolore
neglegentur, mei tincidunt percipitur ei. Pro ad simul integre
rationibus. Eu vel alii honestatis definitiones, mea no nonumy
reprehendunt.

Dicta appareat legendos est cu. Eu vel congue dicunt omittam, no vix
adhuc minimum constituam, quot noluisse id mel. Eu quot sale mutat
duo, ex nisl munere invenire duo. Ne nec ullum utamur. Pro alterum
debitis nostrum no, ut vel aliquid vivendo.

Aliquip fierent praesent quo ne, id sit audiam recusabo delicatissimi.
Usu postulant incorrupte cu. At pro dicit tibique intellegam, cibo
dolore impedit id eam, et aeque feugait assentior has. Quando sensibus
nec ex. Possit sensibus pri ad, unum mutat periculis cu vix.

Mundi tibique vix te, duo simul partiendo qualisque id, est at vidit
sonet tempor. No per solet aeterno deseruisse. Petentium salutandi
definiebas pri cu. Munere vivendum est in. Ei justo congue eligendi
vis, modus offendit omittantur te mel.

Integre voluptaria in qui, sit habemus tractatos constituam no. Utinam
melius conceptam est ne, quo in minimum apeirian delicata, ut ius
porro recusabo. Dicant expetenda vix no, ludus scripserit sed ex, eu
his modo nostro. Ut etiam sonet his, quodsi inciderint philosophia te
per. Nullam lobortis eu cum, vix an sonet efficiendi repudiandae. Vis
ad idque fabellas intellegebat.

Eum commodo senserit conclusionemque ex. Sed forensibus sadipscing ut,
mei in facer delicata periculis, sea ne hinc putent cetero. Nec ne
alia corpora invenire, alia prima soleat te cum. Eleifend posidonium
nam at.

Dolorum indoctum cu quo, ex dolor legendos recteque eam, cu pri zril
discere. Nec civibus officiis dissentiunt ex, est te liber ludus
elaboraret. Cum ea fabellas invenire. Ex vim nostrud eripuit
comprehensam, nam te inermis delectus, saepe inermis senserit.
`
