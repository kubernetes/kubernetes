package main

import (
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

// In this test, lowercap strings are ones that need to be picked up for
// translation, whereas uppercap strings should not be picked up.

func main() {
	p := message.NewPrinter(language.English)

	// TODO: probably should use type instead of string content for argument
	// substitution.
	wrapf(p, "inline %s", "ARG1")
	gwrapf("global printer used %s", "ARG1")

	w := wrapped{p}

	// Comment about wrapf.
	w.wrapf("number: %d, string: %s, bool: %v", 2, "STRING ARG", true)
	w.wrapf("empty string")
	w.wrap("Lovely weather today!")

	more(&w)
}

var printer = message.NewPrinter(language.English)

func more(w wrapper) {
	w.wrap("number one")
	w.wrapf("speed of light: %s", "C")
}

func gwrapf(format string, args ...interface{}) {
	v := format
	a := args
	printer.Printf(v, a...)
}

func wrapf(p *message.Printer, format string, args ...interface{}) {
	v := format
	a := args
	p.Printf(v, a...)
}

func wrap(p *message.Printer, format string) {
	v := format
	b := "0"
	a := []interface{}{3, b}
	s := a[:]
	p.Printf(v, s...)
}

type wrapper interface {
	wrapf(format string, args ...interface{})
	wrap(msg string)
}

type wrapped struct {
	p *message.Printer
}

// TODO: calls over interfaces do not get picked up. It looks like this is
// because w is not a pointer receiver, while the other method is. Mixing of
// receiver types does not seem to be allowed by callgraph/cha.
func (w wrapped) wrapf(format string, args ...interface{}) {
	w.p.Printf(format, args...)
}

func (w *wrapped) wrap(msg string) {
	w.p.Printf(msg)
}

func fint(p *message.Printer, x int) {
	v := "number: %d"
	const c = "DAFDA"
	p.Printf(v, c)
}

const format = "constant local" + " %s"

// NOTE: pass is not called. Ensure it is picked up anyway.
func pass(p *message.Printer, args ...interface{}) {
	// TODO: find an example caller to find substituted types and argument
	// examples.
	p.Sprintf(format, args...)
}

func lookup(p *message.Printer, x int) {
	// TODO: pick up all elements from slice foo.
	p.Printf(foo[x])
}

var foo = []string{
	"aaaa",
	"bbbb",
}

func field(p *message.Printer, x int) {
	// TODO: pick up strings in field BAR from all composite literals of
	// typeof(strct.Foo.Bar).
	p.Printf(strct.Foo.Bar, x)
}

type fooStruct struct {
	Foo barStruct
}

type barStruct struct {
	other int
	Bar   string
}

var strct = fooStruct{
	Foo: barStruct{0, "foo %d"},
}

func call(p *message.Printer, x int) {
	// TODO: pick up constant return values.
	p.Printf(fn())
}

func fn() string {
	return "const str"
}

// Both strings get picked up.
func ifConst(p *message.Printer, cond bool, arg1 string) {
	a := "foo %s %s"
	if cond {
		a = "bar %s %s"
	}
	b := "FOO"
	if cond {
		b = "BAR"
	}
	wrapf(p, a, arg1, b)
}

// Pick up all non-empty strings in this function.
func ifConst2(x int) {
	a := ""
	switch x {
	case 0:
		a = "foo"
	case 1:
		a = "bar"
	case 2:
		a = "baz"
	}
	gwrapf(a)
}

// TODO: pick up strings passed to the second argument in calls to freeVar.
func freeVar(p *message.Printer, str string) {
	fn := func(p *message.Printer) {
		p.Printf(str)
	}
	fn(p)
}

func freeConst(p *message.Printer) {
	// str is a message
	const str = "const str"
	fn := func(p *message.Printer) {
		p.Printf(str)
	}
	fn(p)
}

func global(p *message.Printer) {
	// city describes the expected next meeting place
	city := "Amsterdam"
	// See a person around.
	p.Printf(globalStr, city)
}

// globalStr is a global variable with a string constant assigned to it.
var globalStr = "See you around in %s!"

func global2(p *message.Printer) {
	const food = "Pastrami"
	wrapf(p, constFood,
		food, // the food to be consumed by the subject
	)
}

// Comment applying to all constants in a block are ignored.
var (
	// Ho ho ho
	notAMessage, constFood, msgHello = "NOPE!", consume, hello
)

// A block comment.
var (
	// This comment takes precedence.
	hello = "Hello, %d and %s!"

	consume = "Please eat your %s!"
)
