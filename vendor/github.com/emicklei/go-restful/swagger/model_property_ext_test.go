package swagger

import (
	"net"
	"testing"
)

// clear && go test -v -test.run TestThatExtraTagsAreReadIntoModel ...swagger
func TestThatExtraTagsAreReadIntoModel(t *testing.T) {
	type fakeint int
	type Anything struct {
		Name     string  `description:"name" modelDescription:"a test"`
		Size     int     `minimum:"0" maximum:"10"`
		Stati    string  `enum:"off|on" default:"on" modelDescription:"more description"`
		ID       string  `unique:"true"`
		FakeInt  fakeint `type:"integer"`
		IP       net.IP  `type:"string"`
		Password string
	}
	m := modelsFromStruct(Anything{})
	props, _ := m.At("swagger.Anything")
	p1, _ := props.Properties.At("Name")
	if got, want := p1.Description, "name"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	p2, _ := props.Properties.At("Size")
	if got, want := p2.Minimum, "0"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	if got, want := p2.Maximum, "10"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	p3, _ := props.Properties.At("Stati")
	if got, want := p3.Enum[0], "off"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	if got, want := p3.Enum[1], "on"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	p4, _ := props.Properties.At("ID")
	if got, want := *p4.UniqueItems, true; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	p5, _ := props.Properties.At("Password")
	if got, want := *p5.Type, "string"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	p6, _ := props.Properties.At("FakeInt")
	if got, want := *p6.Type, "integer"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
	p7, _ := props.Properties.At("IP")
	if got, want := *p7.Type, "string"; got != want {
		t.Errorf("got %v want %v", got, want)
	}

	if got, want := props.Description, "a test\nmore description"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}
