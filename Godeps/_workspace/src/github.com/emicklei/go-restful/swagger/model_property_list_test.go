package swagger

import (
	"encoding/json"
	"testing"
)

func TestModelPropertyList(t *testing.T) {
	l := ModelPropertyList{}
	p := ModelProperty{Description: "d"}
	l.Put("p", p)
	q, ok := l.At("p")
	if !ok {
		t.Error("expected p")
	}
	if got, want := q.Description, "d"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestModelPropertyList_Marshal(t *testing.T) {
	l := ModelPropertyList{}
	p := ModelProperty{Description: "d"}
	l.Put("p", p)
	data, err := json.Marshal(l)
	if err != nil {
		t.Error(err)
	}
	if got, want := string(data), `{"p":{"description":"d"}}`; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestModelPropertyList_Unmarshal(t *testing.T) {
	data := `{"p":{"description":"d"}}`
	l := ModelPropertyList{}
	if err := json.Unmarshal([]byte(data), &l); err != nil {
		t.Error(err)
	}
	m, ok := l.At("p")
	if !ok {
		t.Error("expected p")
	}
	if got, want := m.Description, "d"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}
