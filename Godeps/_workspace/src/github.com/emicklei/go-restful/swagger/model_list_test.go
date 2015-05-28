package swagger

import (
	"encoding/json"
	"testing"
)

func TestModelList(t *testing.T) {
	m := Model{}
	m.Id = "m"
	l := ModelList{}
	l.Put("m", m)
	k, ok := l.At("m")
	if !ok {
		t.Error("want model back")
	}
	if got, want := k.Id, "m"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestModelList_Marshal(t *testing.T) {
	l := ModelList{}
	m := Model{Id: "myid"}
	l.Put("myid", m)
	data, err := json.Marshal(l)
	if err != nil {
		t.Error(err)
	}
	if got, want := string(data), `{"myid":{"id":"myid","properties":{}}}`; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}

func TestModelList_Unmarshal(t *testing.T) {
	data := `{"myid":{"id":"myid","properties":{}}}`
	l := ModelList{}
	if err := json.Unmarshal([]byte(data), &l); err != nil {
		t.Error(err)
	}
	m, ok := l.At("myid")
	if !ok {
		t.Error("expected myid")
	}
	if got, want := m.Id, "myid"; got != want {
		t.Errorf("got %v want %v", got, want)
	}
}
