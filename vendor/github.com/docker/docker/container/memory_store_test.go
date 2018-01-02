package container

import (
	"testing"
	"time"
)

func TestNewMemoryStore(t *testing.T) {
	s := NewMemoryStore()
	m, ok := s.(*memoryStore)
	if !ok {
		t.Fatalf("store is not a memory store %v", s)
	}
	if m.s == nil {
		t.Fatal("expected store map to not be nil")
	}
}

func TestAddContainers(t *testing.T) {
	s := NewMemoryStore()
	s.Add("id", NewBaseContainer("id", "root"))
	if s.Size() != 1 {
		t.Fatalf("expected store size 1, got %v", s.Size())
	}
}

func TestGetContainer(t *testing.T) {
	s := NewMemoryStore()
	s.Add("id", NewBaseContainer("id", "root"))
	c := s.Get("id")
	if c == nil {
		t.Fatal("expected container to not be nil")
	}
}

func TestDeleteContainer(t *testing.T) {
	s := NewMemoryStore()
	s.Add("id", NewBaseContainer("id", "root"))
	s.Delete("id")
	if c := s.Get("id"); c != nil {
		t.Fatalf("expected container to be nil after removal, got %v", c)
	}

	if s.Size() != 0 {
		t.Fatalf("expected store size to be 0, got %v", s.Size())
	}
}

func TestListContainers(t *testing.T) {
	s := NewMemoryStore()

	cont := NewBaseContainer("id", "root")
	cont.Created = time.Now()
	cont2 := NewBaseContainer("id2", "root")
	cont2.Created = time.Now().Add(24 * time.Hour)

	s.Add("id", cont)
	s.Add("id2", cont2)

	list := s.List()
	if len(list) != 2 {
		t.Fatalf("expected list size 2, got %v", len(list))
	}
	if list[0].ID != "id2" {
		t.Fatalf("expected id2, got %v", list[0].ID)
	}
}

func TestFirstContainer(t *testing.T) {
	s := NewMemoryStore()

	s.Add("id", NewBaseContainer("id", "root"))
	s.Add("id2", NewBaseContainer("id2", "root"))

	first := s.First(func(cont *Container) bool {
		return cont.ID == "id2"
	})

	if first == nil {
		t.Fatal("expected container to not be nil")
	}
	if first.ID != "id2" {
		t.Fatalf("expected id2, got %v", first)
	}
}

func TestApplyAllContainer(t *testing.T) {
	s := NewMemoryStore()

	s.Add("id", NewBaseContainer("id", "root"))
	s.Add("id2", NewBaseContainer("id2", "root"))

	s.ApplyAll(func(cont *Container) {
		if cont.ID == "id2" {
			cont.ID = "newID"
		}
	})

	cont := s.Get("id2")
	if cont == nil {
		t.Fatal("expected container to not be nil")
	}
	if cont.ID != "newID" {
		t.Fatalf("expected newID, got %v", cont.ID)
	}
}
