/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package diff

import (
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

// TestDiffWithRealGoCmp is a comprehensive test that compares our Diff with the actual go-cmp Diff
// across a variety of data structures and types to ensure compatibility.
func TestDiffWithRealGoCmp(t *testing.T) {
	// Test with simple types
	t.Run("SimpleTypes", func(t *testing.T) {
		testCases := []struct {
			name string
			a    interface{}
			b    interface{}
		}{
			{name: "Integers", a: 42, b: 43},
			{name: "Strings", a: "hello", b: "world"},
			{name: "Booleans", a: true, b: false},
			{name: "Floats", a: 3.14, b: 2.71},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				ourDiff := Diff(tc.a, tc.b)
				goCmpDiff := cmp.Diff(tc.a, tc.b)

				t.Logf("Our diff:\n%s\n\nGo-cmp diff:\n%s", ourDiff, goCmpDiff)

				// Verify both diffs are non-empty
				if ourDiff == "" || goCmpDiff == "" {
					t.Errorf("Expected non-empty diffs, got ourDiff: %v, goCmpDiff: %v",
						ourDiff == "", goCmpDiff == "")
				}
			})
		}
	})

	// Test with a simple struct
	t.Run("SimpleStruct", func(t *testing.T) {
		type TestStruct struct {
			Name string
			Age  int
			Tags []string
			Map  map[string]int
		}

		a := TestStruct{
			Name: "Alice",
			Age:  30,
			Tags: []string{"tag1", "tag2"},
			Map:  map[string]int{"a": 1, "b": 2},
		}

		b := TestStruct{
			Name: "Bob",
			Age:  25,
			Tags: []string{"tag1", "tag3"},
			Map:  map[string]int{"a": 1, "c": 3},
		}

		ourDiff := Diff(a, b)
		goCmpDiff := cmp.Diff(a, b)

		t.Logf("Our diff:\n%s\n\nGo-cmp diff:\n%s", ourDiff, goCmpDiff)

		// Check that our diff contains key differences
		keyDifferences := []string{
			"Name", "Alice", "Bob",
			"Age", "30", "25",
			"Tags", "tag2", "tag3",
			"Map", "b", "c",
		}

		for _, key := range keyDifferences {
			if !strings.Contains(ourDiff, key) {
				t.Errorf("Our diff doesn't contain expected key difference: %q", key)
			}
		}
	})

	// Test with a complex nested struct
	t.Run("ComplexNestedStruct", func(t *testing.T) {
		type Address struct {
			Street     string
			City       string
			State      string
			PostalCode string
			Country    string
		}

		type Contact struct {
			Type  string
			Value string
		}

		type Person struct {
			ID        int
			FirstName string
			LastName  string
			Age       int
			Addresses map[string]Address
			Contacts  []Contact
			Metadata  map[string]interface{}
			CreatedAt time.Time
		}

		now := time.Now()
		later := now.Add(24 * time.Hour)

		person1 := Person{
			ID:        1,
			FirstName: "John",
			LastName:  "Doe",
			Age:       30,
			Addresses: map[string]Address{
				"home": {
					Street:     "123 Main St",
					City:       "Anytown",
					State:      "CA",
					PostalCode: "12345",
					Country:    "USA",
				},
				"work": {
					Street:     "456 Market St",
					City:       "Worktown",
					State:      "CA",
					PostalCode: "54321",
					Country:    "USA",
				},
			},
			Contacts: []Contact{
				{Type: "email", Value: "john.doe@example.com"},
				{Type: "phone", Value: "555-1234"},
			},
			Metadata: map[string]interface{}{
				"created":    "2023-01-01",
				"loginCount": 42,
				"settings": map[string]bool{
					"notifications": true,
					"darkMode":      false,
				},
			},
			CreatedAt: now,
		}

		person2 := Person{
			ID:        1,
			FirstName: "John",
			LastName:  "Smith", // Different
			Age:       31,      // Different
			Addresses: map[string]Address{
				"home": {
					Street:     "123 Main St",
					City:       "Anytown",
					State:      "CA",
					PostalCode: "12345",
					Country:    "USA",
				},
				// "work" address is missing
			},
			Contacts: []Contact{
				{Type: "email", Value: "john.smith@example.com"}, // Different
				{Type: "phone", Value: "555-1234"},
				{Type: "fax", Value: "555-5678"}, // Additional
			},
			Metadata: map[string]interface{}{
				"created":    "2023-01-01",
				"loginCount": 43, // Different
				"settings": map[string]bool{
					"notifications": false, // Different
					"darkMode":      true,  // Different
				},
				"newField": "new value", // New field
			},
			CreatedAt: later, // Different
		}

		ourDiff := Diff(person1, person2)
		goCmpDiff := cmp.Diff(person1, person2)

		t.Logf("Our diff:\n%s\n\nGo-cmp diff:\n%s", ourDiff, goCmpDiff)

		// Check that our diff contains key differences
		keyDifferences := []string{
			"LastName", "Smith",
			"Age",
			"Addresses", "work",
			"Contacts", "john.smith@example.com", "fax",
			"Metadata", "loginCount", "settings", "notifications", "darkMode", "newField",
		}

		for _, key := range keyDifferences {
			if !strings.Contains(ourDiff, key) {
				t.Errorf("Our diff doesn't contain expected key difference: %q", key)
			}
		}

		// Check that both diffs are non-empty
		if ourDiff == "" || goCmpDiff == "" {
			t.Errorf("Expected non-empty diffs, got ourDiff: %v, goCmpDiff: %v",
				ourDiff == "", goCmpDiff == "")
		}
	})

	// Test with slices and maps
	t.Run("SlicesAndMaps", func(t *testing.T) {
		// Test with slices
		a1 := []int{1, 2, 3, 4, 5}
		b1 := []int{1, 2, 6, 4, 7}

		ourDiff1 := Diff(a1, b1)
		goCmpDiff1 := cmp.Diff(a1, b1)

		t.Logf("Our diff (slices):\n%s\n\nGo-cmp diff (slices):\n%s", ourDiff1, goCmpDiff1)

		// Check that our diff contains the differences
		if !strings.Contains(ourDiff1, "3") || !strings.Contains(ourDiff1, "6") ||
			!strings.Contains(ourDiff1, "5") || !strings.Contains(ourDiff1, "7") {
			t.Errorf("Our diff doesn't contain all expected differences for slices")
		}

		// Test with maps
		a2 := map[string]int{"a": 1, "b": 2, "c": 3}
		b2 := map[string]int{"a": 1, "b": 5, "d": 4}

		ourDiff2 := Diff(a2, b2)
		goCmpDiff2 := cmp.Diff(a2, b2)

		t.Logf("Our diff (maps):\n%s\n\nGo-cmp diff (maps):\n%s", ourDiff2, goCmpDiff2)

		// Check that our diff contains the differences
		if !strings.Contains(ourDiff2, "b") || !strings.Contains(ourDiff2, "5") ||
			!strings.Contains(ourDiff2, "c") || !strings.Contains(ourDiff2, "d") {
			t.Errorf("Our diff doesn't contain all expected differences for maps")
		}
	})

	// Test with unexported fields
	t.Run("UnexportedFields", func(t *testing.T) {
		type WithUnexported struct {
			Exported   int
			unexported int
		}

		a := WithUnexported{Exported: 1, unexported: 2}
		b := WithUnexported{Exported: 3, unexported: 4}

		ourDiff := Diff(a, b)
		// Use cmpopts.IgnoreUnexported to ignore unexported fields in go-cmp
		goCmpDiff := cmp.Diff(a, b, cmpopts.IgnoreUnexported(WithUnexported{}))

		t.Logf("Our diff (unexported):\n%s\n\nGo-cmp diff (unexported):\n%s", ourDiff, goCmpDiff)

		// Check that our diff contains only the exported field difference
		if !strings.Contains(ourDiff, "Exported") || !strings.Contains(ourDiff, "1") || !strings.Contains(ourDiff, "3") {
			t.Errorf("Our diff doesn't contain the exported field difference")
		}
	})

	// Test with embedded structs
	t.Run("EmbeddedStructs", func(t *testing.T) {
		type Embedded struct {
			Value int
		}

		type Container struct {
			Embedded
			Extra string
		}

		a := Container{Embedded: Embedded{Value: 1}, Extra: "a"}
		b := Container{Embedded: Embedded{Value: 2}, Extra: "b"}

		ourDiff := Diff(a, b)
		goCmpDiff := cmp.Diff(a, b)

		t.Logf("Our diff (embedded):\n%s\n\nGo-cmp diff (embedded):\n%s", ourDiff, goCmpDiff)

		// Check that our diff contains the container field difference
		if !strings.Contains(ourDiff, "Extra") || !strings.Contains(ourDiff, "a") || !strings.Contains(ourDiff, "b") {
			t.Errorf("Our diff doesn't contain the container field difference")
		}
	})

	// Test with interface values of same type
	t.Run("InterfaceValues", func(t *testing.T) {
		type Container struct {
			Value interface{}
		}

		// Test with same type in interface
		c := Container{Value: 42}
		d := Container{Value: 43}

		ourDiff := Diff(c, d)
		goCmpDiff := cmp.Diff(c, d)

		t.Logf("Our diff (interface same type):\n%s\n\nGo-cmp diff (interface same type):\n%s", ourDiff, goCmpDiff)

		// Check that our diff contains the value difference
		if !strings.Contains(ourDiff, "42") || !strings.Contains(ourDiff, "43") {
			t.Errorf("Our diff doesn't contain the value difference for interface values of same type")
		}
	})

	// Test with objects that cannot be marshaled to JSON
	t.Run("UnmarshalableObjects", func(t *testing.T) {
		// Test with a circular reference, which cannot be marshaled to JSON
		type Node struct {
			Value int
			Next  *Node
		}

		// Create a circular reference
		nodeA := &Node{Value: 1}
		nodeA.Next = nodeA // Points to itself

		nodeB := &Node{Value: 2}
		nodeB.Next = nodeB // Points to itself

		// This should fall back to using dump.Pretty
		circularDiff := Diff(nodeA, nodeB)

		t.Logf("Diff for circular references:\n%s", circularDiff)

		// Verify the diff contains the values
		if !strings.Contains(circularDiff, "1") || !strings.Contains(circularDiff, "2") {
			t.Errorf("Diff doesn't contain expected value differences for circular references")
		}
	})
}
