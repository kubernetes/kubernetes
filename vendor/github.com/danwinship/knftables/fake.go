/*
Copyright 2023 Red Hat, Inc.

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

package knftables

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// Fake is a fake implementation of Interface
type Fake struct {
	nftContext

	nextHandle int

	// Table contains the Interface's table. This will be `nil` until you `tx.Add()`
	// the table.
	Table *FakeTable
}

// FakeTable wraps Table for the Fake implementation
type FakeTable struct {
	Table

	// Chains contains the table's chains, keyed by name
	Chains map[string]*FakeChain

	// Sets contains the table's sets, keyed by name
	Sets map[string]*FakeSet

	// Maps contains the table's maps, keyed by name
	Maps map[string]*FakeMap
}

// FakeChain wraps Chain for the Fake implementation
type FakeChain struct {
	Chain

	// Rules contains the chain's rules, in order
	Rules []*Rule
}

// FakeSet wraps Set for the Fake implementation
type FakeSet struct {
	Set

	// Elements contains the set's elements. You can also use the FakeSet's
	// FindElement() method to see if a particular element is present.
	Elements []*Element
}

// FakeMap wraps Set for the Fake implementation
type FakeMap struct {
	Map

	// Elements contains the map's elements. You can also use the FakeMap's
	// FindElement() method to see if a particular element is present.
	Elements []*Element
}

// NewFake creates a new fake Interface, for unit tests
func NewFake(family Family, table string) *Fake {
	return &Fake{
		nftContext: nftContext{
			family: family,
			table:  table,
		},
	}
}

var _ Interface = &Fake{}

// List is part of Interface.
func (fake *Fake) List(ctx context.Context, objectType string) ([]string, error) {
	if fake.Table == nil {
		return nil, notFoundError("no such table %q", fake.table)
	}

	var result []string

	switch objectType {
	case "chain", "chains":
		for name := range fake.Table.Chains {
			result = append(result, name)
		}
	case "set", "sets":
		for name := range fake.Table.Sets {
			result = append(result, name)
		}
	case "map", "maps":
		for name := range fake.Table.Maps {
			result = append(result, name)
		}

	default:
		return nil, fmt.Errorf("unsupported object type %q", objectType)
	}

	return result, nil
}

// ListRules is part of Interface
func (fake *Fake) ListRules(ctx context.Context, chain string) ([]*Rule, error) {
	if fake.Table == nil {
		return nil, notFoundError("no such chain %q", chain)
	}
	ch := fake.Table.Chains[chain]
	if ch == nil {
		return nil, notFoundError("no such chain %q", chain)
	}
	return ch.Rules, nil
}

// ListElements is part of Interface
func (fake *Fake) ListElements(ctx context.Context, objectType, name string) ([]*Element, error) {
	if fake.Table == nil {
		return nil, notFoundError("no such %s %q", objectType, name)
	}
	if objectType == "set" {
		s := fake.Table.Sets[name]
		if s != nil {
			return s.Elements, nil
		}
	} else if objectType == "map" {
		m := fake.Table.Maps[name]
		if m != nil {
			return m.Elements, nil
		}
	}
	return nil, notFoundError("no such %s %q", objectType, name)
}

// NewTransaction is part of Interface
func (fake *Fake) NewTransaction() *Transaction {
	return &Transaction{nftContext: &fake.nftContext}
}

// Run is part of Interface
func (fake *Fake) Run(ctx context.Context, tx *Transaction) error {
	if tx.err != nil {
		return tx.err
	}

	// FIXME: not actually transactional!

	for _, op := range tx.operations {
		// If the table hasn't been created, and this isn't a Table operation, then fail
		if fake.Table == nil {
			if _, ok := op.obj.(*Table); !ok {
				return notFoundError("no such table \"%s %s\"", fake.family, fake.table)
			}
		}

		if op.verb == addVerb || op.verb == createVerb || op.verb == insertVerb {
			fake.nextHandle++
		}

		switch obj := op.obj.(type) {
		case *Table:
			err := checkExists(op.verb, "table", fake.table, fake.Table != nil)
			if err != nil {
				return err
			}
			switch op.verb {
			case flushVerb:
				fake.Table = nil
				fallthrough
			case addVerb, createVerb:
				if fake.Table != nil {
					continue
				}
				table := *obj
				table.Handle = PtrTo(fake.nextHandle)
				fake.Table = &FakeTable{
					Table:  table,
					Chains: make(map[string]*FakeChain),
					Sets:   make(map[string]*FakeSet),
					Maps:   make(map[string]*FakeMap),
				}
			case deleteVerb:
				fake.Table = nil
			default:
				return fmt.Errorf("unhandled operation %q", op.verb)
			}

		case *Chain:
			existingChain := fake.Table.Chains[obj.Name]
			err := checkExists(op.verb, "chain", obj.Name, existingChain != nil)
			if err != nil {
				return err
			}
			switch op.verb {
			case addVerb, createVerb:
				if existingChain != nil {
					continue
				}
				chain := *obj
				chain.Handle = PtrTo(fake.nextHandle)
				fake.Table.Chains[obj.Name] = &FakeChain{
					Chain: chain,
				}
			case flushVerb:
				existingChain.Rules = nil
			case deleteVerb:
				// FIXME delete-by-handle
				delete(fake.Table.Chains, obj.Name)
			default:
				return fmt.Errorf("unhandled operation %q", op.verb)
			}

		case *Rule:
			existingChain := fake.Table.Chains[obj.Chain]
			if existingChain == nil {
				return notFoundError("no such chain %q", obj.Chain)
			}
			if op.verb == deleteVerb {
				i := findRule(existingChain.Rules, *obj.Handle)
				if i == -1 {
					return notFoundError("no rule with handle %d", *obj.Handle)
				}
				existingChain.Rules = append(existingChain.Rules[:i], existingChain.Rules[i+1:]...)
				continue
			}

			rule := *obj
			refRule := -1
			if rule.Handle != nil {
				refRule = findRule(existingChain.Rules, *obj.Handle)
				if refRule == -1 {
					return notFoundError("no rule with handle %d", *obj.Handle)
				}
			} else if obj.Index != nil {
				if *obj.Index >= len(existingChain.Rules) {
					return notFoundError("no rule with index %d", *obj.Index)
				}
				refRule = *obj.Index
			}

			switch op.verb {
			case addVerb:
				if refRule == -1 {
					existingChain.Rules = append(existingChain.Rules, &rule)
				} else {
					existingChain.Rules = append(existingChain.Rules[:refRule+1], append([]*Rule{&rule}, existingChain.Rules[refRule+1:]...)...)
				}
				rule.Handle = PtrTo(fake.nextHandle)
			case insertVerb:
				if refRule == -1 {
					existingChain.Rules = append([]*Rule{&rule}, existingChain.Rules...)
				} else {
					existingChain.Rules = append(existingChain.Rules[:refRule], append([]*Rule{&rule}, existingChain.Rules[refRule:]...)...)
				}
				rule.Handle = PtrTo(fake.nextHandle)
			case replaceVerb:
				existingChain.Rules[refRule] = &rule
			default:
				return fmt.Errorf("unhandled operation %q", op.verb)
			}

		case *Set:
			existingSet := fake.Table.Sets[obj.Name]
			err := checkExists(op.verb, "set", obj.Name, existingSet != nil)
			if err != nil {
				return err
			}
			switch op.verb {
			case addVerb, createVerb:
				if existingSet != nil {
					continue
				}
				set := *obj
				set.Handle = PtrTo(fake.nextHandle)
				fake.Table.Sets[obj.Name] = &FakeSet{
					Set: set,
				}
			case flushVerb:
				existingSet.Elements = nil
			case deleteVerb:
				// FIXME delete-by-handle
				delete(fake.Table.Sets, obj.Name)
			default:
				return fmt.Errorf("unhandled operation %q", op.verb)
			}
		case *Map:
			existingMap := fake.Table.Maps[obj.Name]
			err := checkExists(op.verb, "map", obj.Name, existingMap != nil)
			if err != nil {
				return err
			}
			switch op.verb {
			case addVerb:
				if existingMap != nil {
					continue
				}
				mapObj := *obj
				mapObj.Handle = PtrTo(fake.nextHandle)
				fake.Table.Maps[obj.Name] = &FakeMap{
					Map: mapObj,
				}
			case flushVerb:
				existingMap.Elements = nil
			case deleteVerb:
				// FIXME delete-by-handle
				delete(fake.Table.Maps, obj.Name)
			default:
				return fmt.Errorf("unhandled operation %q", op.verb)
			}
		case *Element:
			if len(obj.Value) == 0 {
				existingSet := fake.Table.Sets[obj.Set]
				if existingSet == nil {
					return notFoundError("no such set %q", obj.Set)
				}
				switch op.verb {
				case addVerb, createVerb:
					element := *obj
					if i := findElement(existingSet.Elements, element.Key); i != -1 {
						if op.verb == createVerb {
							return existsError("element %q already exists", strings.Join(element.Key, " . "))
						}
						existingSet.Elements[i] = &element
					} else {
						existingSet.Elements = append(existingSet.Elements, &element)
					}
				case deleteVerb:
					element := *obj
					if i := findElement(existingSet.Elements, element.Key); i != -1 {
						existingSet.Elements = append(existingSet.Elements[:i], existingSet.Elements[i+1:]...)
					} else {
						return notFoundError("no such element %q", strings.Join(element.Key, " . "))
					}
				default:
					return fmt.Errorf("unhandled operation %q", op.verb)
				}
			} else {
				existingMap := fake.Table.Maps[obj.Map]
				if existingMap == nil {
					return notFoundError("no such map %q", obj.Map)
				}
				switch op.verb {
				case addVerb, createVerb:
					element := *obj
					if i := findElement(existingMap.Elements, element.Key); i != -1 {
						if op.verb == createVerb {
							return existsError("element %q already exists", strings.Join(element.Key, ". "))
						}
						existingMap.Elements[i] = &element
					} else {
						existingMap.Elements = append(existingMap.Elements, &element)
					}
				case deleteVerb:
					element := *obj
					if i := findElement(existingMap.Elements, element.Key); i != -1 {
						existingMap.Elements = append(existingMap.Elements[:i], existingMap.Elements[i+1:]...)
					} else {
						return notFoundError("no such element %q", strings.Join(element.Key, " . "))
					}
				default:
					return fmt.Errorf("unhandled operation %q", op.verb)
				}
			}
		default:
			return fmt.Errorf("unhandled object type %T", op.obj)
		}
	}

	return nil
}

func checkExists(verb verb, objectType, name string, exists bool) error {
	switch verb {
	case addVerb:
		// It's fine if the object either exists or doesn't
		return nil
	case createVerb:
		if exists {
			return existsError("%s %q already exists", objectType, name)
		}
	default:
		if !exists {
			return notFoundError("no such %s %q", objectType, name)
		}
	}
	return nil
}

// Dump dumps the current contents of fake, in a way that looks like an nft transaction,
// but not actually guaranteed to be usable as such. (e.g., chains may be referenced
// before they are created, etc)
func (fake *Fake) Dump() string {
	if fake.Table == nil {
		return ""
	}

	buf := &strings.Builder{}

	table := fake.Table
	table.writeOperation(addVerb, &fake.nftContext, buf)

	for _, cname := range sortKeys(table.Chains) {
		ch := table.Chains[cname]
		ch.writeOperation(addVerb, &fake.nftContext, buf)

		for _, rule := range ch.Rules {
			// Avoid outputing handles
			dumpRule := *rule
			dumpRule.Handle = nil
			dumpRule.Index = nil
			dumpRule.writeOperation(addVerb, &fake.nftContext, buf)
		}
	}

	for _, sname := range sortKeys(table.Sets) {
		s := table.Sets[sname]
		s.writeOperation(addVerb, &fake.nftContext, buf)

		for _, element := range s.Elements {
			element.writeOperation(addVerb, &fake.nftContext, buf)
		}
	}
	for _, mname := range sortKeys(table.Maps) {
		m := table.Maps[mname]
		m.writeOperation(addVerb, &fake.nftContext, buf)

		for _, element := range m.Elements {
			element.writeOperation(addVerb, &fake.nftContext, buf)
		}
	}

	return buf.String()
}

func sortKeys[K ~string, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func findRule(rules []*Rule, handle int) int {
	for i := range rules {
		if rules[i].Handle != nil && *rules[i].Handle == handle {
			return i
		}
	}
	return -1
}

func findElement(elements []*Element, key []string) int {
	for i := range elements {
		if reflect.DeepEqual(elements[i].Key, key) {
			return i
		}
	}
	return -1
}

// FindElement finds an element of the set with the given key. If there is no matching
// element, it returns nil.
func (s *FakeSet) FindElement(key ...string) *Element {
	index := findElement(s.Elements, key)
	if index == -1 {
		return nil
	}
	return s.Elements[index]
}

// FindElement finds an element of the map with the given key. If there is no matching
// element, it returns nil.
func (m *FakeMap) FindElement(key ...string) *Element {
	index := findElement(m.Elements, key)
	if index == -1 {
		return nil
	}
	return m.Elements[index]
}
