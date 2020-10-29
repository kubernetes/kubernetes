package rankedset

import "github.com/google/btree"

// Item represents a single object in a RankedSet.
type Item interface {
	// Key returns the unique identifier for this item.
	Key() string
	// Rank is used to sort items.
	// Items with the same rank are sorted lexicographically based on Key.
	Rank() int64
}

// RankedSet stores Items based on Key (uniqueness) and Rank (sorting).
type RankedSet struct {
	rank *btree.BTree
	set  map[string]*treeItem
}

// StringItem implements Item using a string.
// It has two main uses:
// 1. If all items in a RankedSet are StringItems, the set becomes a store of unique strings sorted lexicographically.
// 2. It serves as a Key item that can be passed into methods that ignore Rank such as RankedSet.Delete.
type StringItem string

func (s StringItem) Key() string {
	return string(s)
}

func (s StringItem) Rank() int64 {
	return 0
}

func New() *RankedSet {
	return &RankedSet{
		rank: btree.New(32),
		set:  make(map[string]*treeItem),
	}
}

// Insert adds the item into the set.
// If an item with the same Key existed in the set, it is deleted and returned.
func (s *RankedSet) Insert(item Item) Item {
	old := s.Delete(item)

	key := item.Key()
	value := &treeItem{item: item}

	s.rank.ReplaceOrInsert(value) // should always return nil because we call Delete first
	s.set[key] = value

	return old
}

// Delete removes the item from the set based on Key (Rank is ignored).
// The removed item is returned if it existed in the set.
func (s *RankedSet) Delete(item Item) Item {
	key := item.Key()
	value, ok := s.set[key]
	if !ok {
		return nil
	}

	s.rank.Delete(value) // should always return the same data as value (non-nil)
	delete(s.set, key)

	return value.item
}

func (s *RankedSet) Min() Item {
	if min := s.rank.Min(); min != nil {
		return min.(*treeItem).item
	}
	return nil
}

func (s *RankedSet) Max() Item {
	if max := s.rank.Max(); max != nil {
		return max.(*treeItem).item
	}
	return nil
}

func (s *RankedSet) Len() int {
	return len(s.set)
}

func (s *RankedSet) Get(item Item) Item {
	if value, ok := s.set[item.Key()]; ok {
		return value.item
	}
	return nil
}

func (s *RankedSet) Has(item Item) bool {
	_, ok := s.set[item.Key()]
	return ok
}

// List returns all items in the set in ranked order.
// If delete is set to true, the returned items are removed from the set.
func (s *RankedSet) List(delete bool) []Item {
	return s.ascend(
		func(item Item) bool {
			return true
		},
		delete,
	)
}

// LessThan returns all items less than the given rank in ranked order.
// If delete is set to true, the returned items are removed from the set.
func (s *RankedSet) LessThan(rank int64, delete bool) []Item {
	return s.ascend(
		func(item Item) bool {
			return item.Rank() < rank
		},
		delete,
	)
}

// setItemIterator allows callers of ascend to iterate in-order over the set.
// When this function returns false, iteration will stop.
type setItemIterator func(item Item) bool

func (s *RankedSet) ascend(iterator setItemIterator, delete bool) []Item {
	var items []Item
	s.rank.Ascend(func(i btree.Item) bool {
		item := i.(*treeItem).item
		if !iterator(item) {
			return false
		}
		items = append(items, item)
		return true
	})
	// delete after Ascend since it is probably not safe to remove while iterating
	if delete {
		for _, item := range items {
			s.Delete(item)
		}
	}
	return items
}

var _ btree.Item = &treeItem{}

type treeItem struct {
	item Item
}

func (i *treeItem) Less(than btree.Item) bool {
	other := than.(*treeItem).item

	selfRank := i.item.Rank()
	otherRank := other.Rank()

	if selfRank == otherRank {
		return i.item.Key() < other.Key()
	}

	return selfRank < otherRank
}
