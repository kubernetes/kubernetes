package porcupine

import (
	"context"
	"sort"
	"time"
)

type entryKind bool

const (
	callEntry   entryKind = false
	returnEntry entryKind = true
)

type entry struct {
	kind     entryKind
	value    interface{}
	id       int
	time     int64
	clientId int
	metadata interface{}
}

type LinearizationInfo struct {
	history               [][]entry // for each partition, a list of entries
	partialLinearizations [][][]int // for each partition, a set of histories (list of ids)
	annotations           []Annotation
}

// PartialLinearizations returns partial linearizations found during the
// linearizability check, as sets of operation IDs.
//
// For each partition, it returns a set of possible linearization histories,
// where each history is represented as a sequence of operation IDs. If the
// history is linearizable, this will contain a complete linearization. If not
// linearizable, it contains the maximal partial linearizations found.
func (li *LinearizationInfo) PartialLinearizations() [][][]int {
	return li.partialLinearizations
}

// PartialLinearizationsOperations returns partial linearizations found during
// the linearizability check, as sets of sequences of [Operation].
//
// For each partition, it returns a set of possible linearization histories,
// where each history is represented as a sequence of [Operation]. If the
// history is linearizable, this will contain a complete linearization. If not
// linearizable, it contains the maximal partial linearizations found.
func (li *LinearizationInfo) PartialLinearizationsOperations() [][][]Operation {
	result := make([][][]Operation, len(li.history))
	for p, partition := range li.history {
		// reconstruct operations based on entries
		callMap := make(map[int]entry)
		retMap := make(map[int]entry)
		for _, e := range partition {
			if e.kind == callEntry {
				callMap[e.id] = e
			} else {
				retMap[e.id] = e
			}
		}

		opMap := make(map[int]Operation)
		for id, call := range callMap {
			ret, ok := retMap[id]
			if !ok {
				// this should never happen, because the LinearizationInfo
				// object should always contain valid partial linearizations,
				// where there is a return for every call
				panic("cannot find corresponding return for call")
			}
			// prefer return metadata over call metadata
			metadata := call.metadata
			if ret.metadata != nil {
				metadata = ret.metadata
			}
			opMap[id] = Operation{
				ClientId: call.clientId,
				Input:    call.value,
				Call:     call.time,
				Output:   ret.value,
				Return:   ret.time,
				Metadata: metadata,
			}
		}

		partials := make([][]Operation, len(li.partialLinearizations[p]))
		for i, linearization := range li.partialLinearizations[p] {
			partials[i] = make([]Operation, len(linearization))
			for j, id := range linearization {
				op, exists := opMap[id]
				if !exists {
					// this should never happen, because the LinearizationInfo
					// object should always contain valid partial
					// linearizations, where every ID in the partial
					// linearization is in the history
					panic("cannot find operation for given id in linearization")
				}
				partials[i][j] = op
			}
		}
		result[p] = partials
	}
	return result
}

type byTime []entry

func (a byTime) Len() int {
	return len(a)
}

func (a byTime) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a byTime) Less(i, j int) bool {
	if a[i].time != a[j].time {
		return a[i].time < a[j].time
	}
	// if the timestamps are the same, we need to make sure we order calls
	// before returns
	return a[i].kind == callEntry && a[j].kind == returnEntry
}

func makeEntries(history []Operation) []entry {
	var entries []entry = nil
	id := 0
	for _, elem := range history {
		entries = append(entries, entry{
			callEntry, elem.Input, id, elem.Call, elem.ClientId, elem.Metadata})
		entries = append(entries, entry{
			returnEntry, elem.Output, id, elem.Return, elem.ClientId, elem.Metadata})
		id++
	}
	sort.Sort(byTime(entries))
	return entries
}

type node struct {
	value interface{}
	match *node // call if match is nil, otherwise return
	id    int
	next  *node
	prev  *node
}

func insertBefore(n *node, mark *node) *node {
	if mark != nil {
		beforeMark := mark.prev
		mark.prev = n
		n.next = mark
		if beforeMark != nil {
			n.prev = beforeMark
			beforeMark.next = n
		}
	}
	return n
}

func renumber(events []Event) []Event {
	var e []Event
	m := make(map[int]int) // renumbering
	id := 0
	for _, v := range events {
		if r, ok := m[v.Id]; ok {
			e = append(e, Event{ClientId: v.ClientId, Kind: v.Kind, Value: v.Value, Id: r, Metadata: v.Metadata})
		} else {
			e = append(e, Event{ClientId: v.ClientId, Kind: v.Kind, Value: v.Value, Id: id, Metadata: v.Metadata})
			m[v.Id] = id
			id++
		}
	}
	return e
}

func convertEntries(events []Event) []entry {
	var entries []entry
	for i, elem := range events {
		kind := callEntry
		if elem.Kind == ReturnEvent {
			kind = returnEntry
		}
		// use index as "time"
		entries = append(entries, entry{kind, elem.Value, elem.Id, int64(i), elem.ClientId, elem.Metadata})
	}
	return entries
}

func makeLinkedEntries(entries []entry) *node {
	var root *node = nil
	match := make(map[int]*node)
	for i := len(entries) - 1; i >= 0; i-- {
		elem := entries[i]
		if elem.kind == returnEntry {
			entry := &node{value: elem.value, match: nil, id: elem.id}
			match[elem.id] = entry
			insertBefore(entry, root)
			root = entry
		} else {
			entry := &node{value: elem.value, match: match[elem.id], id: elem.id}
			insertBefore(entry, root)
			root = entry
		}
	}
	return root
}

type cacheEntry struct {
	linearized bitset
	state      interface{}
}

func cacheKey(model Model, linearized bitset, state interface{}) uint64 {
	h := linearized.hash()
	if model.Hash != nil {
		h ^= model.Hash(state)
	}
	return h
}

func cacheContains(model Model, bucket []cacheEntry, linearized bitset, state interface{}) bool {
	for _, elem := range bucket {
		if linearized.equal(elem.linearized) && model.Equal(state, elem.state) {
			return true
		}
	}
	return false
}

type callsEntry struct {
	entry *node
	state interface{}
}

func lift(entry *node) {
	entry.prev.next = entry.next
	entry.next.prev = entry.prev
	match := entry.match
	match.prev.next = match.next
	if match.next != nil {
		match.next.prev = match.prev
	}
}

func unlift(entry *node) {
	match := entry.match
	match.prev.next = match
	if match.next != nil {
		match.next.prev = match
	}
	entry.prev.next = entry
	entry.next.prev = entry
}

func checkSingle(ctx context.Context, model Model, history []entry, computePartial bool) (bool, []*[]int) {
	entry := makeLinkedEntries(history)
	n := len(history) / 2
	linearized := newBitset(uint(n))
	cache := make(map[uint64][]cacheEntry) // map from hash to cache entry
	var calls []callsEntry
	// longest linearizable prefix that includes the given entry
	longest := make([]*[]int, n)

	state := model.Init()
	headEntry := insertBefore(&node{value: nil, match: nil, id: -1}, entry)
	for headEntry.next != nil {
		if ctx.Err() != nil {
			return false, longest
		}
		if entry.match != nil {
			matching := entry.match // the return entry
			ok, newState := model.StepContext(ctx, state, entry.value, matching.value)
			if ctx.Err() != nil {
				return false, longest
			}
			if ok {
				linearized.set(uint(entry.id))
				key := cacheKey(model, linearized, newState)
				bucket := cache[key]
				if !cacheContains(model, bucket, linearized, newState) {
					cache[key] = append(bucket, cacheEntry{linearized.clone(), newState})
					calls = append(calls, callsEntry{entry, state})
					state = newState
					lift(entry)
					entry = headEntry.next
				} else {
					linearized.clear(uint(entry.id))
					entry = entry.next
				}
			} else {
				entry = entry.next
			}
		} else {
			if len(calls) == 0 {
				return false, longest
			}
			// longest
			if computePartial {
				callsLen := len(calls)
				var seq []int = nil
				for _, v := range calls {
					if longest[v.entry.id] == nil || callsLen > len(*longest[v.entry.id]) {
						// create seq lazily
						if seq == nil {
							seq = make([]int, len(calls))
							for i, v := range calls {
								seq[i] = v.entry.id
							}
						}
						longest[v.entry.id] = &seq
					}
				}
			}
			callsTop := calls[len(calls)-1]
			entry = callsTop.entry
			state = callsTop.state
			linearized.clear(uint(entry.id))
			calls = calls[:len(calls)-1]
			unlift(entry)
			entry = entry.next
		}
	}
	// longest linearization is the complete linearization, which is calls
	seq := make([]int, len(calls))
	for i, v := range calls {
		seq[i] = v.entry.id
	}
	for i := 0; i < n; i++ {
		longest[i] = &seq
	}
	return true, longest
}

func fillDefault(model Model) Model {
	if model.Partition == nil {
		model.Partition = noPartition
	}
	if model.PartitionEvent == nil {
		model.PartitionEvent = noPartitionEvent
	}
	if model.Equal == nil {
		model.Equal = shallowEqual
	}
	if model.DescribeOperation == nil {
		model.DescribeOperation = defaultDescribeOperation
	}
	if model.DescribeState == nil {
		model.DescribeState = defaultDescribeState
	}
	if model.DescribeOperationMetadata == nil {
		model.DescribeOperationMetadata = defaultDescribeOperationMetadata
	}
	switch {
	case model.Step == nil && model.StepContext == nil:
		panic("model must define Step or StepContext")
	case model.Step == nil:
		ctx := context.Background()
		model.Step = func(state, input, output interface{}) (bool, interface{}) {
			return model.StepContext(ctx, state, input, output)
		}
	case model.StepContext == nil:
		model.StepContext = func(ctx context.Context, state interface{}, input interface{}, output interface{}) (bool, interface{}) {
			return model.Step(state, input, output)
		}
	}
	return model
}

func checkParallel(model Model, history [][]entry, computeInfo bool, timeout time.Duration) (CheckResult, LinearizationInfo) {
	if len(history) == 0 {
		return Ok, LinearizationInfo{}
	}
	ok := true
	timedOut := false
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	results := make(chan bool, len(history))
	longest := make([][]*[]int, len(history))
	for i, subhistory := range history {
		go func(i int, subhistory []entry) {
			ok, l := checkSingle(ctx, model, subhistory, computeInfo)
			longest[i] = l
			results <- ok
		}(i, subhistory)
	}
	var timeoutChan <-chan time.Time
	if timeout > 0 {
		timeoutChan = time.After(timeout)
	}
	count := 0
loop:
	for {
		select {
		case result := <-results:
			count++
			ok = ok && result
			if !ok && !computeInfo {
				cancel()
				break loop
			}
			if count >= len(history) {
				break loop
			}
		case <-timeoutChan:
			timedOut = true
			cancel()
			break loop // if we time out, we might get a false positive
		}
	}
	var info LinearizationInfo
	if computeInfo {
		// make sure we've waited for all goroutines to finish,
		// otherwise we might race on access to longest[]
		for count < len(history) {
			<-results
			count++
		}
		// return longest linearizable prefixes that include each history element
		partialLinearizations := make([][][]int, len(history))
		for i := 0; i < len(history); i++ {
			var partials [][]int
			// turn longest into a set of unique linearizations
			set := make(map[*[]int]struct{})
			for _, v := range longest[i] {
				if v != nil {
					set[v] = struct{}{}
				}
			}
			for k := range set {
				arr := make([]int, len(*k))
				copy(arr, *k)
				partials = append(partials, arr)
			}
			partialLinearizations[i] = partials
		}
		info.history = history
		info.partialLinearizations = partialLinearizations
	}
	var result CheckResult
	if !ok {
		result = Illegal
	} else {
		if timedOut {
			result = Unknown
		} else {
			result = Ok
		}
	}
	return result, info
}

func checkEvents(model Model, history []Event, verbose bool, timeout time.Duration) (CheckResult, LinearizationInfo) {
	model = fillDefault(model)
	partitions := model.PartitionEvent(history)
	l := make([][]entry, len(partitions))
	for i, subhistory := range partitions {
		l[i] = convertEntries(renumber(subhistory))
	}
	return checkParallel(model, l, verbose, timeout)
}

func checkOperations(model Model, history []Operation, verbose bool, timeout time.Duration) (CheckResult, LinearizationInfo) {
	model = fillDefault(model)
	partitions := model.Partition(history)
	l := make([][]entry, len(partitions))
	for i, subhistory := range partitions {
		l[i] = makeEntries(subhistory)
	}
	return checkParallel(model, l, verbose, timeout)
}
