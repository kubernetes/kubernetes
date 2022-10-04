package internal

import (
	"fmt"
	"reflect"
	"sort"

	"sync"

	"github.com/onsi/ginkgo/v2/types"
)

var _global_node_id_counter = uint(0)
var _global_id_mutex = &sync.Mutex{}

func UniqueNodeID() uint {
	//There's a reace in the internal integration tests if we don't make
	//accessing _global_node_id_counter safe across goroutines.
	_global_id_mutex.Lock()
	defer _global_id_mutex.Unlock()
	_global_node_id_counter += 1
	return _global_node_id_counter
}

type Node struct {
	ID       uint
	NodeType types.NodeType

	Text         string
	Body         func()
	CodeLocation types.CodeLocation
	NestingLevel int

	SynchronizedBeforeSuiteProc1Body    func() []byte
	SynchronizedBeforeSuiteAllProcsBody func([]byte)

	SynchronizedAfterSuiteAllProcsBody func()
	SynchronizedAfterSuiteProc1Body    func()

	ReportEachBody       func(types.SpecReport)
	ReportAfterSuiteBody func(types.Report)

	MarkedFocus                     bool
	MarkedPending                   bool
	MarkedSerial                    bool
	MarkedOrdered                   bool
	MarkedOncePerOrdered            bool
	MarkedSuppressProgressReporting bool
	FlakeAttempts                   int
	Labels                          Labels

	NodeIDWhereCleanupWasGenerated uint
}

// Decoration Types
type focusType bool
type pendingType bool
type serialType bool
type orderedType bool
type honorsOrderedType bool
type suppressProgressReporting bool

const Focus = focusType(true)
const Pending = pendingType(true)
const Serial = serialType(true)
const Ordered = orderedType(true)
const OncePerOrdered = honorsOrderedType(true)
const SuppressProgressReporting = suppressProgressReporting(true)

type FlakeAttempts uint
type Offset uint
type Done chan<- interface{} // Deprecated Done Channel for asynchronous testing
type Labels []string

func UnionOfLabels(labels ...Labels) Labels {
	out := Labels{}
	seen := map[string]bool{}
	for _, labelSet := range labels {
		for _, label := range labelSet {
			if !seen[label] {
				seen[label] = true
				out = append(out, label)
			}
		}
	}
	return out
}

func PartitionDecorations(args ...interface{}) ([]interface{}, []interface{}) {
	decorations := []interface{}{}
	remainingArgs := []interface{}{}
	for _, arg := range args {
		if isDecoration(arg) {
			decorations = append(decorations, arg)
		} else {
			remainingArgs = append(remainingArgs, arg)
		}
	}
	return decorations, remainingArgs
}

func isDecoration(arg interface{}) bool {
	switch t := reflect.TypeOf(arg); {
	case t == nil:
		return false
	case t == reflect.TypeOf(Offset(0)):
		return true
	case t == reflect.TypeOf(types.CodeLocation{}):
		return true
	case t == reflect.TypeOf(Focus):
		return true
	case t == reflect.TypeOf(Pending):
		return true
	case t == reflect.TypeOf(Serial):
		return true
	case t == reflect.TypeOf(Ordered):
		return true
	case t == reflect.TypeOf(OncePerOrdered):
		return true
	case t == reflect.TypeOf(SuppressProgressReporting):
		return true
	case t == reflect.TypeOf(FlakeAttempts(0)):
		return true
	case t == reflect.TypeOf(Labels{}):
		return true
	case t.Kind() == reflect.Slice && isSliceOfDecorations(arg):
		return true
	default:
		return false
	}
}

func isSliceOfDecorations(slice interface{}) bool {
	vSlice := reflect.ValueOf(slice)
	if vSlice.Len() == 0 {
		return false
	}
	for i := 0; i < vSlice.Len(); i++ {
		if !isDecoration(vSlice.Index(i).Interface()) {
			return false
		}
	}
	return true
}

func NewNode(deprecationTracker *types.DeprecationTracker, nodeType types.NodeType, text string, args ...interface{}) (Node, []error) {
	baseOffset := 2
	node := Node{
		ID:           UniqueNodeID(),
		NodeType:     nodeType,
		Text:         text,
		Labels:       Labels{},
		CodeLocation: types.NewCodeLocation(baseOffset),
		NestingLevel: -1,
	}
	errors := []error{}
	appendError := func(err error) {
		if err != nil {
			errors = append(errors, err)
		}
	}

	args = unrollInterfaceSlice(args)

	remainingArgs := []interface{}{}
	//First get the CodeLocation up-to-date
	for _, arg := range args {
		switch v := arg.(type) {
		case Offset:
			node.CodeLocation = types.NewCodeLocation(baseOffset + int(v))
		case types.CodeLocation:
			node.CodeLocation = v
		default:
			remainingArgs = append(remainingArgs, arg)
		}
	}

	labelsSeen := map[string]bool{}
	trackedFunctionError := false
	args = remainingArgs
	remainingArgs = []interface{}{}
	//now process the rest of the args
	for _, arg := range args {
		switch t := reflect.TypeOf(arg); {
		case t == reflect.TypeOf(float64(0)):
			break //ignore deprecated timeouts
		case t == reflect.TypeOf(Focus):
			node.MarkedFocus = bool(arg.(focusType))
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "Focus"))
			}
		case t == reflect.TypeOf(Pending):
			node.MarkedPending = bool(arg.(pendingType))
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "Pending"))
			}
		case t == reflect.TypeOf(Serial):
			node.MarkedSerial = bool(arg.(serialType))
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "Serial"))
			}
		case t == reflect.TypeOf(Ordered):
			node.MarkedOrdered = bool(arg.(orderedType))
			if !nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "Ordered"))
			}
		case t == reflect.TypeOf(OncePerOrdered):
			node.MarkedOncePerOrdered = bool(arg.(honorsOrderedType))
			if !nodeType.Is(types.NodeTypeBeforeEach | types.NodeTypeJustBeforeEach | types.NodeTypeAfterEach | types.NodeTypeJustAfterEach) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "OncePerOrdered"))
			}
		case t == reflect.TypeOf(SuppressProgressReporting):
			node.MarkedSuppressProgressReporting = bool(arg.(suppressProgressReporting))
			if nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "SuppressProgressReporting"))
			}
		case t == reflect.TypeOf(FlakeAttempts(0)):
			node.FlakeAttempts = int(arg.(FlakeAttempts))
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "FlakeAttempts"))
			}
		case t == reflect.TypeOf(Labels{}):
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "Label"))
			}
			for _, label := range arg.(Labels) {
				if !labelsSeen[label] {
					labelsSeen[label] = true
					label, err := types.ValidateAndCleanupLabel(label, node.CodeLocation)
					node.Labels = append(node.Labels, label)
					appendError(err)
				}
			}
		case t.Kind() == reflect.Func:
			if nodeType.Is(types.NodeTypeReportBeforeEach | types.NodeTypeReportAfterEach) {
				if node.ReportEachBody != nil {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}

				//we can trust that the function is valid because the compiler has our back here
				node.ReportEachBody = arg.(func(types.SpecReport))
				break
			}

			if node.Body != nil {
				appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
				trackedFunctionError = true
				break
			}
			isValid := (t.NumOut() == 0) && (t.NumIn() <= 1) && (t.NumIn() == 0 || t.In(0) == reflect.TypeOf(make(Done)))
			if !isValid {
				appendError(types.GinkgoErrors.InvalidBodyType(t, node.CodeLocation, nodeType))
				trackedFunctionError = true
				break
			}
			if t.NumIn() == 0 {
				node.Body = arg.(func())
			} else {
				deprecationTracker.TrackDeprecation(types.Deprecations.Async(), node.CodeLocation)
				deprecatedAsyncBody := arg.(func(Done))
				node.Body = func() { deprecatedAsyncBody(make(Done)) }
			}
		default:
			remainingArgs = append(remainingArgs, arg)
		}
	}

	//validations
	if node.MarkedPending && node.MarkedFocus {
		appendError(types.GinkgoErrors.InvalidDeclarationOfFocusedAndPending(node.CodeLocation, nodeType))
	}

	if node.Body == nil && node.ReportEachBody == nil && !node.MarkedPending && !trackedFunctionError {
		appendError(types.GinkgoErrors.MissingBodyFunction(node.CodeLocation, nodeType))
	}
	for _, arg := range remainingArgs {
		appendError(types.GinkgoErrors.UnknownDecorator(node.CodeLocation, nodeType, arg))
	}

	if len(errors) > 0 {
		return Node{}, errors
	}

	return node, errors
}

func NewSynchronizedBeforeSuiteNode(proc1Body func() []byte, allProcsBody func([]byte), codeLocation types.CodeLocation) (Node, []error) {
	return Node{
		ID:                                  UniqueNodeID(),
		NodeType:                            types.NodeTypeSynchronizedBeforeSuite,
		SynchronizedBeforeSuiteProc1Body:    proc1Body,
		SynchronizedBeforeSuiteAllProcsBody: allProcsBody,
		CodeLocation:                        codeLocation,
	}, nil
}

func NewSynchronizedAfterSuiteNode(allProcsBody func(), proc1Body func(), codeLocation types.CodeLocation) (Node, []error) {
	return Node{
		ID:                                 UniqueNodeID(),
		NodeType:                           types.NodeTypeSynchronizedAfterSuite,
		SynchronizedAfterSuiteAllProcsBody: allProcsBody,
		SynchronizedAfterSuiteProc1Body:    proc1Body,
		CodeLocation:                       codeLocation,
	}, nil
}

func NewReportAfterSuiteNode(text string, body func(types.Report), codeLocation types.CodeLocation) (Node, []error) {
	return Node{
		ID:                   UniqueNodeID(),
		Text:                 text,
		NodeType:             types.NodeTypeReportAfterSuite,
		ReportAfterSuiteBody: body,
		CodeLocation:         codeLocation,
	}, nil
}

func NewCleanupNode(fail func(string, types.CodeLocation), args ...interface{}) (Node, []error) {
	baseOffset := 2
	node := Node{
		ID:           UniqueNodeID(),
		NodeType:     types.NodeTypeCleanupInvalid,
		CodeLocation: types.NewCodeLocation(baseOffset),
		NestingLevel: -1,
	}
	remainingArgs := []interface{}{}
	for _, arg := range args {
		switch t := reflect.TypeOf(arg); {
		case t == reflect.TypeOf(Offset(0)):
			node.CodeLocation = types.NewCodeLocation(baseOffset + int(arg.(Offset)))
		case t == reflect.TypeOf(types.CodeLocation{}):
			node.CodeLocation = arg.(types.CodeLocation)
		default:
			remainingArgs = append(remainingArgs, arg)
		}
	}

	if len(remainingArgs) == 0 {
		return Node{}, []error{types.GinkgoErrors.DeferCleanupInvalidFunction(node.CodeLocation)}
	}
	callback := reflect.ValueOf(remainingArgs[0])
	if !(callback.Kind() == reflect.Func && callback.Type().NumOut() <= 1) {
		return Node{}, []error{types.GinkgoErrors.DeferCleanupInvalidFunction(node.CodeLocation)}
	}
	callArgs := []reflect.Value{}
	for _, arg := range remainingArgs[1:] {
		callArgs = append(callArgs, reflect.ValueOf(arg))
	}
	cl := node.CodeLocation
	node.Body = func() {
		out := callback.Call(callArgs)
		if len(out) == 1 && !out[0].IsNil() {
			fail(fmt.Sprintf("DeferCleanup callback returned error: %v", out[0]), cl)
		}
	}

	return node, nil
}

func (n Node) IsZero() bool {
	return n.ID == 0
}

/* Nodes */
type Nodes []Node

func (n Nodes) CopyAppend(nodes ...Node) Nodes {
	numN := len(n)
	out := make(Nodes, numN+len(nodes))
	for i, node := range n {
		out[i] = node
	}
	for j, node := range nodes {
		out[numN+j] = node
	}
	return out
}

func (n Nodes) SplitAround(pivot Node) (Nodes, Nodes) {
	pivotIdx := len(n)
	for i := range n {
		if n[i].ID == pivot.ID {
			pivotIdx = i
			break
		}
	}
	left := n[:pivotIdx]
	right := Nodes{}
	if pivotIdx+1 < len(n) {
		right = n[pivotIdx+1:]
	}

	return left, right
}

func (n Nodes) FirstNodeWithType(nodeTypes types.NodeType) Node {
	for i := range n {
		if n[i].NodeType.Is(nodeTypes) {
			return n[i]
		}
	}
	return Node{}
}

func (n Nodes) WithType(nodeTypes types.NodeType) Nodes {
	count := 0
	for i := range n {
		if n[i].NodeType.Is(nodeTypes) {
			count++
		}
	}

	out, j := make(Nodes, count), 0
	for i := range n {
		if n[i].NodeType.Is(nodeTypes) {
			out[j] = n[i]
			j++
		}
	}
	return out
}

func (n Nodes) WithoutType(nodeTypes types.NodeType) Nodes {
	count := 0
	for i := range n {
		if !n[i].NodeType.Is(nodeTypes) {
			count++
		}
	}

	out, j := make(Nodes, count), 0
	for i := range n {
		if !n[i].NodeType.Is(nodeTypes) {
			out[j] = n[i]
			j++
		}
	}
	return out
}

func (n Nodes) WithoutNode(nodeToExclude Node) Nodes {
	idxToExclude := len(n)
	for i := range n {
		if n[i].ID == nodeToExclude.ID {
			idxToExclude = i
			break
		}
	}
	if idxToExclude == len(n) {
		return n
	}
	out, j := make(Nodes, len(n)-1), 0
	for i := range n {
		if i == idxToExclude {
			continue
		}
		out[j] = n[i]
		j++
	}
	return out
}

func (n Nodes) Filter(filter func(Node) bool) Nodes {
	trufa, count := make([]bool, len(n)), 0
	for i := range n {
		if filter(n[i]) {
			trufa[i] = true
			count += 1
		}
	}
	out, j := make(Nodes, count), 0
	for i := range n {
		if trufa[i] {
			out[j] = n[i]
			j++
		}
	}
	return out
}

func (n Nodes) FirstSatisfying(filter func(Node) bool) Node {
	for i := range n {
		if filter(n[i]) {
			return n[i]
		}
	}
	return Node{}
}

func (n Nodes) WithinNestingLevel(deepestNestingLevel int) Nodes {
	count := 0
	for i := range n {
		if n[i].NestingLevel <= deepestNestingLevel {
			count++
		}
	}
	out, j := make(Nodes, count), 0
	for i := range n {
		if n[i].NestingLevel <= deepestNestingLevel {
			out[j] = n[i]
			j++
		}
	}
	return out
}

func (n Nodes) SortedByDescendingNestingLevel() Nodes {
	out := make(Nodes, len(n))
	copy(out, n)
	sort.SliceStable(out, func(i int, j int) bool {
		return out[i].NestingLevel > out[j].NestingLevel
	})

	return out
}

func (n Nodes) SortedByAscendingNestingLevel() Nodes {
	out := make(Nodes, len(n))
	copy(out, n)
	sort.SliceStable(out, func(i int, j int) bool {
		return out[i].NestingLevel < out[j].NestingLevel
	})

	return out
}

func (n Nodes) FirstWithNestingLevel(level int) Node {
	for i := range n {
		if n[i].NestingLevel == level {
			return n[i]
		}
	}
	return Node{}
}

func (n Nodes) Reverse() Nodes {
	out := make(Nodes, len(n))
	for i := range n {
		out[len(n)-1-i] = n[i]
	}
	return out
}

func (n Nodes) Texts() []string {
	out := make([]string, len(n))
	for i := range n {
		out[i] = n[i].Text
	}
	return out
}

func (n Nodes) Labels() [][]string {
	out := make([][]string, len(n))
	for i := range n {
		if n[i].Labels == nil {
			out[i] = []string{}
		} else {
			out[i] = []string(n[i].Labels)
		}
	}
	return out
}

func (n Nodes) UnionOfLabels() []string {
	out := []string{}
	seen := map[string]bool{}
	for i := range n {
		for _, label := range n[i].Labels {
			if !seen[label] {
				seen[label] = true
				out = append(out, label)
			}
		}
	}
	return out
}

func (n Nodes) CodeLocations() []types.CodeLocation {
	out := make([]types.CodeLocation, len(n))
	for i := range n {
		out[i] = n[i].CodeLocation
	}
	return out
}

func (n Nodes) BestTextFor(node Node) string {
	if node.Text != "" {
		return node.Text
	}
	parentNestingLevel := node.NestingLevel - 1
	for i := range n {
		if n[i].Text != "" && n[i].NestingLevel == parentNestingLevel {
			return n[i].Text
		}
	}

	return ""
}

func (n Nodes) ContainsNodeID(id uint) bool {
	for i := range n {
		if n[i].ID == id {
			return true
		}
	}
	return false
}

func (n Nodes) HasNodeMarkedPending() bool {
	for i := range n {
		if n[i].MarkedPending {
			return true
		}
	}
	return false
}

func (n Nodes) HasNodeMarkedFocus() bool {
	for i := range n {
		if n[i].MarkedFocus {
			return true
		}
	}
	return false
}

func (n Nodes) HasNodeMarkedSerial() bool {
	for i := range n {
		if n[i].MarkedSerial {
			return true
		}
	}
	return false
}

func (n Nodes) FirstNodeMarkedOrdered() Node {
	for i := range n {
		if n[i].MarkedOrdered {
			return n[i]
		}
	}
	return Node{}
}

func unrollInterfaceSlice(args interface{}) []interface{} {
	v := reflect.ValueOf(args)
	if v.Kind() != reflect.Slice {
		return []interface{}{args}
	}
	out := []interface{}{}
	for i := 0; i < v.Len(); i++ {
		el := reflect.ValueOf(v.Index(i).Interface())
		if el.Kind() == reflect.Slice && el.Type() != reflect.TypeOf(Labels{}) {
			out = append(out, unrollInterfaceSlice(el.Interface())...)
		} else {
			out = append(out, v.Index(i).Interface())
		}
	}
	return out
}
