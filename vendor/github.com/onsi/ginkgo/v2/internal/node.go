package internal

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"time"

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
	Body         func(SpecContext)
	CodeLocation types.CodeLocation
	NestingLevel int
	HasContext   bool

	SynchronizedBeforeSuiteProc1Body              func(SpecContext) []byte
	SynchronizedBeforeSuiteProc1BodyHasContext    bool
	SynchronizedBeforeSuiteAllProcsBody           func(SpecContext, []byte)
	SynchronizedBeforeSuiteAllProcsBodyHasContext bool

	SynchronizedAfterSuiteAllProcsBody           func(SpecContext)
	SynchronizedAfterSuiteAllProcsBodyHasContext bool
	SynchronizedAfterSuiteProc1Body              func(SpecContext)
	SynchronizedAfterSuiteProc1BodyHasContext    bool

	ReportEachBody  func(types.SpecReport)
	ReportSuiteBody func(types.Report)

	MarkedFocus             bool
	MarkedPending           bool
	MarkedSerial            bool
	MarkedOrdered           bool
	MarkedContinueOnFailure bool
	MarkedOncePerOrdered    bool
	FlakeAttempts           int
	MustPassRepeatedly      int
	Labels                  Labels
	PollProgressAfter       time.Duration
	PollProgressInterval    time.Duration
	NodeTimeout             time.Duration
	SpecTimeout             time.Duration
	GracePeriod             time.Duration

	NodeIDWhereCleanupWasGenerated uint
}

// Decoration Types
type focusType bool
type pendingType bool
type serialType bool
type orderedType bool
type continueOnFailureType bool
type honorsOrderedType bool
type suppressProgressReporting bool

const Focus = focusType(true)
const Pending = pendingType(true)
const Serial = serialType(true)
const Ordered = orderedType(true)
const ContinueOnFailure = continueOnFailureType(true)
const OncePerOrdered = honorsOrderedType(true)
const SuppressProgressReporting = suppressProgressReporting(true)

type FlakeAttempts uint
type MustPassRepeatedly uint
type Offset uint
type Done chan<- interface{} // Deprecated Done Channel for asynchronous testing
type Labels []string
type PollProgressInterval time.Duration
type PollProgressAfter time.Duration
type NodeTimeout time.Duration
type SpecTimeout time.Duration
type GracePeriod time.Duration

func (l Labels) MatchesLabelFilter(query string) bool {
	return types.MustParseLabelFilter(query)(l)
}

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
	case t == reflect.TypeOf(ContinueOnFailure):
		return true
	case t == reflect.TypeOf(OncePerOrdered):
		return true
	case t == reflect.TypeOf(SuppressProgressReporting):
		return true
	case t == reflect.TypeOf(FlakeAttempts(0)):
		return true
	case t == reflect.TypeOf(MustPassRepeatedly(0)):
		return true
	case t == reflect.TypeOf(Labels{}):
		return true
	case t == reflect.TypeOf(PollProgressInterval(0)):
		return true
	case t == reflect.TypeOf(PollProgressAfter(0)):
		return true
	case t == reflect.TypeOf(NodeTimeout(0)):
		return true
	case t == reflect.TypeOf(SpecTimeout(0)):
		return true
	case t == reflect.TypeOf(GracePeriod(0)):
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

var contextType = reflect.TypeOf(new(context.Context)).Elem()
var specContextType = reflect.TypeOf(new(SpecContext)).Elem()

func NewNode(deprecationTracker *types.DeprecationTracker, nodeType types.NodeType, text string, args ...interface{}) (Node, []error) {
	baseOffset := 2
	node := Node{
		ID:                   UniqueNodeID(),
		NodeType:             nodeType,
		Text:                 text,
		Labels:               Labels{},
		CodeLocation:         types.NewCodeLocation(baseOffset),
		NestingLevel:         -1,
		PollProgressAfter:    -1,
		PollProgressInterval: -1,
		GracePeriod:          -1,
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
		case t == reflect.TypeOf(ContinueOnFailure):
			node.MarkedContinueOnFailure = bool(arg.(continueOnFailureType))
			if !nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "ContinueOnFailure"))
			}
		case t == reflect.TypeOf(OncePerOrdered):
			node.MarkedOncePerOrdered = bool(arg.(honorsOrderedType))
			if !nodeType.Is(types.NodeTypeBeforeEach | types.NodeTypeJustBeforeEach | types.NodeTypeAfterEach | types.NodeTypeJustAfterEach) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "OncePerOrdered"))
			}
		case t == reflect.TypeOf(SuppressProgressReporting):
			deprecationTracker.TrackDeprecation(types.Deprecations.SuppressProgressReporting())
		case t == reflect.TypeOf(FlakeAttempts(0)):
			node.FlakeAttempts = int(arg.(FlakeAttempts))
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "FlakeAttempts"))
			}
		case t == reflect.TypeOf(MustPassRepeatedly(0)):
			node.MustPassRepeatedly = int(arg.(MustPassRepeatedly))
			if !nodeType.Is(types.NodeTypesForContainerAndIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "MustPassRepeatedly"))
			}
		case t == reflect.TypeOf(PollProgressAfter(0)):
			node.PollProgressAfter = time.Duration(arg.(PollProgressAfter))
			if nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "PollProgressAfter"))
			}
		case t == reflect.TypeOf(PollProgressInterval(0)):
			node.PollProgressInterval = time.Duration(arg.(PollProgressInterval))
			if nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "PollProgressInterval"))
			}
		case t == reflect.TypeOf(NodeTimeout(0)):
			node.NodeTimeout = time.Duration(arg.(NodeTimeout))
			if nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "NodeTimeout"))
			}
		case t == reflect.TypeOf(SpecTimeout(0)):
			node.SpecTimeout = time.Duration(arg.(SpecTimeout))
			if !nodeType.Is(types.NodeTypeIt) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "SpecTimeout"))
			}
		case t == reflect.TypeOf(GracePeriod(0)):
			node.GracePeriod = time.Duration(arg.(GracePeriod))
			if nodeType.Is(types.NodeTypeContainer) {
				appendError(types.GinkgoErrors.InvalidDecoratorForNodeType(node.CodeLocation, nodeType, "GracePeriod"))
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
			if nodeType.Is(types.NodeTypeContainer) {
				if node.Body != nil {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
				if t.NumOut() > 0 || t.NumIn() > 0 {
					appendError(types.GinkgoErrors.InvalidBodyTypeForContainer(t, node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
				body := arg.(func())
				node.Body = func(SpecContext) { body() }
			} else if nodeType.Is(types.NodeTypeReportBeforeEach | types.NodeTypeReportAfterEach) {
				if node.ReportEachBody == nil {
					node.ReportEachBody = arg.(func(types.SpecReport))
				} else {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
			} else if nodeType.Is(types.NodeTypeReportBeforeSuite | types.NodeTypeReportAfterSuite) {
				if node.ReportSuiteBody == nil {
					node.ReportSuiteBody = arg.(func(types.Report))
				} else {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
			} else if nodeType.Is(types.NodeTypeSynchronizedBeforeSuite) {
				if node.SynchronizedBeforeSuiteProc1Body != nil && node.SynchronizedBeforeSuiteAllProcsBody != nil {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
				if node.SynchronizedBeforeSuiteProc1Body == nil {
					body, hasContext := extractSynchronizedBeforeSuiteProc1Body(arg)
					if body == nil {
						appendError(types.GinkgoErrors.InvalidBodyTypeForSynchronizedBeforeSuiteProc1(t, node.CodeLocation))
						trackedFunctionError = true
					}
					node.SynchronizedBeforeSuiteProc1Body, node.SynchronizedBeforeSuiteProc1BodyHasContext = body, hasContext
				} else if node.SynchronizedBeforeSuiteAllProcsBody == nil {
					body, hasContext := extractSynchronizedBeforeSuiteAllProcsBody(arg)
					if body == nil {
						appendError(types.GinkgoErrors.InvalidBodyTypeForSynchronizedBeforeSuiteAllProcs(t, node.CodeLocation))
						trackedFunctionError = true
					}
					node.SynchronizedBeforeSuiteAllProcsBody, node.SynchronizedBeforeSuiteAllProcsBodyHasContext = body, hasContext
				}
			} else if nodeType.Is(types.NodeTypeSynchronizedAfterSuite) {
				if node.SynchronizedAfterSuiteAllProcsBody != nil && node.SynchronizedAfterSuiteProc1Body != nil {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
				body, hasContext := extractBodyFunction(deprecationTracker, node.CodeLocation, arg)
				if body == nil {
					appendError(types.GinkgoErrors.InvalidBodyType(t, node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
				if node.SynchronizedAfterSuiteAllProcsBody == nil {
					node.SynchronizedAfterSuiteAllProcsBody, node.SynchronizedAfterSuiteAllProcsBodyHasContext = body, hasContext
				} else if node.SynchronizedAfterSuiteProc1Body == nil {
					node.SynchronizedAfterSuiteProc1Body, node.SynchronizedAfterSuiteProc1BodyHasContext = body, hasContext
				}
			} else {
				if node.Body != nil {
					appendError(types.GinkgoErrors.MultipleBodyFunctions(node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
				node.Body, node.HasContext = extractBodyFunction(deprecationTracker, node.CodeLocation, arg)
				if node.Body == nil {
					appendError(types.GinkgoErrors.InvalidBodyType(t, node.CodeLocation, nodeType))
					trackedFunctionError = true
					break
				}
			}
		default:
			remainingArgs = append(remainingArgs, arg)
		}
	}

	//validations
	if node.MarkedPending && node.MarkedFocus {
		appendError(types.GinkgoErrors.InvalidDeclarationOfFocusedAndPending(node.CodeLocation, nodeType))
	}

	if node.MarkedContinueOnFailure && !node.MarkedOrdered {
		appendError(types.GinkgoErrors.InvalidContinueOnFailureDecoration(node.CodeLocation))
	}

	hasContext := node.HasContext || node.SynchronizedAfterSuiteProc1BodyHasContext || node.SynchronizedAfterSuiteAllProcsBodyHasContext || node.SynchronizedBeforeSuiteProc1BodyHasContext || node.SynchronizedBeforeSuiteAllProcsBodyHasContext

	if !hasContext && (node.NodeTimeout > 0 || node.SpecTimeout > 0 || node.GracePeriod > 0) && len(errors) == 0 {
		appendError(types.GinkgoErrors.InvalidTimeoutOrGracePeriodForNonContextNode(node.CodeLocation, nodeType))
	}

	if !node.NodeType.Is(types.NodeTypeReportBeforeEach|types.NodeTypeReportAfterEach|types.NodeTypeSynchronizedBeforeSuite|types.NodeTypeSynchronizedAfterSuite|types.NodeTypeReportBeforeSuite|types.NodeTypeReportAfterSuite) && node.Body == nil && !node.MarkedPending && !trackedFunctionError {
		appendError(types.GinkgoErrors.MissingBodyFunction(node.CodeLocation, nodeType))
	}

	if node.NodeType.Is(types.NodeTypeSynchronizedBeforeSuite) && !trackedFunctionError && (node.SynchronizedBeforeSuiteProc1Body == nil || node.SynchronizedBeforeSuiteAllProcsBody == nil) {
		appendError(types.GinkgoErrors.MissingBodyFunction(node.CodeLocation, nodeType))
	}

	if node.NodeType.Is(types.NodeTypeSynchronizedAfterSuite) && !trackedFunctionError && (node.SynchronizedAfterSuiteProc1Body == nil || node.SynchronizedAfterSuiteAllProcsBody == nil) {
		appendError(types.GinkgoErrors.MissingBodyFunction(node.CodeLocation, nodeType))
	}

	for _, arg := range remainingArgs {
		appendError(types.GinkgoErrors.UnknownDecorator(node.CodeLocation, nodeType, arg))
	}

	if node.FlakeAttempts > 0 && node.MustPassRepeatedly > 0 {
		appendError(types.GinkgoErrors.InvalidDeclarationOfFlakeAttemptsAndMustPassRepeatedly(node.CodeLocation, nodeType))
	}

	if len(errors) > 0 {
		return Node{}, errors
	}

	return node, errors
}

var doneType = reflect.TypeOf(make(Done))

func extractBodyFunction(deprecationTracker *types.DeprecationTracker, cl types.CodeLocation, arg interface{}) (func(SpecContext), bool) {
	t := reflect.TypeOf(arg)
	if t.NumOut() > 0 || t.NumIn() > 1 {
		return nil, false
	}
	if t.NumIn() == 1 {
		if t.In(0) == doneType {
			deprecationTracker.TrackDeprecation(types.Deprecations.Async(), cl)
			deprecatedAsyncBody := arg.(func(Done))
			return func(SpecContext) { deprecatedAsyncBody(make(Done)) }, false
		} else if t.In(0).Implements(specContextType) {
			return arg.(func(SpecContext)), true
		} else if t.In(0).Implements(contextType) {
			body := arg.(func(context.Context))
			return func(c SpecContext) { body(c) }, true
		}

		return nil, false
	}

	body := arg.(func())
	return func(SpecContext) { body() }, false
}

var byteType = reflect.TypeOf([]byte{})

func extractSynchronizedBeforeSuiteProc1Body(arg interface{}) (func(SpecContext) []byte, bool) {
	t := reflect.TypeOf(arg)
	v := reflect.ValueOf(arg)

	if t.NumOut() > 1 || t.NumIn() > 1 {
		return nil, false
	} else if t.NumOut() == 1 && t.Out(0) != byteType {
		return nil, false
	} else if t.NumIn() == 1 && !t.In(0).Implements(contextType) {
		return nil, false
	}
	hasContext := t.NumIn() == 1

	return func(c SpecContext) []byte {
		var out []reflect.Value
		if hasContext {
			out = v.Call([]reflect.Value{reflect.ValueOf(c)})
		} else {
			out = v.Call([]reflect.Value{})
		}
		if len(out) == 1 {
			return (out[0].Interface()).([]byte)
		} else {
			return []byte{}
		}
	}, hasContext
}

func extractSynchronizedBeforeSuiteAllProcsBody(arg interface{}) (func(SpecContext, []byte), bool) {
	t := reflect.TypeOf(arg)
	v := reflect.ValueOf(arg)
	hasContext, hasByte := false, false

	if t.NumOut() > 0 || t.NumIn() > 2 {
		return nil, false
	} else if t.NumIn() == 2 && t.In(0).Implements(contextType) && t.In(1) == byteType {
		hasContext, hasByte = true, true
	} else if t.NumIn() == 1 && t.In(0).Implements(contextType) {
		hasContext = true
	} else if t.NumIn() == 1 && t.In(0) == byteType {
		hasByte = true
	} else if t.NumIn() != 0 {
		return nil, false
	}

	return func(c SpecContext, b []byte) {
		in := []reflect.Value{}
		if hasContext {
			in = append(in, reflect.ValueOf(c))
		}
		if hasByte {
			in = append(in, reflect.ValueOf(b))
		}
		v.Call(in)
	}, hasContext
}

var errInterface = reflect.TypeOf((*error)(nil)).Elem()

func NewCleanupNode(deprecationTracker *types.DeprecationTracker, fail func(string, types.CodeLocation), args ...interface{}) (Node, []error) {
	decorations, remainingArgs := PartitionDecorations(args...)
	baseOffset := 2
	cl := types.NewCodeLocation(baseOffset)
	finalArgs := []interface{}{}
	for _, arg := range decorations {
		switch t := reflect.TypeOf(arg); {
		case t == reflect.TypeOf(Offset(0)):
			cl = types.NewCodeLocation(baseOffset + int(arg.(Offset)))
		case t == reflect.TypeOf(types.CodeLocation{}):
			cl = arg.(types.CodeLocation)
		default:
			finalArgs = append(finalArgs, arg)
		}
	}
	finalArgs = append(finalArgs, cl)

	if len(remainingArgs) == 0 {
		return Node{}, []error{types.GinkgoErrors.DeferCleanupInvalidFunction(cl)}
	}

	callback := reflect.ValueOf(remainingArgs[0])
	if !(callback.Kind() == reflect.Func) {
		return Node{}, []error{types.GinkgoErrors.DeferCleanupInvalidFunction(cl)}
	}

	callArgs := []reflect.Value{}
	for _, arg := range remainingArgs[1:] {
		callArgs = append(callArgs, reflect.ValueOf(arg))
	}

	hasContext := false
	t := callback.Type()
	if t.NumIn() > 0 {
		if t.In(0).Implements(specContextType) {
			hasContext = true
		} else if t.In(0).Implements(contextType) && (len(callArgs) == 0 || !callArgs[0].Type().Implements(contextType)) {
			hasContext = true
		}
	}

	handleFailure := func(out []reflect.Value) {
		if len(out) == 0 {
			return
		}
		last := out[len(out)-1]
		if last.Type().Implements(errInterface) && !last.IsNil() {
			fail(fmt.Sprintf("DeferCleanup callback returned error: %v", last), cl)
		}
	}

	if hasContext {
		finalArgs = append(finalArgs, func(c SpecContext) {
			out := callback.Call(append([]reflect.Value{reflect.ValueOf(c)}, callArgs...))
			handleFailure(out)
		})
	} else {
		finalArgs = append(finalArgs, func() {
			out := callback.Call(callArgs)
			handleFailure(out)
		})
	}

	return NewNode(deprecationTracker, types.NodeTypeCleanupInvalid, "", finalArgs...)
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

func (n Nodes) GetMaxFlakeAttempts() int {
	maxFlakeAttempts := 0
	for i := range n {
		if n[i].FlakeAttempts > 0 {
			maxFlakeAttempts = n[i].FlakeAttempts
		}
	}
	return maxFlakeAttempts
}

func (n Nodes) GetMaxMustPassRepeatedly() int {
	maxMustPassRepeatedly := 0
	for i := range n {
		if n[i].MustPassRepeatedly > 0 {
			maxMustPassRepeatedly = n[i].MustPassRepeatedly
		}
	}
	return maxMustPassRepeatedly
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
