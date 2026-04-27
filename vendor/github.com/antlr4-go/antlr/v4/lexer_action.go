// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import "strconv"

const (
	// LexerActionTypeChannel represents a [LexerChannelAction] action.
	LexerActionTypeChannel = 0

	// LexerActionTypeCustom represents a [LexerCustomAction] action.
	LexerActionTypeCustom = 1

	// LexerActionTypeMode represents a [LexerModeAction] action.
	LexerActionTypeMode = 2

	// LexerActionTypeMore represents a [LexerMoreAction] action.
	LexerActionTypeMore = 3

	// LexerActionTypePopMode represents a [LexerPopModeAction] action.
	LexerActionTypePopMode = 4

	// LexerActionTypePushMode represents a [LexerPushModeAction] action.
	LexerActionTypePushMode = 5

	// LexerActionTypeSkip represents a [LexerSkipAction] action.
	LexerActionTypeSkip = 6

	// LexerActionTypeType represents a [LexerTypeAction] action.
	LexerActionTypeType = 7
)

type LexerAction interface {
	getActionType() int
	getIsPositionDependent() bool
	execute(lexer Lexer)
	Hash() int
	Equals(other LexerAction) bool
}

type BaseLexerAction struct {
	actionType          int
	isPositionDependent bool
}

func NewBaseLexerAction(action int) *BaseLexerAction {
	la := new(BaseLexerAction)

	la.actionType = action
	la.isPositionDependent = false

	return la
}

func (b *BaseLexerAction) execute(_ Lexer) {
	panic("Not implemented")
}

func (b *BaseLexerAction) getActionType() int {
	return b.actionType
}

func (b *BaseLexerAction) getIsPositionDependent() bool {
	return b.isPositionDependent
}

func (b *BaseLexerAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, b.actionType)
	return murmurFinish(h, 1)
}

func (b *BaseLexerAction) Equals(other LexerAction) bool {
	return b.actionType == other.getActionType()
}

// LexerSkipAction implements the [BaseLexerAction.Skip] lexer action by calling [Lexer.Skip].
//
// The Skip command does not have any parameters, so this action is
// implemented as a singleton instance exposed by the [LexerSkipActionINSTANCE].
type LexerSkipAction struct {
	*BaseLexerAction
}

func NewLexerSkipAction() *LexerSkipAction {
	la := new(LexerSkipAction)
	la.BaseLexerAction = NewBaseLexerAction(LexerActionTypeSkip)
	return la
}

// LexerSkipActionINSTANCE provides a singleton instance of this parameterless lexer action.
var LexerSkipActionINSTANCE = NewLexerSkipAction()

func (l *LexerSkipAction) execute(lexer Lexer) {
	lexer.Skip()
}

// String returns a string representation of the current [LexerSkipAction].
func (l *LexerSkipAction) String() string {
	return "skip"
}

func (b *LexerSkipAction) Equals(other LexerAction) bool {
	return other.getActionType() == LexerActionTypeSkip
}

//	Implements the {@code type} lexer action by calling {@link Lexer//setType}
//
// with the assigned type.
type LexerTypeAction struct {
	*BaseLexerAction

	thetype int
}

func NewLexerTypeAction(thetype int) *LexerTypeAction {
	l := new(LexerTypeAction)
	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypeType)
	l.thetype = thetype
	return l
}

func (l *LexerTypeAction) execute(lexer Lexer) {
	lexer.SetType(l.thetype)
}

func (l *LexerTypeAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.thetype)
	return murmurFinish(h, 2)
}

func (l *LexerTypeAction) Equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerTypeAction); !ok {
		return false
	} else {
		return l.thetype == other.(*LexerTypeAction).thetype
	}
}

func (l *LexerTypeAction) String() string {
	return "actionType(" + strconv.Itoa(l.thetype) + ")"
}

// LexerPushModeAction implements the pushMode lexer action by calling
// [Lexer.pushMode] with the assigned mode.
type LexerPushModeAction struct {
	*BaseLexerAction
	mode int
}

func NewLexerPushModeAction(mode int) *LexerPushModeAction {

	l := new(LexerPushModeAction)
	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypePushMode)

	l.mode = mode
	return l
}

// <p>This action is implemented by calling {@link Lexer//pushMode} with the
// value provided by {@link //getMode}.</p>
func (l *LexerPushModeAction) execute(lexer Lexer) {
	lexer.PushMode(l.mode)
}

func (l *LexerPushModeAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.mode)
	return murmurFinish(h, 2)
}

func (l *LexerPushModeAction) Equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerPushModeAction); !ok {
		return false
	} else {
		return l.mode == other.(*LexerPushModeAction).mode
	}
}

func (l *LexerPushModeAction) String() string {
	return "pushMode(" + strconv.Itoa(l.mode) + ")"
}

// LexerPopModeAction implements the popMode lexer action by calling [Lexer.popMode].
//
// The popMode command does not have any parameters, so this action is
// implemented as a singleton instance exposed by [LexerPopModeActionINSTANCE]
type LexerPopModeAction struct {
	*BaseLexerAction
}

func NewLexerPopModeAction() *LexerPopModeAction {

	l := new(LexerPopModeAction)

	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypePopMode)

	return l
}

var LexerPopModeActionINSTANCE = NewLexerPopModeAction()

// <p>This action is implemented by calling {@link Lexer//popMode}.</p>
func (l *LexerPopModeAction) execute(lexer Lexer) {
	lexer.PopMode()
}

func (l *LexerPopModeAction) String() string {
	return "popMode"
}

// Implements the {@code more} lexer action by calling {@link Lexer//more}.
//
// <p>The {@code more} command does not have any parameters, so l action is
// implemented as a singleton instance exposed by {@link //INSTANCE}.</p>

type LexerMoreAction struct {
	*BaseLexerAction
}

func NewLexerMoreAction() *LexerMoreAction {
	l := new(LexerMoreAction)
	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypeMore)

	return l
}

var LexerMoreActionINSTANCE = NewLexerMoreAction()

// <p>This action is implemented by calling {@link Lexer//popMode}.</p>
func (l *LexerMoreAction) execute(lexer Lexer) {
	lexer.More()
}

func (l *LexerMoreAction) String() string {
	return "more"
}

// LexerModeAction implements the mode lexer action by calling [Lexer.mode] with
// the assigned mode.
type LexerModeAction struct {
	*BaseLexerAction
	mode int
}

func NewLexerModeAction(mode int) *LexerModeAction {
	l := new(LexerModeAction)
	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypeMode)
	l.mode = mode
	return l
}

// <p>This action is implemented by calling {@link Lexer//mode} with the
// value provided by {@link //getMode}.</p>
func (l *LexerModeAction) execute(lexer Lexer) {
	lexer.SetMode(l.mode)
}

func (l *LexerModeAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.mode)
	return murmurFinish(h, 2)
}

func (l *LexerModeAction) Equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerModeAction); !ok {
		return false
	} else {
		return l.mode == other.(*LexerModeAction).mode
	}
}

func (l *LexerModeAction) String() string {
	return "mode(" + strconv.Itoa(l.mode) + ")"
}

// Executes a custom lexer action by calling {@link Recognizer//action} with the
// rule and action indexes assigned to the custom action. The implementation of
// a custom action is added to the generated code for the lexer in an override
// of {@link Recognizer//action} when the grammar is compiled.
//
// <p>This class may represent embedded actions created with the <code>{...}</code>
// syntax in ANTLR 4, as well as actions created for lexer commands where the
// command argument could not be evaluated when the grammar was compiled.</p>

// Constructs a custom lexer action with the specified rule and action
// indexes.
//
// @param ruleIndex The rule index to use for calls to
// {@link Recognizer//action}.
// @param actionIndex The action index to use for calls to
// {@link Recognizer//action}.

type LexerCustomAction struct {
	*BaseLexerAction
	ruleIndex, actionIndex int
}

func NewLexerCustomAction(ruleIndex, actionIndex int) *LexerCustomAction {
	l := new(LexerCustomAction)
	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypeCustom)
	l.ruleIndex = ruleIndex
	l.actionIndex = actionIndex
	l.isPositionDependent = true
	return l
}

// <p>Custom actions are implemented by calling {@link Lexer//action} with the
// appropriate rule and action indexes.</p>
func (l *LexerCustomAction) execute(lexer Lexer) {
	lexer.Action(nil, l.ruleIndex, l.actionIndex)
}

func (l *LexerCustomAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.ruleIndex)
	h = murmurUpdate(h, l.actionIndex)
	return murmurFinish(h, 3)
}

func (l *LexerCustomAction) Equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerCustomAction); !ok {
		return false
	} else {
		return l.ruleIndex == other.(*LexerCustomAction).ruleIndex &&
			l.actionIndex == other.(*LexerCustomAction).actionIndex
	}
}

// LexerChannelAction implements the channel lexer action by calling
// [Lexer.setChannel] with the assigned channel.
//
// Constructs a new channel action with the specified channel value.
type LexerChannelAction struct {
	*BaseLexerAction
	channel int
}

// NewLexerChannelAction creates a channel lexer action by calling
// [Lexer.setChannel] with the assigned channel.
//
// Constructs a new channel action with the specified channel value.
func NewLexerChannelAction(channel int) *LexerChannelAction {
	l := new(LexerChannelAction)
	l.BaseLexerAction = NewBaseLexerAction(LexerActionTypeChannel)
	l.channel = channel
	return l
}

// <p>This action is implemented by calling {@link Lexer//setChannel} with the
// value provided by {@link //getChannel}.</p>
func (l *LexerChannelAction) execute(lexer Lexer) {
	lexer.SetChannel(l.channel)
}

func (l *LexerChannelAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.channel)
	return murmurFinish(h, 2)
}

func (l *LexerChannelAction) Equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerChannelAction); !ok {
		return false
	} else {
		return l.channel == other.(*LexerChannelAction).channel
	}
}

func (l *LexerChannelAction) String() string {
	return "channel(" + strconv.Itoa(l.channel) + ")"
}

// This implementation of {@link LexerAction} is used for tracking input offsets
// for position-dependent actions within a {@link LexerActionExecutor}.
//
// <p>This action is not serialized as part of the ATN, and is only required for
// position-dependent lexer actions which appear at a location other than the
// end of a rule. For more information about DFA optimizations employed for
// lexer actions, see {@link LexerActionExecutor//append} and
// {@link LexerActionExecutor//fixOffsetBeforeMatch}.</p>

type LexerIndexedCustomAction struct {
	*BaseLexerAction
	offset              int
	lexerAction         LexerAction
	isPositionDependent bool
}

// NewLexerIndexedCustomAction constructs a new indexed custom action by associating a character offset
// with a [LexerAction].
//
// Note: This class is only required for lexer actions for which
// [LexerAction.isPositionDependent] returns true.
//
// The offset points into the input [CharStream], relative to
// the token start index, at which the specified lexerAction should be
// executed.
func NewLexerIndexedCustomAction(offset int, lexerAction LexerAction) *LexerIndexedCustomAction {

	l := new(LexerIndexedCustomAction)
	l.BaseLexerAction = NewBaseLexerAction(lexerAction.getActionType())

	l.offset = offset
	l.lexerAction = lexerAction
	l.isPositionDependent = true

	return l
}

// <p>This method calls {@link //execute} on the result of {@link //getAction}
// using the provided {@code lexer}.</p>
func (l *LexerIndexedCustomAction) execute(lexer Lexer) {
	// assume the input stream position was properly set by the calling code
	l.lexerAction.execute(lexer)
}

func (l *LexerIndexedCustomAction) Hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.offset)
	h = murmurUpdate(h, l.lexerAction.Hash())
	return murmurFinish(h, 2)
}

func (l *LexerIndexedCustomAction) equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerIndexedCustomAction); !ok {
		return false
	} else {
		return l.offset == other.(*LexerIndexedCustomAction).offset &&
			l.lexerAction.Equals(other.(*LexerIndexedCustomAction).lexerAction)
	}
}
