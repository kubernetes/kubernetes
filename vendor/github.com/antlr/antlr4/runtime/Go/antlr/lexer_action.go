// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import "strconv"

const (
	LexerActionTypeChannel  = 0 //The type of a {@link LexerChannelAction} action.
	LexerActionTypeCustom   = 1 //The type of a {@link LexerCustomAction} action.
	LexerActionTypeMode     = 2 //The type of a {@link LexerModeAction} action.
	LexerActionTypeMore     = 3 //The type of a {@link LexerMoreAction} action.
	LexerActionTypePopMode  = 4 //The type of a {@link LexerPopModeAction} action.
	LexerActionTypePushMode = 5 //The type of a {@link LexerPushModeAction} action.
	LexerActionTypeSkip     = 6 //The type of a {@link LexerSkipAction} action.
	LexerActionTypeType     = 7 //The type of a {@link LexerTypeAction} action.
)

type LexerAction interface {
	getActionType() int
	getIsPositionDependent() bool
	execute(lexer Lexer)
	hash() int
	equals(other LexerAction) bool
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

func (b *BaseLexerAction) execute(lexer Lexer) {
	panic("Not implemented")
}

func (b *BaseLexerAction) getActionType() int {
	return b.actionType
}

func (b *BaseLexerAction) getIsPositionDependent() bool {
	return b.isPositionDependent
}

func (b *BaseLexerAction) hash() int {
	return b.actionType
}

func (b *BaseLexerAction) equals(other LexerAction) bool {
	return b == other
}

//
// Implements the {@code Skip} lexer action by calling {@link Lexer//Skip}.
//
// <p>The {@code Skip} command does not have any parameters, so l action is
// implemented as a singleton instance exposed by {@link //INSTANCE}.</p>
type LexerSkipAction struct {
	*BaseLexerAction
}

func NewLexerSkipAction() *LexerSkipAction {
	la := new(LexerSkipAction)
	la.BaseLexerAction = NewBaseLexerAction(LexerActionTypeSkip)
	return la
}

// Provides a singleton instance of l parameterless lexer action.
var LexerSkipActionINSTANCE = NewLexerSkipAction()

func (l *LexerSkipAction) execute(lexer Lexer) {
	lexer.Skip()
}

func (l *LexerSkipAction) String() string {
	return "skip"
}

//  Implements the {@code type} lexer action by calling {@link Lexer//setType}
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

func (l *LexerTypeAction) hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.thetype)
	return murmurFinish(h, 2)
}

func (l *LexerTypeAction) equals(other LexerAction) bool {
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

// Implements the {@code pushMode} lexer action by calling
// {@link Lexer//pushMode} with the assigned mode.
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

func (l *LexerPushModeAction) hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.mode)
	return murmurFinish(h, 2)
}

func (l *LexerPushModeAction) equals(other LexerAction) bool {
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

// Implements the {@code popMode} lexer action by calling {@link Lexer//popMode}.
//
// <p>The {@code popMode} command does not have any parameters, so l action is
// implemented as a singleton instance exposed by {@link //INSTANCE}.</p>
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

// Implements the {@code mode} lexer action by calling {@link Lexer//mode} with
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

func (l *LexerModeAction) hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.mode)
	return murmurFinish(h, 2)
}

func (l *LexerModeAction) equals(other LexerAction) bool {
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

func (l *LexerCustomAction) hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.ruleIndex)
	h = murmurUpdate(h, l.actionIndex)
	return murmurFinish(h, 3)
}

func (l *LexerCustomAction) equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerCustomAction); !ok {
		return false
	} else {
		return l.ruleIndex == other.(*LexerCustomAction).ruleIndex && l.actionIndex == other.(*LexerCustomAction).actionIndex
	}
}

// Implements the {@code channel} lexer action by calling
// {@link Lexer//setChannel} with the assigned channel.
// Constructs a New{@code channel} action with the specified channel value.
// @param channel The channel value to pass to {@link Lexer//setChannel}.
type LexerChannelAction struct {
	*BaseLexerAction

	channel int
}

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

func (l *LexerChannelAction) hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.actionType)
	h = murmurUpdate(h, l.channel)
	return murmurFinish(h, 2)
}

func (l *LexerChannelAction) equals(other LexerAction) bool {
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

// Constructs a Newindexed custom action by associating a character offset
// with a {@link LexerAction}.
//
// <p>Note: This class is only required for lexer actions for which
// {@link LexerAction//isPositionDependent} returns {@code true}.</p>
//
// @param offset The offset into the input {@link CharStream}, relative to
// the token start index, at which the specified lexer action should be
// executed.
// @param action The lexer action to execute at a particular offset in the
// input {@link CharStream}.
type LexerIndexedCustomAction struct {
	*BaseLexerAction

	offset              int
	lexerAction         LexerAction
	isPositionDependent bool
}

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

func (l *LexerIndexedCustomAction) hash() int {
	h := murmurInit(0)
	h = murmurUpdate(h, l.offset)
	h = murmurUpdate(h, l.lexerAction.hash())
	return murmurFinish(h, 2)
}

func (l *LexerIndexedCustomAction) equals(other LexerAction) bool {
	if l == other {
		return true
	} else if _, ok := other.(*LexerIndexedCustomAction); !ok {
		return false
	} else {
		return l.offset == other.(*LexerIndexedCustomAction).offset && l.lexerAction == other.(*LexerIndexedCustomAction).lexerAction
	}
}
