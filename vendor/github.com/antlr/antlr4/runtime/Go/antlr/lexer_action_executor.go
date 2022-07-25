// Copyright (c) 2012-2017 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

// Represents an executor for a sequence of lexer actions which traversed during
// the Matching operation of a lexer rule (token).
//
// <p>The executor tracks position information for position-dependent lexer actions
// efficiently, ensuring that actions appearing only at the end of the rule do
// not cause bloating of the {@link DFA} created for the lexer.</p>

type LexerActionExecutor struct {
	lexerActions     []LexerAction
	cachedHash       int
}

func NewLexerActionExecutor(lexerActions []LexerAction) *LexerActionExecutor {

	if lexerActions == nil {
		lexerActions = make([]LexerAction, 0)
	}

	l := new(LexerActionExecutor)

	l.lexerActions = lexerActions

	// Caches the result of {@link //hashCode} since the hash code is an element
	// of the performance-critical {@link LexerATNConfig//hashCode} operation.
	l.cachedHash = murmurInit(57)
	for _, a := range lexerActions {
		l.cachedHash = murmurUpdate(l.cachedHash, a.hash())
	}

	return l
}

// Creates a {@link LexerActionExecutor} which executes the actions for
// the input {@code lexerActionExecutor} followed by a specified
// {@code lexerAction}.
//
// @param lexerActionExecutor The executor for actions already traversed by
// the lexer while Matching a token within a particular
// {@link LexerATNConfig}. If this is {@code nil}, the method behaves as
// though it were an empty executor.
// @param lexerAction The lexer action to execute after the actions
// specified in {@code lexerActionExecutor}.
//
// @return A {@link LexerActionExecutor} for executing the combine actions
// of {@code lexerActionExecutor} and {@code lexerAction}.
func LexerActionExecutorappend(lexerActionExecutor *LexerActionExecutor, lexerAction LexerAction) *LexerActionExecutor {
	if lexerActionExecutor == nil {
		return NewLexerActionExecutor([]LexerAction{lexerAction})
	}

	return NewLexerActionExecutor(append(lexerActionExecutor.lexerActions, lexerAction))
}

// Creates a {@link LexerActionExecutor} which encodes the current offset
// for position-dependent lexer actions.
//
// <p>Normally, when the executor encounters lexer actions where
// {@link LexerAction//isPositionDependent} returns {@code true}, it calls
// {@link IntStream//seek} on the input {@link CharStream} to set the input
// position to the <em>end</em> of the current token. This behavior provides
// for efficient DFA representation of lexer actions which appear at the end
// of a lexer rule, even when the lexer rule Matches a variable number of
// characters.</p>
//
// <p>Prior to traversing a Match transition in the ATN, the current offset
// from the token start index is assigned to all position-dependent lexer
// actions which have not already been assigned a fixed offset. By storing
// the offsets relative to the token start index, the DFA representation of
// lexer actions which appear in the middle of tokens remains efficient due
// to sharing among tokens of the same length, regardless of their absolute
// position in the input stream.</p>
//
// <p>If the current executor already has offsets assigned to all
// position-dependent lexer actions, the method returns {@code this}.</p>
//
// @param offset The current offset to assign to all position-dependent
// lexer actions which do not already have offsets assigned.
//
// @return A {@link LexerActionExecutor} which stores input stream offsets
// for all position-dependent lexer actions.
// /
func (l *LexerActionExecutor) fixOffsetBeforeMatch(offset int) *LexerActionExecutor {
	var updatedLexerActions []LexerAction
	for i := 0; i < len(l.lexerActions); i++ {
		_, ok := l.lexerActions[i].(*LexerIndexedCustomAction)
		if l.lexerActions[i].getIsPositionDependent() && !ok {
			if updatedLexerActions == nil {
				updatedLexerActions = make([]LexerAction, 0)

				for _, a := range l.lexerActions {
					updatedLexerActions = append(updatedLexerActions, a)
				}
			}

			updatedLexerActions[i] = NewLexerIndexedCustomAction(offset, l.lexerActions[i])
		}
	}
	if updatedLexerActions == nil {
		return l
	}

	return NewLexerActionExecutor(updatedLexerActions)
}

// Execute the actions encapsulated by l executor within the context of a
// particular {@link Lexer}.
//
// <p>This method calls {@link IntStream//seek} to set the position of the
// {@code input} {@link CharStream} prior to calling
// {@link LexerAction//execute} on a position-dependent action. Before the
// method returns, the input position will be restored to the same position
// it was in when the method was invoked.</p>
//
// @param lexer The lexer instance.
// @param input The input stream which is the source for the current token.
// When l method is called, the current {@link IntStream//index} for
// {@code input} should be the start of the following token, i.e. 1
// character past the end of the current token.
// @param startIndex The token start index. This value may be passed to
// {@link IntStream//seek} to set the {@code input} position to the beginning
// of the token.
// /
func (l *LexerActionExecutor) execute(lexer Lexer, input CharStream, startIndex int) {
	requiresSeek := false
	stopIndex := input.Index()

	defer func() {
		if requiresSeek {
			input.Seek(stopIndex)
		}
	}()

	for i := 0; i < len(l.lexerActions); i++ {
		lexerAction := l.lexerActions[i]
		if la, ok := lexerAction.(*LexerIndexedCustomAction); ok {
			offset := la.offset
			input.Seek(startIndex + offset)
			lexerAction = la.lexerAction
			requiresSeek = (startIndex + offset) != stopIndex
		} else if lexerAction.getIsPositionDependent() {
			input.Seek(stopIndex)
			requiresSeek = false
		}
		lexerAction.execute(lexer)
	}
}

func (l *LexerActionExecutor) hash() int {
	if l == nil {
		return 61
	}
	return l.cachedHash
}

func (l *LexerActionExecutor) equals(other interface{}) bool {
	if l == other {
		return true
	}
	othert, ok := other.(*LexerActionExecutor)
	if !ok {
		return false
	}
	if othert == nil {
		return false
	}
	return l.cachedHash == othert.cachedHash && &l.lexerActions == &othert.lexerActions
}
