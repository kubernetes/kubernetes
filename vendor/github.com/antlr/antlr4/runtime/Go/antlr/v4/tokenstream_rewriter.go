// Copyright (c) 2012-2022 The ANTLR Project. All rights reserved.
// Use of this file is governed by the BSD 3-clause license that
// can be found in the LICENSE.txt file in the project root.

package antlr

import (
	"bytes"
	"fmt"
)

//
// Useful for rewriting out a buffered input token stream after doing some
// augmentation or other manipulations on it.

// <p>
// You can insert stuff, replace, and delete chunks. Note that the operations
// are done lazily--only if you convert the buffer to a {@link String} with
// {@link TokenStream#getText()}. This is very efficient because you are not
// moving data around all the time. As the buffer of tokens is converted to
// strings, the {@link #getText()} method(s) scan the input token stream and
// check to see if there is an operation at the current index. If so, the
// operation is done and then normal {@link String} rendering continues on the
// buffer. This is like having multiple Turing machine instruction streams
// (programs) operating on a single input tape. :)</p>
// <p>

// This rewriter makes no modifications to the token stream. It does not ask the
// stream to fill itself up nor does it advance the input cursor. The token
// stream {@link TokenStream#index()} will return the same value before and
// after any {@link #getText()} call.</p>

// <p>
// The rewriter only works on tokens that you have in the buffer and ignores the
// current input cursor. If you are buffering tokens on-demand, calling
// {@link #getText()} halfway through the input will only do rewrites for those
// tokens in the first half of the file.</p>

// <p>
// Since the operations are done lazily at {@link #getText}-time, operations do
// not screw up the token index values. That is, an insert operation at token
// index {@code i} does not change the index values for tokens
// {@code i}+1..n-1.</p>

// <p>
// Because operations never actually alter the buffer, you may always get the
// original token stream back without undoing anything. Since the instructions
// are queued up, you can easily simulate transactions and roll back any changes
// if there is an error just by removing instructions. For example,</p>

// <pre>
// CharStream input = new ANTLRFileStream("input");
// TLexer lex = new TLexer(input);
// CommonTokenStream tokens = new CommonTokenStream(lex);
// T parser = new T(tokens);
// TokenStreamRewriter rewriter = new TokenStreamRewriter(tokens);
// parser.startRule();
// </pre>

// <p>
// Then in the rules, you can execute (assuming rewriter is visible):</p>

// <pre>
// Token t,u;
// ...
// rewriter.insertAfter(t, "text to put after t");}
// rewriter.insertAfter(u, "text after u");}
// System.out.println(rewriter.getText());
// </pre>

// <p>
// You can also have multiple "instruction streams" and get multiple rewrites
// from a single pass over the input. Just name the instruction streams and use
// that name again when printing the buffer. This could be useful for generating
// a C file and also its header file--all from the same buffer:</p>

// <pre>
// rewriter.insertAfter("pass1", t, "text to put after t");}
// rewriter.insertAfter("pass2", u, "text after u");}
// System.out.println(rewriter.getText("pass1"));
// System.out.println(rewriter.getText("pass2"));
// </pre>

// <p>
// If you don't use named rewrite streams, a "default" stream is used as the
// first example shows.</p>

const (
	Default_Program_Name = "default"
	Program_Init_Size    = 100
	Min_Token_Index      = 0
)

// Define the rewrite operation hierarchy

type RewriteOperation interface {
	// Execute the rewrite operation by possibly adding to the buffer.
	// Return the index of the next token to operate on.
	Execute(buffer *bytes.Buffer) int
	String() string
	GetInstructionIndex() int
	GetIndex() int
	GetText() string
	GetOpName() string
	GetTokens() TokenStream
	SetInstructionIndex(val int)
	SetIndex(int)
	SetText(string)
	SetOpName(string)
	SetTokens(TokenStream)
}

type BaseRewriteOperation struct {
	//Current index of rewrites list
	instruction_index int
	//Token buffer index
	index int
	//Substitution text
	text string
	//Actual operation name
	op_name string
	//Pointer to token steam
	tokens TokenStream
}

func (op *BaseRewriteOperation) GetInstructionIndex() int {
	return op.instruction_index
}

func (op *BaseRewriteOperation) GetIndex() int {
	return op.index
}

func (op *BaseRewriteOperation) GetText() string {
	return op.text
}

func (op *BaseRewriteOperation) GetOpName() string {
	return op.op_name
}

func (op *BaseRewriteOperation) GetTokens() TokenStream {
	return op.tokens
}

func (op *BaseRewriteOperation) SetInstructionIndex(val int) {
	op.instruction_index = val
}

func (op *BaseRewriteOperation) SetIndex(val int) {
	op.index = val
}

func (op *BaseRewriteOperation) SetText(val string) {
	op.text = val
}

func (op *BaseRewriteOperation) SetOpName(val string) {
	op.op_name = val
}

func (op *BaseRewriteOperation) SetTokens(val TokenStream) {
	op.tokens = val
}

func (op *BaseRewriteOperation) Execute(buffer *bytes.Buffer) int {
	return op.index
}

func (op *BaseRewriteOperation) String() string {
	return fmt.Sprintf("<%s@%d:\"%s\">",
		op.op_name,
		op.tokens.Get(op.GetIndex()),
		op.text,
	)

}

type InsertBeforeOp struct {
	BaseRewriteOperation
}

func NewInsertBeforeOp(index int, text string, stream TokenStream) *InsertBeforeOp {
	return &InsertBeforeOp{BaseRewriteOperation: BaseRewriteOperation{
		index:   index,
		text:    text,
		op_name: "InsertBeforeOp",
		tokens:  stream,
	}}
}

func (op *InsertBeforeOp) Execute(buffer *bytes.Buffer) int {
	buffer.WriteString(op.text)
	if op.tokens.Get(op.index).GetTokenType() != TokenEOF {
		buffer.WriteString(op.tokens.Get(op.index).GetText())
	}
	return op.index + 1
}

func (op *InsertBeforeOp) String() string {
	return op.BaseRewriteOperation.String()
}

// Distinguish between insert after/before to do the "insert afters"
//  first and then the "insert befores" at same index. Implementation
//  of "insert after" is "insert before index+1".

type InsertAfterOp struct {
	BaseRewriteOperation
}

func NewInsertAfterOp(index int, text string, stream TokenStream) *InsertAfterOp {
	return &InsertAfterOp{BaseRewriteOperation: BaseRewriteOperation{
		index:  index + 1,
		text:   text,
		tokens: stream,
	}}
}

func (op *InsertAfterOp) Execute(buffer *bytes.Buffer) int {
	buffer.WriteString(op.text)
	if op.tokens.Get(op.index).GetTokenType() != TokenEOF {
		buffer.WriteString(op.tokens.Get(op.index).GetText())
	}
	return op.index + 1
}

func (op *InsertAfterOp) String() string {
	return op.BaseRewriteOperation.String()
}

// I'm going to try replacing range from x..y with (y-x)+1 ReplaceOp
// instructions.
type ReplaceOp struct {
	BaseRewriteOperation
	LastIndex int
}

func NewReplaceOp(from, to int, text string, stream TokenStream) *ReplaceOp {
	return &ReplaceOp{
		BaseRewriteOperation: BaseRewriteOperation{
			index:   from,
			text:    text,
			op_name: "ReplaceOp",
			tokens:  stream,
		},
		LastIndex: to,
	}
}

func (op *ReplaceOp) Execute(buffer *bytes.Buffer) int {
	if op.text != "" {
		buffer.WriteString(op.text)
	}
	return op.LastIndex + 1
}

func (op *ReplaceOp) String() string {
	if op.text == "" {
		return fmt.Sprintf("<DeleteOP@%d..%d>",
			op.tokens.Get(op.index), op.tokens.Get(op.LastIndex))
	}
	return fmt.Sprintf("<ReplaceOp@%d..%d:\"%s\">",
		op.tokens.Get(op.index), op.tokens.Get(op.LastIndex), op.text)
}

type TokenStreamRewriter struct {
	//Our source stream
	tokens TokenStream
	// You may have multiple, named streams of rewrite operations.
	//  I'm calling these things "programs."
	//  Maps String (name) &rarr; rewrite (List)
	programs                   map[string][]RewriteOperation
	last_rewrite_token_indexes map[string]int
}

func NewTokenStreamRewriter(tokens TokenStream) *TokenStreamRewriter {
	return &TokenStreamRewriter{
		tokens: tokens,
		programs: map[string][]RewriteOperation{
			Default_Program_Name: make([]RewriteOperation, 0, Program_Init_Size),
		},
		last_rewrite_token_indexes: map[string]int{},
	}
}

func (tsr *TokenStreamRewriter) GetTokenStream() TokenStream {
	return tsr.tokens
}

// Rollback the instruction stream for a program so that
// the indicated instruction (via instructionIndex) is no
// longer in the stream. UNTESTED!
func (tsr *TokenStreamRewriter) Rollback(program_name string, instruction_index int) {
	is, ok := tsr.programs[program_name]
	if ok {
		tsr.programs[program_name] = is[Min_Token_Index:instruction_index]
	}
}

func (tsr *TokenStreamRewriter) RollbackDefault(instruction_index int) {
	tsr.Rollback(Default_Program_Name, instruction_index)
}

// Reset the program so that no instructions exist
func (tsr *TokenStreamRewriter) DeleteProgram(program_name string) {
	tsr.Rollback(program_name, Min_Token_Index) //TODO: double test on that cause lower bound is not included
}

func (tsr *TokenStreamRewriter) DeleteProgramDefault() {
	tsr.DeleteProgram(Default_Program_Name)
}

func (tsr *TokenStreamRewriter) InsertAfter(program_name string, index int, text string) {
	// to insert after, just insert before next index (even if past end)
	var op RewriteOperation = NewInsertAfterOp(index, text, tsr.tokens)
	rewrites := tsr.GetProgram(program_name)
	op.SetInstructionIndex(len(rewrites))
	tsr.AddToProgram(program_name, op)
}

func (tsr *TokenStreamRewriter) InsertAfterDefault(index int, text string) {
	tsr.InsertAfter(Default_Program_Name, index, text)
}

func (tsr *TokenStreamRewriter) InsertAfterToken(program_name string, token Token, text string) {
	tsr.InsertAfter(program_name, token.GetTokenIndex(), text)
}

func (tsr *TokenStreamRewriter) InsertBefore(program_name string, index int, text string) {
	var op RewriteOperation = NewInsertBeforeOp(index, text, tsr.tokens)
	rewrites := tsr.GetProgram(program_name)
	op.SetInstructionIndex(len(rewrites))
	tsr.AddToProgram(program_name, op)
}

func (tsr *TokenStreamRewriter) InsertBeforeDefault(index int, text string) {
	tsr.InsertBefore(Default_Program_Name, index, text)
}

func (tsr *TokenStreamRewriter) InsertBeforeToken(program_name string, token Token, text string) {
	tsr.InsertBefore(program_name, token.GetTokenIndex(), text)
}

func (tsr *TokenStreamRewriter) Replace(program_name string, from, to int, text string) {
	if from > to || from < 0 || to < 0 || to >= tsr.tokens.Size() {
		panic(fmt.Sprintf("replace: range invalid: %d..%d(size=%d)",
			from, to, tsr.tokens.Size()))
	}
	var op RewriteOperation = NewReplaceOp(from, to, text, tsr.tokens)
	rewrites := tsr.GetProgram(program_name)
	op.SetInstructionIndex(len(rewrites))
	tsr.AddToProgram(program_name, op)
}

func (tsr *TokenStreamRewriter) ReplaceDefault(from, to int, text string) {
	tsr.Replace(Default_Program_Name, from, to, text)
}

func (tsr *TokenStreamRewriter) ReplaceDefaultPos(index int, text string) {
	tsr.ReplaceDefault(index, index, text)
}

func (tsr *TokenStreamRewriter) ReplaceToken(program_name string, from, to Token, text string) {
	tsr.Replace(program_name, from.GetTokenIndex(), to.GetTokenIndex(), text)
}

func (tsr *TokenStreamRewriter) ReplaceTokenDefault(from, to Token, text string) {
	tsr.ReplaceToken(Default_Program_Name, from, to, text)
}

func (tsr *TokenStreamRewriter) ReplaceTokenDefaultPos(index Token, text string) {
	tsr.ReplaceTokenDefault(index, index, text)
}

func (tsr *TokenStreamRewriter) Delete(program_name string, from, to int) {
	tsr.Replace(program_name, from, to, "")
}

func (tsr *TokenStreamRewriter) DeleteDefault(from, to int) {
	tsr.Delete(Default_Program_Name, from, to)
}

func (tsr *TokenStreamRewriter) DeleteDefaultPos(index int) {
	tsr.DeleteDefault(index, index)
}

func (tsr *TokenStreamRewriter) DeleteToken(program_name string, from, to Token) {
	tsr.ReplaceToken(program_name, from, to, "")
}

func (tsr *TokenStreamRewriter) DeleteTokenDefault(from, to Token) {
	tsr.DeleteToken(Default_Program_Name, from, to)
}

func (tsr *TokenStreamRewriter) GetLastRewriteTokenIndex(program_name string) int {
	i, ok := tsr.last_rewrite_token_indexes[program_name]
	if !ok {
		return -1
	}
	return i
}

func (tsr *TokenStreamRewriter) GetLastRewriteTokenIndexDefault() int {
	return tsr.GetLastRewriteTokenIndex(Default_Program_Name)
}

func (tsr *TokenStreamRewriter) SetLastRewriteTokenIndex(program_name string, i int) {
	tsr.last_rewrite_token_indexes[program_name] = i
}

func (tsr *TokenStreamRewriter) InitializeProgram(name string) []RewriteOperation {
	is := make([]RewriteOperation, 0, Program_Init_Size)
	tsr.programs[name] = is
	return is
}

func (tsr *TokenStreamRewriter) AddToProgram(name string, op RewriteOperation) {
	is := tsr.GetProgram(name)
	is = append(is, op)
	tsr.programs[name] = is
}

func (tsr *TokenStreamRewriter) GetProgram(name string) []RewriteOperation {
	is, ok := tsr.programs[name]
	if !ok {
		is = tsr.InitializeProgram(name)
	}
	return is
}

// Return the text from the original tokens altered per the
// instructions given to this rewriter.
func (tsr *TokenStreamRewriter) GetTextDefault() string {
	return tsr.GetText(
		Default_Program_Name,
		NewInterval(0, tsr.tokens.Size()-1))
}

// Return the text from the original tokens altered per the
// instructions given to this rewriter.
func (tsr *TokenStreamRewriter) GetText(program_name string, interval *Interval) string {
	rewrites := tsr.programs[program_name]
	start := interval.Start
	stop := interval.Stop
	// ensure start/end are in range
	stop = min(stop, tsr.tokens.Size()-1)
	start = max(start, 0)
	if rewrites == nil || len(rewrites) == 0 {
		return tsr.tokens.GetTextFromInterval(interval) // no instructions to execute
	}
	buf := bytes.Buffer{}
	// First, optimize instruction stream
	indexToOp := reduceToSingleOperationPerIndex(rewrites)
	// Walk buffer, executing instructions and emitting tokens
	for i := start; i <= stop && i < tsr.tokens.Size(); {
		op := indexToOp[i]
		delete(indexToOp, i) // remove so any left have index size-1
		t := tsr.tokens.Get(i)
		if op == nil {
			// no operation at that index, just dump token
			if t.GetTokenType() != TokenEOF {
				buf.WriteString(t.GetText())
			}
			i++ // move to next token
		} else {
			i = op.Execute(&buf) // execute operation and skip
		}
	}
	// include stuff after end if it's last index in buffer
	// So, if they did an insertAfter(lastValidIndex, "foo"), include
	// foo if end==lastValidIndex.
	if stop == tsr.tokens.Size()-1 {
		// Scan any remaining operations after last token
		// should be included (they will be inserts).
		for _, op := range indexToOp {
			if op.GetIndex() >= tsr.tokens.Size()-1 {
				buf.WriteString(op.GetText())
			}
		}
	}
	return buf.String()
}

//	 We need to combine operations and report invalid operations (like
//	 overlapping replaces that are not completed nested). Inserts to
//	 same index need to be combined etc...  Here are the cases:
//
//	 I.i.u I.j.v								leave alone, nonoverlapping
//	 I.i.u I.i.v								combine: Iivu
//
//	 R.i-j.u R.x-y.v	| i-j in x-y			delete first R
//	 R.i-j.u R.i-j.v							delete first R
//	 R.i-j.u R.x-y.v	| x-y in i-j			ERROR
//	 R.i-j.u R.x-y.v	| boundaries overlap	ERROR
//
//	 Delete special case of replace (text==null):
//	 D.i-j.u D.x-y.v	| boundaries overlap	combine to max(min)..max(right)
//
//	 I.i.u R.x-y.v | i in (x+1)-y			delete I (since insert before
//												we're not deleting i)
//	 I.i.u R.x-y.v | i not in (x+1)-y		leave alone, nonoverlapping
//	 R.x-y.v I.i.u | i in x-y				ERROR
//	 R.x-y.v I.x.u 							R.x-y.uv (combine, delete I)
//	 R.x-y.v I.i.u | i not in x-y			leave alone, nonoverlapping
//
//	 I.i.u = insert u before op @ index i
//	 R.x-y.u = replace x-y indexed tokens with u
//
//	 First we need to examine replaces. For any replace op:
//
//			1. wipe out any insertions before op within that range.
//			2. Drop any replace op before that is contained completely within
//		 	that range.
//			3. Throw exception upon boundary overlap with any previous replace.
//
//	 	Then we can deal with inserts:
//
//			1. for any inserts to same index, combine even if not adjacent.
//			2. for any prior replace with same left boundary, combine this
//		 	insert with replace and delete this replace.
//			3. throw exception if index in same range as previous replace
//
//	 Don't actually delete; make op null in list. Easier to walk list.
//	 Later we can throw as we add to index &rarr; op map.
//
//	 Note that I.2 R.2-2 will wipe out I.2 even though, technically, the
//	 inserted stuff would be before the replace range. But, if you
//	 add tokens in front of a method body '{' and then delete the method
//	 body, I think the stuff before the '{' you added should disappear too.
//
//	 Return a map from token index to operation.
func reduceToSingleOperationPerIndex(rewrites []RewriteOperation) map[int]RewriteOperation {
	// WALK REPLACES
	for i := 0; i < len(rewrites); i++ {
		op := rewrites[i]
		if op == nil {
			continue
		}
		rop, ok := op.(*ReplaceOp)
		if !ok {
			continue
		}
		// Wipe prior inserts within range
		for j := 0; j < i && j < len(rewrites); j++ {
			if iop, ok := rewrites[j].(*InsertBeforeOp); ok {
				if iop.index == rop.index {
					// E.g., insert before 2, delete 2..2; update replace
					// text to include insert before, kill insert
					rewrites[iop.instruction_index] = nil
					if rop.text != "" {
						rop.text = iop.text + rop.text
					} else {
						rop.text = iop.text
					}
				} else if iop.index > rop.index && iop.index <= rop.LastIndex {
					// delete insert as it's a no-op.
					rewrites[iop.instruction_index] = nil
				}
			}
		}
		// Drop any prior replaces contained within
		for j := 0; j < i && j < len(rewrites); j++ {
			if prevop, ok := rewrites[j].(*ReplaceOp); ok {
				if prevop.index >= rop.index && prevop.LastIndex <= rop.LastIndex {
					// delete replace as it's a no-op.
					rewrites[prevop.instruction_index] = nil
					continue
				}
				// throw exception unless disjoint or identical
				disjoint := prevop.LastIndex < rop.index || prevop.index > rop.LastIndex
				// Delete special case of replace (text==null):
				// D.i-j.u D.x-y.v	| boundaries overlap	combine to max(min)..max(right)
				if prevop.text == "" && rop.text == "" && !disjoint {
					rewrites[prevop.instruction_index] = nil
					rop.index = min(prevop.index, rop.index)
					rop.LastIndex = max(prevop.LastIndex, rop.LastIndex)
					println("new rop" + rop.String()) //TODO: remove console write, taken from Java version
				} else if !disjoint {
					panic("replace op boundaries of " + rop.String() + " overlap with previous " + prevop.String())
				}
			}
		}
	}
	// WALK INSERTS
	for i := 0; i < len(rewrites); i++ {
		op := rewrites[i]
		if op == nil {
			continue
		}
		//hack to replicate inheritance in composition
		_, iok := rewrites[i].(*InsertBeforeOp)
		_, aok := rewrites[i].(*InsertAfterOp)
		if !iok && !aok {
			continue
		}
		iop := rewrites[i]
		// combine current insert with prior if any at same index
		// deviating a bit from TokenStreamRewriter.java - hard to incorporate inheritance logic
		for j := 0; j < i && j < len(rewrites); j++ {
			if nextIop, ok := rewrites[j].(*InsertAfterOp); ok {
				if nextIop.index == iop.GetIndex() {
					iop.SetText(nextIop.text + iop.GetText())
					rewrites[j] = nil
				}
			}
			if prevIop, ok := rewrites[j].(*InsertBeforeOp); ok {
				if prevIop.index == iop.GetIndex() {
					iop.SetText(iop.GetText() + prevIop.text)
					rewrites[prevIop.instruction_index] = nil
				}
			}
		}
		// look for replaces where iop.index is in range; error
		for j := 0; j < i && j < len(rewrites); j++ {
			if rop, ok := rewrites[j].(*ReplaceOp); ok {
				if iop.GetIndex() == rop.index {
					rop.text = iop.GetText() + rop.text
					rewrites[i] = nil
					continue
				}
				if iop.GetIndex() >= rop.index && iop.GetIndex() <= rop.LastIndex {
					panic("insert op " + iop.String() + " within boundaries of previous " + rop.String())
				}
			}
		}
	}
	m := map[int]RewriteOperation{}
	for i := 0; i < len(rewrites); i++ {
		op := rewrites[i]
		if op == nil {
			continue
		}
		if _, ok := m[op.GetIndex()]; ok {
			panic("should only be one op per index")
		}
		m[op.GetIndex()] = op
	}
	return m
}

/*
	Quick fixing Go lack of overloads
*/

func max(a, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}
func min(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}
