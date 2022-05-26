package ssh_config

import (
	"bytes"
)

// Define state functions
type sshLexStateFn func() sshLexStateFn

type sshLexer struct {
	inputIdx int
	input    []rune // Textual source

	buffer        []rune // Runes composing the current token
	tokens        chan token
	line          int
	col           int
	endbufferLine int
	endbufferCol  int
}

func (s *sshLexer) lexComment(previousState sshLexStateFn) sshLexStateFn {
	return func() sshLexStateFn {
		growingString := ""
		for next := s.peek(); next != '\n' && next != eof; next = s.peek() {
			if next == '\r' && s.follow("\r\n") {
				break
			}
			growingString += string(next)
			s.next()
		}
		s.emitWithValue(tokenComment, growingString)
		s.skip()
		return previousState
	}
}

// lex the space after an equals sign in a function
func (s *sshLexer) lexRspace() sshLexStateFn {
	for {
		next := s.peek()
		if !isSpace(next) {
			break
		}
		s.skip()
	}
	return s.lexRvalue
}

func (s *sshLexer) lexEquals() sshLexStateFn {
	for {
		next := s.peek()
		if next == '=' {
			s.emit(tokenEquals)
			s.skip()
			return s.lexRspace
		}
		// TODO error handling here; newline eof etc.
		if !isSpace(next) {
			break
		}
		s.skip()
	}
	return s.lexRvalue
}

func (s *sshLexer) lexKey() sshLexStateFn {
	growingString := ""

	for r := s.peek(); isKeyChar(r); r = s.peek() {
		// simplified a lot here
		if isSpace(r) || r == '=' {
			s.emitWithValue(tokenKey, growingString)
			s.skip()
			return s.lexEquals
		}
		growingString += string(r)
		s.next()
	}
	s.emitWithValue(tokenKey, growingString)
	return s.lexEquals
}

func (s *sshLexer) lexRvalue() sshLexStateFn {
	growingString := ""
	for {
		next := s.peek()
		switch next {
		case '\r':
			if s.follow("\r\n") {
				s.emitWithValue(tokenString, growingString)
				s.skip()
				return s.lexVoid
			}
		case '\n':
			s.emitWithValue(tokenString, growingString)
			s.skip()
			return s.lexVoid
		case '#':
			s.emitWithValue(tokenString, growingString)
			s.skip()
			return s.lexComment(s.lexVoid)
		case eof:
			s.next()
		}
		if next == eof {
			break
		}
		growingString += string(next)
		s.next()
	}
	s.emit(tokenEOF)
	return nil
}

func (s *sshLexer) read() rune {
	r := s.peek()
	if r == '\n' {
		s.endbufferLine++
		s.endbufferCol = 1
	} else {
		s.endbufferCol++
	}
	s.inputIdx++
	return r
}

func (s *sshLexer) next() rune {
	r := s.read()

	if r != eof {
		s.buffer = append(s.buffer, r)
	}
	return r
}

func (s *sshLexer) lexVoid() sshLexStateFn {
	for {
		next := s.peek()
		switch next {
		case '#':
			s.skip()
			return s.lexComment(s.lexVoid)
		case '\r':
			fallthrough
		case '\n':
			s.emit(tokenEmptyLine)
			s.skip()
			continue
		}

		if isSpace(next) {
			s.skip()
		}

		if isKeyStartChar(next) {
			return s.lexKey
		}

		// removed IsKeyStartChar and lexKey. probably will need to readd

		if next == eof {
			s.next()
			break
		}
	}

	s.emit(tokenEOF)
	return nil
}

func (s *sshLexer) ignore() {
	s.buffer = make([]rune, 0)
	s.line = s.endbufferLine
	s.col = s.endbufferCol
}

func (s *sshLexer) skip() {
	s.next()
	s.ignore()
}

func (s *sshLexer) emit(t tokenType) {
	s.emitWithValue(t, string(s.buffer))
}

func (s *sshLexer) emitWithValue(t tokenType, value string) {
	tok := token{
		Position: Position{s.line, s.col},
		typ:      t,
		val:      value,
	}
	s.tokens <- tok
	s.ignore()
}

func (s *sshLexer) peek() rune {
	if s.inputIdx >= len(s.input) {
		return eof
	}

	r := s.input[s.inputIdx]
	return r
}

func (s *sshLexer) follow(next string) bool {
	inputIdx := s.inputIdx
	for _, expectedRune := range next {
		if inputIdx >= len(s.input) {
			return false
		}
		r := s.input[inputIdx]
		inputIdx++
		if expectedRune != r {
			return false
		}
	}
	return true
}

func (s *sshLexer) run() {
	for state := s.lexVoid; state != nil; {
		state = state()
	}
	close(s.tokens)
}

func lexSSH(input []byte) chan token {
	runes := bytes.Runes(input)
	l := &sshLexer{
		input:         runes,
		tokens:        make(chan token),
		line:          1,
		col:           1,
		endbufferLine: 1,
		endbufferCol:  1,
	}
	go l.run()
	return l.tokens
}
