package ssh_config

import (
	"fmt"
	"strings"
)

type sshParser struct {
	flow          chan token
	config        *Config
	tokensBuffer  []token
	currentTable  []string
	seenTableKeys []string
	// /etc/ssh parser or local parser - used to find the default for relative
	// filepaths in the Include directive
	system bool
	depth  uint8
}

type sshParserStateFn func() sshParserStateFn

// Formats and panics an error message based on a token
func (p *sshParser) raiseErrorf(tok *token, msg string, args ...interface{}) {
	// TODO this format is ugly
	panic(tok.Position.String() + ": " + fmt.Sprintf(msg, args...))
}

func (p *sshParser) raiseError(tok *token, err error) {
	if err == ErrDepthExceeded {
		panic(err)
	}
	// TODO this format is ugly
	panic(tok.Position.String() + ": " + err.Error())
}

func (p *sshParser) run() {
	for state := p.parseStart; state != nil; {
		state = state()
	}
}

func (p *sshParser) peek() *token {
	if len(p.tokensBuffer) != 0 {
		return &(p.tokensBuffer[0])
	}

	tok, ok := <-p.flow
	if !ok {
		return nil
	}
	p.tokensBuffer = append(p.tokensBuffer, tok)
	return &tok
}

func (p *sshParser) getToken() *token {
	if len(p.tokensBuffer) != 0 {
		tok := p.tokensBuffer[0]
		p.tokensBuffer = p.tokensBuffer[1:]
		return &tok
	}
	tok, ok := <-p.flow
	if !ok {
		return nil
	}
	return &tok
}

func (p *sshParser) parseStart() sshParserStateFn {
	tok := p.peek()

	// end of stream, parsing is finished
	if tok == nil {
		return nil
	}

	switch tok.typ {
	case tokenComment, tokenEmptyLine:
		return p.parseComment
	case tokenKey:
		return p.parseKV
	case tokenEOF:
		return nil
	default:
		p.raiseErrorf(tok, fmt.Sprintf("unexpected token %q\n", tok))
	}
	return nil
}

func (p *sshParser) parseKV() sshParserStateFn {
	key := p.getToken()
	hasEquals := false
	val := p.getToken()
	if val.typ == tokenEquals {
		hasEquals = true
		val = p.getToken()
	}
	comment := ""
	tok := p.peek()
	if tok == nil {
		tok = &token{typ: tokenEOF}
	}
	if tok.typ == tokenComment && tok.Position.Line == val.Position.Line {
		tok = p.getToken()
		comment = tok.val
	}
	if strings.ToLower(key.val) == "match" {
		// https://github.com/kevinburke/ssh_config/issues/6
		p.raiseErrorf(val, "ssh_config: Match directive parsing is unsupported")
		return nil
	}
	if strings.ToLower(key.val) == "host" {
		strPatterns := strings.Split(val.val, " ")
		patterns := make([]*Pattern, 0)
		for i := range strPatterns {
			if strPatterns[i] == "" {
				continue
			}
			pat, err := NewPattern(strPatterns[i])
			if err != nil {
				p.raiseErrorf(val, "Invalid host pattern: %v", err)
				return nil
			}
			patterns = append(patterns, pat)
		}
		p.config.Hosts = append(p.config.Hosts, &Host{
			Patterns:   patterns,
			Nodes:      make([]Node, 0),
			EOLComment: comment,
			hasEquals:  hasEquals,
		})
		return p.parseStart
	}
	lastHost := p.config.Hosts[len(p.config.Hosts)-1]
	if strings.ToLower(key.val) == "include" {
		inc, err := NewInclude(strings.Split(val.val, " "), hasEquals, key.Position, comment, p.system, p.depth+1)
		if err == ErrDepthExceeded {
			p.raiseError(val, err)
			return nil
		}
		if err != nil {
			p.raiseErrorf(val, "Error parsing Include directive: %v", err)
			return nil
		}
		lastHost.Nodes = append(lastHost.Nodes, inc)
		return p.parseStart
	}
	kv := &KV{
		Key:          key.val,
		Value:        val.val,
		Comment:      comment,
		hasEquals:    hasEquals,
		leadingSpace: key.Position.Col - 1,
		position:     key.Position,
	}
	lastHost.Nodes = append(lastHost.Nodes, kv)
	return p.parseStart
}

func (p *sshParser) parseComment() sshParserStateFn {
	comment := p.getToken()
	lastHost := p.config.Hosts[len(p.config.Hosts)-1]
	lastHost.Nodes = append(lastHost.Nodes, &Empty{
		Comment: comment.val,
		// account for the "#" as well
		leadingSpace: comment.Position.Col - 2,
		position:     comment.Position,
	})
	return p.parseStart
}

func parseSSH(flow chan token, system bool, depth uint8) *Config {
	// Ensure we consume tokens to completion even if parser exits early
	defer func() {
		for range flow {
		}
	}()

	result := newConfig()
	result.position = Position{1, 1}
	parser := &sshParser{
		flow:          flow,
		config:        result,
		tokensBuffer:  make([]token, 0),
		currentTable:  make([]string, 0),
		seenTableKeys: make([]string, 0),
		system:        system,
		depth:         depth,
	}
	parser.run()
	return result
}
