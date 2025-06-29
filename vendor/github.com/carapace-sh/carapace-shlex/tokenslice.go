package shlex

import (
	"strconv"
)

type TokenSlice []Token

func (t TokenSlice) Strings() []string {
	s := make([]string, 0, len(t))
	for _, token := range t {
		s = append(s, token.Value)
	}
	return s
}

func (t TokenSlice) Pipelines() []TokenSlice {
	pipelines := make([]TokenSlice, 0)

	pipeline := make(TokenSlice, 0)
	for _, token := range t {
		switch {
		case token.Type == WORDBREAK_TOKEN && wordbreakType(token).IsPipelineDelimiter():
			pipelines = append(pipelines, pipeline)
			pipeline = make(TokenSlice, 0)
		default:
			pipeline = append(pipeline, token)
		}
	}
	return append(pipelines, pipeline)
}

func (t TokenSlice) CurrentPipeline() TokenSlice {
	pipelines := t.Pipelines()
	return pipelines[len(pipelines)-1]
}

func (t TokenSlice) Words() TokenSlice {
	words := make(TokenSlice, 0)
	for index, token := range t {
		switch {
		case index == 0:
			words = append(words, token)
		case t[index-1].adjoins(token):
			words[len(words)-1].Value += token.Value
			words[len(words)-1].RawValue += token.RawValue
			words[len(words)-1].State = token.State
		default:
			words = append(words, token)
		}
	}
	return words
}

func (t TokenSlice) FilterRedirects() TokenSlice {
	filtered := make(TokenSlice, 0)
	for index, token := range t {
		switch token.Type {
		case WORDBREAK_TOKEN:
			if wordbreakType(token).IsRedirect() {
				continue
			}
		}

		if index > 0 {
			if wordbreakType(t[index-1]).IsRedirect() {
				continue
			}
		}

		if index < len(t)-1 {
			next := t[index+1]
			if token.adjoins(next) {
				if _, err := strconv.Atoi(token.RawValue); err == nil {
					if wordbreakType(t[index+1]).IsRedirect() {
						continue
					}
				}
			}

		}

		filtered = append(filtered, token)
	}
	return filtered
}

func (t TokenSlice) CurrentToken() (token Token) {
	if len(t) > 0 {
		token = t[len(t)-1]
	}
	return
}

func (t TokenSlice) WordbreakPrefix() string {
	found := false
	prefix := ""

	last := t[len(t)-1]
	switch last.State {
	case QUOTING_STATE, QUOTING_ESCAPING_STATE, ESCAPING_QUOTED_STATE:
		// Seems bash handles the last opening quote as wordbreak when in quoting state.
		// So add value up to last opening quote to prefix.
		found = true
		prefix = last.Value[:last.WordbreakIndex]
	}

	for i := len(t) - 2; i >= 0; i-- {
		token := t[i]
		if !token.adjoins(t[i+1]) {
			break
		}

		if token.Type == WORDBREAK_TOKEN {
			found = true
		}

		if found {
			prefix = token.Value + prefix
		}
	}
	return prefix
}
