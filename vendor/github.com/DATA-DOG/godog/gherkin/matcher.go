package gherkin

import (
	"regexp"
	"strings"
	"unicode/utf8"
)

const (
	DEFAULT_DIALECT                 = "en"
	COMMENT_PREFIX                  = "#"
	TAG_PREFIX                      = "@"
	TITLE_KEYWORD_SEPARATOR         = ":"
	TABLE_CELL_SEPARATOR            = '|'
	ESCAPE_CHAR                     = '\\'
	ESCAPED_NEWLINE                 = 'n'
	DOCSTRING_SEPARATOR             = "\"\"\""
	DOCSTRING_ALTERNATIVE_SEPARATOR = "```"
)

type matcher struct {
	gdp                      GherkinDialectProvider
	default_lang             string
	lang                     string
	dialect                  *GherkinDialect
	activeDocStringSeparator string
	indentToRemove           int
	languagePattern          *regexp.Regexp
}

func NewMatcher(gdp GherkinDialectProvider) Matcher {
	return &matcher{
		gdp:             gdp,
		default_lang:    DEFAULT_DIALECT,
		lang:            DEFAULT_DIALECT,
		dialect:         gdp.GetDialect(DEFAULT_DIALECT),
		languagePattern: regexp.MustCompile("^\\s*#\\s*language\\s*:\\s*([a-zA-Z\\-_]+)\\s*$"),
	}
}

func NewLanguageMatcher(gdp GherkinDialectProvider, language string) Matcher {
	return &matcher{
		gdp:             gdp,
		default_lang:    language,
		lang:            language,
		dialect:         gdp.GetDialect(language),
		languagePattern: regexp.MustCompile("^\\s*#\\s*language\\s*:\\s*([a-zA-Z\\-_]+)\\s*$"),
	}
}

func (m *matcher) Reset() {
	m.indentToRemove = 0
	m.activeDocStringSeparator = ""
	if m.lang != "en" {
		m.dialect = m.gdp.GetDialect(m.default_lang)
		m.lang = "en"
	}
}

func (m *matcher) newTokenAtLocation(line, index int) (token *Token) {
	column := index + 1
	token = new(Token)
	token.GherkinDialect = m.lang
	token.Location = &Location{line, column}
	return
}

func (m *matcher) MatchEOF(line *Line) (ok bool, token *Token, err error) {
	if line.IsEof() {
		token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
		token.Type = TokenType_EOF
	}
	return
}

func (m *matcher) MatchEmpty(line *Line) (ok bool, token *Token, err error) {
	if line.IsEmpty() {
		token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
		token.Type = TokenType_Empty
	}
	return
}

func (m *matcher) MatchComment(line *Line) (ok bool, token *Token, err error) {
	if line.StartsWith(COMMENT_PREFIX) {
		token, ok = m.newTokenAtLocation(line.LineNumber, 0), true
		token.Type = TokenType_Comment
		token.Text = line.LineText
	}
	return
}

func (m *matcher) MatchTagLine(line *Line) (ok bool, token *Token, err error) {
	if line.StartsWith(TAG_PREFIX) {
		var tags []*LineSpan
		var column = line.Indent()
		splits := strings.Split(line.TrimmedLineText, TAG_PREFIX)
		for i := range splits {
			txt := strings.Trim(splits[i], " ")
			if txt != "" {
				tags = append(tags, &LineSpan{column, TAG_PREFIX + txt})
			}
			column = column + len(splits[i]) + 1
		}

		token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
		token.Type = TokenType_TagLine
		token.Items = tags
	}
	return
}

func (m *matcher) matchTitleLine(line *Line, tokenType TokenType, keywords []string) (ok bool, token *Token, err error) {
	for i := range keywords {
		keyword := keywords[i]
		if line.StartsWith(keyword + TITLE_KEYWORD_SEPARATOR) {
			token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
			token.Type = tokenType
			token.Keyword = keyword
			token.Text = strings.Trim(line.TrimmedLineText[len(keyword)+1:], " ")
			return
		}
	}
	return
}

func (m *matcher) MatchFeatureLine(line *Line) (ok bool, token *Token, err error) {
	return m.matchTitleLine(line, TokenType_FeatureLine, m.dialect.FeatureKeywords())
}
func (m *matcher) MatchBackgroundLine(line *Line) (ok bool, token *Token, err error) {
	return m.matchTitleLine(line, TokenType_BackgroundLine, m.dialect.BackgroundKeywords())
}
func (m *matcher) MatchScenarioLine(line *Line) (ok bool, token *Token, err error) {
	return m.matchTitleLine(line, TokenType_ScenarioLine, m.dialect.ScenarioKeywords())
}
func (m *matcher) MatchScenarioOutlineLine(line *Line) (ok bool, token *Token, err error) {
	return m.matchTitleLine(line, TokenType_ScenarioOutlineLine, m.dialect.ScenarioOutlineKeywords())
}
func (m *matcher) MatchExamplesLine(line *Line) (ok bool, token *Token, err error) {
	return m.matchTitleLine(line, TokenType_ExamplesLine, m.dialect.ExamplesKeywords())
}
func (m *matcher) MatchStepLine(line *Line) (ok bool, token *Token, err error) {
	keywords := m.dialect.StepKeywords()
	for i := range keywords {
		keyword := keywords[i]
		if line.StartsWith(keyword) {
			token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
			token.Type = TokenType_StepLine
			token.Keyword = keyword
			token.Text = strings.Trim(line.TrimmedLineText[len(keyword):], " ")
			return
		}
	}
	return
}

func (m *matcher) MatchDocStringSeparator(line *Line) (ok bool, token *Token, err error) {
	if m.activeDocStringSeparator != "" {
		if line.StartsWith(m.activeDocStringSeparator) {
			// close
			token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
			token.Type = TokenType_DocStringSeparator

			m.indentToRemove = 0
			m.activeDocStringSeparator = ""
		}
		return
	}
	if line.StartsWith(DOCSTRING_SEPARATOR) {
		m.activeDocStringSeparator = DOCSTRING_SEPARATOR
	} else if line.StartsWith(DOCSTRING_ALTERNATIVE_SEPARATOR) {
		m.activeDocStringSeparator = DOCSTRING_ALTERNATIVE_SEPARATOR
	}
	if m.activeDocStringSeparator != "" {
		// open
		contentType := line.TrimmedLineText[len(m.activeDocStringSeparator):]
		m.indentToRemove = line.Indent()
		token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
		token.Type = TokenType_DocStringSeparator
		token.Text = contentType
	}
	return
}

func (m *matcher) MatchTableRow(line *Line) (ok bool, token *Token, err error) {
	var firstChar, firstPos = utf8.DecodeRuneInString(line.TrimmedLineText)
	if firstChar == TABLE_CELL_SEPARATOR {
		var cells []*LineSpan
		var cell []rune
		var startCol = line.Indent() + 2 // column where the current cell started
		// start after the first separator, it's not included in the cell
		for i, w, col := firstPos, 0, startCol; i < len(line.TrimmedLineText); i += w {
			var char rune
			char, w = utf8.DecodeRuneInString(line.TrimmedLineText[i:])
			if char == TABLE_CELL_SEPARATOR {
				// append current cell
				txt := string(cell)
				txtTrimmed := strings.TrimLeft(txt, " ")
				ind := len(txt) - len(txtTrimmed)
				cells = append(cells, &LineSpan{startCol + ind, strings.TrimRight(txtTrimmed, " ")})
				// start building next
				cell = make([]rune, 0)
				startCol = col + 1
			} else if char == ESCAPE_CHAR {
				// skip this character but count the column
				i += w
				col++
				char, w = utf8.DecodeRuneInString(line.TrimmedLineText[i:])
				if char == ESCAPED_NEWLINE {
					cell = append(cell, '\n')
				} else {
					if char != TABLE_CELL_SEPARATOR && char != ESCAPE_CHAR {
						cell = append(cell, ESCAPE_CHAR)
					}
					cell = append(cell, char)
				}
			} else {
				cell = append(cell, char)
			}
			col++
		}

		token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
		token.Type = TokenType_TableRow
		token.Items = cells
	}
	return
}

func (m *matcher) MatchLanguage(line *Line) (ok bool, token *Token, err error) {
	matches := m.languagePattern.FindStringSubmatch(line.TrimmedLineText)
	if len(matches) > 0 {
		lang := matches[1]
		token, ok = m.newTokenAtLocation(line.LineNumber, line.Indent()), true
		token.Type = TokenType_Language
		token.Text = lang

		dialect := m.gdp.GetDialect(lang)
		if dialect == nil {
			err = &parseError{"Language not supported: " + lang, token.Location}
		} else {
			m.lang = lang
			m.dialect = dialect
		}
	}
	return
}

func (m *matcher) MatchOther(line *Line) (ok bool, token *Token, err error) {
	token, ok = m.newTokenAtLocation(line.LineNumber, 0), true
	token.Type = TokenType_Other

	element := line.LineText
	txt := strings.TrimLeft(element, " ")

	if len(element)-len(txt) > m.indentToRemove {
		token.Text = m.unescapeDocString(element[m.indentToRemove:])
	} else {
		token.Text = m.unescapeDocString(txt)
	}
	return
}

func (m *matcher) unescapeDocString(text string) string {
	if m.activeDocStringSeparator != "" {
		return strings.Replace(text, "\\\"\\\"\\\"", "\"\"\"", -1)
	} else {
		return text
	}
}
