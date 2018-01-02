package dockerfile

import (
	"bytes"
	"strings"
	"text/scanner"
	"unicode"

	"github.com/pkg/errors"
)

// ShellLex performs shell word splitting and variable expansion.
//
// ShellLex takes a string and an array of env variables and
// process all quotes (" and ') as well as $xxx and ${xxx} env variable
// tokens.  Tries to mimic bash shell process.
// It doesn't support all flavors of ${xx:...} formats but new ones can
// be added by adding code to the "special ${} format processing" section
type ShellLex struct {
	escapeToken rune
}

// NewShellLex creates a new ShellLex which uses escapeToken to escape quotes.
func NewShellLex(escapeToken rune) *ShellLex {
	return &ShellLex{escapeToken: escapeToken}
}

// ProcessWord will use the 'env' list of environment variables,
// and replace any env var references in 'word'.
func (s *ShellLex) ProcessWord(word string, env []string) (string, error) {
	word, _, err := s.process(word, env)
	return word, err
}

// ProcessWords will use the 'env' list of environment variables,
// and replace any env var references in 'word' then it will also
// return a slice of strings which represents the 'word'
// split up based on spaces - taking into account quotes.  Note that
// this splitting is done **after** the env var substitutions are done.
// Note, each one is trimmed to remove leading and trailing spaces (unless
// they are quoted", but ProcessWord retains spaces between words.
func (s *ShellLex) ProcessWords(word string, env []string) ([]string, error) {
	_, words, err := s.process(word, env)
	return words, err
}

func (s *ShellLex) process(word string, env []string) (string, []string, error) {
	sw := &shellWord{
		envs:        env,
		escapeToken: s.escapeToken,
	}
	sw.scanner.Init(strings.NewReader(word))
	return sw.process(word)
}

type shellWord struct {
	scanner     scanner.Scanner
	envs        []string
	escapeToken rune
}

func (sw *shellWord) process(source string) (string, []string, error) {
	word, words, err := sw.processStopOn(scanner.EOF)
	if err != nil {
		err = errors.Wrapf(err, "failed to process %q", source)
	}
	return word, words, err
}

type wordsStruct struct {
	word   string
	words  []string
	inWord bool
}

func (w *wordsStruct) addChar(ch rune) {
	if unicode.IsSpace(ch) && w.inWord {
		if len(w.word) != 0 {
			w.words = append(w.words, w.word)
			w.word = ""
			w.inWord = false
		}
	} else if !unicode.IsSpace(ch) {
		w.addRawChar(ch)
	}
}

func (w *wordsStruct) addRawChar(ch rune) {
	w.word += string(ch)
	w.inWord = true
}

func (w *wordsStruct) addString(str string) {
	var scan scanner.Scanner
	scan.Init(strings.NewReader(str))
	for scan.Peek() != scanner.EOF {
		w.addChar(scan.Next())
	}
}

func (w *wordsStruct) addRawString(str string) {
	w.word += str
	w.inWord = true
}

func (w *wordsStruct) getWords() []string {
	if len(w.word) > 0 {
		w.words = append(w.words, w.word)

		// Just in case we're called again by mistake
		w.word = ""
		w.inWord = false
	}
	return w.words
}

// Process the word, starting at 'pos', and stop when we get to the
// end of the word or the 'stopChar' character
func (sw *shellWord) processStopOn(stopChar rune) (string, []string, error) {
	var result bytes.Buffer
	var words wordsStruct

	var charFuncMapping = map[rune]func() (string, error){
		'\'': sw.processSingleQuote,
		'"':  sw.processDoubleQuote,
		'$':  sw.processDollar,
	}

	for sw.scanner.Peek() != scanner.EOF {
		ch := sw.scanner.Peek()

		if stopChar != scanner.EOF && ch == stopChar {
			sw.scanner.Next()
			break
		}
		if fn, ok := charFuncMapping[ch]; ok {
			// Call special processing func for certain chars
			tmp, err := fn()
			if err != nil {
				return "", []string{}, err
			}
			result.WriteString(tmp)

			if ch == rune('$') {
				words.addString(tmp)
			} else {
				words.addRawString(tmp)
			}
		} else {
			// Not special, just add it to the result
			ch = sw.scanner.Next()

			if ch == sw.escapeToken {
				// '\' (default escape token, but ` allowed) escapes, except end of line
				ch = sw.scanner.Next()

				if ch == scanner.EOF {
					break
				}

				words.addRawChar(ch)
			} else {
				words.addChar(ch)
			}

			result.WriteRune(ch)
		}
	}

	return result.String(), words.getWords(), nil
}

func (sw *shellWord) processSingleQuote() (string, error) {
	// All chars between single quotes are taken as-is
	// Note, you can't escape '
	//
	// From the "sh" man page:
	// Single Quotes
	//   Enclosing characters in single quotes preserves the literal meaning of
	//   all the characters (except single quotes, making it impossible to put
	//   single-quotes in a single-quoted string).

	var result bytes.Buffer

	sw.scanner.Next()

	for {
		ch := sw.scanner.Next()
		switch ch {
		case scanner.EOF:
			return "", errors.New("unexpected end of statement while looking for matching single-quote")
		case '\'':
			return result.String(), nil
		}
		result.WriteRune(ch)
	}
}

func (sw *shellWord) processDoubleQuote() (string, error) {
	// All chars up to the next " are taken as-is, even ', except any $ chars
	// But you can escape " with a \ (or ` if escape token set accordingly)
	//
	// From the "sh" man page:
	// Double Quotes
	//  Enclosing characters within double quotes preserves the literal meaning
	//  of all characters except dollarsign ($), backquote (`), and backslash
	//  (\).  The backslash inside double quotes is historically weird, and
	//  serves to quote only the following characters:
	//    $ ` " \ <newline>.
	//  Otherwise it remains literal.

	var result bytes.Buffer

	sw.scanner.Next()

	for {
		switch sw.scanner.Peek() {
		case scanner.EOF:
			return "", errors.New("unexpected end of statement while looking for matching double-quote")
		case '"':
			sw.scanner.Next()
			return result.String(), nil
		case '$':
			value, err := sw.processDollar()
			if err != nil {
				return "", err
			}
			result.WriteString(value)
		default:
			ch := sw.scanner.Next()
			if ch == sw.escapeToken {
				switch sw.scanner.Peek() {
				case scanner.EOF:
					// Ignore \ at end of word
					continue
				case '"', '$', sw.escapeToken:
					// These chars can be escaped, all other \'s are left as-is
					// Note: for now don't do anything special with ` chars.
					// Not sure what to do with them anyway since we're not going
					// to execute the text in there (not now anyway).
					ch = sw.scanner.Next()
				}
			}
			result.WriteRune(ch)
		}
	}
}

func (sw *shellWord) processDollar() (string, error) {
	sw.scanner.Next()

	// $xxx case
	if sw.scanner.Peek() != '{' {
		name := sw.processName()
		if name == "" {
			return "$", nil
		}
		return sw.getEnv(name), nil
	}

	sw.scanner.Next()
	name := sw.processName()
	ch := sw.scanner.Peek()
	if ch == '}' {
		// Normal ${xx} case
		sw.scanner.Next()
		return sw.getEnv(name), nil
	}
	if ch == ':' {
		// Special ${xx:...} format processing
		// Yes it allows for recursive $'s in the ... spot

		sw.scanner.Next() // skip over :
		modifier := sw.scanner.Next()

		word, _, err := sw.processStopOn('}')
		if err != nil {
			return "", err
		}

		// Grab the current value of the variable in question so we
		// can use to to determine what to do based on the modifier
		newValue := sw.getEnv(name)

		switch modifier {
		case '+':
			if newValue != "" {
				newValue = word
			}
			return newValue, nil

		case '-':
			if newValue == "" {
				newValue = word
			}
			return newValue, nil

		default:
			return "", errors.Errorf("unsupported modifier (%c) in substitution", modifier)
		}
	}
	return "", errors.Errorf("missing ':' in substitution")
}

func (sw *shellWord) processName() string {
	// Read in a name (alphanumeric or _)
	// If it starts with a numeric then just return $#
	var name bytes.Buffer

	for sw.scanner.Peek() != scanner.EOF {
		ch := sw.scanner.Peek()
		if name.Len() == 0 && unicode.IsDigit(ch) {
			ch = sw.scanner.Next()
			return string(ch)
		}
		if !unicode.IsLetter(ch) && !unicode.IsDigit(ch) && ch != '_' {
			break
		}
		ch = sw.scanner.Next()
		name.WriteRune(ch)
	}

	return name.String()
}

func (sw *shellWord) getEnv(name string) string {
	for _, env := range sw.envs {
		i := strings.Index(env, "=")
		if i < 0 {
			if equalEnvKeys(name, env) {
				// Should probably never get here, but just in case treat
				// it like "var" and "var=" are the same
				return ""
			}
			continue
		}
		compareName := env[:i]
		if !equalEnvKeys(name, compareName) {
			continue
		}
		return env[i+1:]
	}
	return ""
}
