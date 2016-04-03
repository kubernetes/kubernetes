package builder

// This will take a single word and an array of env variables and
// process all quotes (" and ') as well as $xxx and ${xxx} env variable
// tokens.  Tries to mimic bash shell process.
// It doesn't support all flavors of ${xx:...} formats but new ones can
// be added by adding code to the "special ${} format processing" section

import (
	"fmt"
	"strings"
	"unicode"
)

type shellWord struct {
	word string
	envs []string
	pos  int
}

func ProcessWord(word string, env []string) (string, error) {
	sw := &shellWord{
		word: word,
		envs: env,
		pos:  0,
	}
	return sw.process()
}

func (sw *shellWord) process() (string, error) {
	return sw.processStopOn('\000')
}

// Process the word, starting at 'pos', and stop when we get to the
// end of the word or the 'stopChar' character
func (sw *shellWord) processStopOn(stopChar rune) (string, error) {
	var result string
	var charFuncMapping = map[rune]func() (string, error){
		'\'': sw.processSingleQuote,
		'"':  sw.processDoubleQuote,
		'$':  sw.processDollar,
	}

	for sw.pos < len(sw.word) {
		ch := sw.peek()
		if stopChar != '\000' && ch == stopChar {
			sw.next()
			break
		}
		if fn, ok := charFuncMapping[ch]; ok {
			// Call special processing func for certain chars
			tmp, err := fn()
			if err != nil {
				return "", err
			}
			result += tmp
		} else {
			// Not special, just add it to the result
			ch = sw.next()
			if ch == '\\' {
				// '\' escapes, except end of line
				ch = sw.next()
				if ch == '\000' {
					continue
				}
			}
			result += string(ch)
		}
	}

	return result, nil
}

func (sw *shellWord) peek() rune {
	if sw.pos == len(sw.word) {
		return '\000'
	}
	return rune(sw.word[sw.pos])
}

func (sw *shellWord) next() rune {
	if sw.pos == len(sw.word) {
		return '\000'
	}
	ch := rune(sw.word[sw.pos])
	sw.pos++
	return ch
}

func (sw *shellWord) processSingleQuote() (string, error) {
	// All chars between single quotes are taken as-is
	// Note, you can't escape '
	var result string

	sw.next()

	for {
		ch := sw.next()
		if ch == '\000' || ch == '\'' {
			break
		}
		result += string(ch)
	}
	return result, nil
}

func (sw *shellWord) processDoubleQuote() (string, error) {
	// All chars up to the next " are taken as-is, even ', except any $ chars
	// But you can escape " with a \
	var result string

	sw.next()

	for sw.pos < len(sw.word) {
		ch := sw.peek()
		if ch == '"' {
			sw.next()
			break
		}
		if ch == '$' {
			tmp, err := sw.processDollar()
			if err != nil {
				return "", err
			}
			result += tmp
		} else {
			ch = sw.next()
			if ch == '\\' {
				chNext := sw.peek()

				if chNext == '\000' {
					// Ignore \ at end of word
					continue
				}

				if chNext == '"' || chNext == '$' {
					// \" and \$ can be escaped, all other \'s are left as-is
					ch = sw.next()
				}
			}
			result += string(ch)
		}
	}

	return result, nil
}

func (sw *shellWord) processDollar() (string, error) {
	sw.next()
	ch := sw.peek()
	if ch == '{' {
		sw.next()
		name := sw.processName()
		ch = sw.peek()
		if ch == '}' {
			// Normal ${xx} case
			sw.next()
			return sw.getEnv(name), nil
		}
		if ch == ':' {
			// Special ${xx:...} format processing
			// Yes it allows for recursive $'s in the ... spot

			sw.next() // skip over :
			modifier := sw.next()

			word, err := sw.processStopOn('}')
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
				return "", fmt.Errorf("Unsupported modifier (%c) in substitution: %s", modifier, sw.word)
			}
		}
		return "", fmt.Errorf("Missing ':' in substitution: %s", sw.word)
	}
	// $xxx case
	name := sw.processName()
	if name == "" {
		return "$", nil
	}
	return sw.getEnv(name), nil
}

func (sw *shellWord) processName() string {
	// Read in a name (alphanumeric or _)
	// If it starts with a numeric then just return $#
	var name string

	for sw.pos < len(sw.word) {
		ch := sw.peek()
		if len(name) == 0 && unicode.IsDigit(ch) {
			ch = sw.next()
			return string(ch)
		}
		if !unicode.IsLetter(ch) && !unicode.IsDigit(ch) && ch != '_' {
			break
		}
		ch = sw.next()
		name += string(ch)
	}

	return name
}

func (sw *shellWord) getEnv(name string) string {
	for _, env := range sw.envs {
		i := strings.Index(env, "=")
		if i < 0 {
			if name == env {
				// Should probably never get here, but just in case treat
				// it like "var" and "var=" are the same
				return ""
			}
			continue
		}
		if name != env[:i] {
			continue
		}
		return env[i+1:]
	}
	return ""
}
