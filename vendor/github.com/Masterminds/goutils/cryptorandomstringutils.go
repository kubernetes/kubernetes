/*
Copyright 2014 Alexander Okoli

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package goutils

import (
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"unicode"
)

/*
CryptoRandomNonAlphaNumeric creates a random string whose length is the number of characters specified.
Characters will be chosen from the set of all characters (ASCII/Unicode values between 0 to 2,147,483,647 (math.MaxInt32)).

Parameter:
	count - the length of random string to create

Returns:
	string - the random string
	error - an error stemming from an invalid parameter within underlying function, CryptoRandom(...)
*/
func CryptoRandomNonAlphaNumeric(count int) (string, error) {
	return CryptoRandomAlphaNumericCustom(count, false, false)
}

/*
CryptoRandomAscii creates a random string whose length is the number of characters specified.
Characters will be chosen from the set of characters whose ASCII value is between 32 and 126 (inclusive).

Parameter:
	count - the length of random string to create

Returns:
	string - the random string
	error - an error stemming from an invalid parameter within underlying function, CryptoRandom(...)
*/
func CryptoRandomAscii(count int) (string, error) {
	return CryptoRandom(count, 32, 127, false, false)
}

/*
CryptoRandomNumeric creates a random string whose length is the number of characters specified.
Characters will be chosen from the set of numeric characters.

Parameter:
	count - the length of random string to create

Returns:
	string - the random string
	error - an error stemming from an invalid parameter within underlying function, CryptoRandom(...)
*/
func CryptoRandomNumeric(count int) (string, error) {
	return CryptoRandom(count, 0, 0, false, true)
}

/*
CryptoRandomAlphabetic creates a random string whose length is the number of characters specified.
Characters will be chosen from the set of alpha-numeric characters as indicated by the arguments.

Parameters:
	count - the length of random string to create
	letters - if true, generated string may include alphabetic characters
	numbers - if true, generated string may include numeric characters

Returns:
	string - the random string
	error - an error stemming from an invalid parameter within underlying function, CryptoRandom(...)
*/
func CryptoRandomAlphabetic(count int) (string, error) {
	return CryptoRandom(count, 0, 0, true, false)
}

/*
CryptoRandomAlphaNumeric creates a random string whose length is the number of characters specified.
Characters will be chosen from the set of alpha-numeric characters.

Parameter:
	count - the length of random string to create

Returns:
	string - the random string
	error - an error stemming from an invalid parameter within underlying function, CryptoRandom(...)
*/
func CryptoRandomAlphaNumeric(count int) (string, error) {
	if count == 0 {
		return "", nil
	}
	RandomString, err := CryptoRandom(count, 0, 0, true, true)
	if err != nil {
		return "", fmt.Errorf("Error: %s", err)
	}
	match, err := regexp.MatchString("([0-9]+)", RandomString)
	if err != nil {
		panic(err)
	}

	if !match {
		//Get the position between 0 and the length of the string-1  to insert a random number
		position := getCryptoRandomInt(count)
		//Insert a random number between [0-9] in the position
		RandomString = RandomString[:position] + string('0' + getCryptoRandomInt(10)) + RandomString[position + 1:]
		return RandomString, err
	}
	return RandomString, err

}

/*
CryptoRandomAlphaNumericCustom creates a random string whose length is the number of characters specified.
Characters will be chosen from the set of alpha-numeric characters as indicated by the arguments.

Parameters:
	count - the length of random string to create
	letters - if true, generated string may include alphabetic characters
	numbers - if true, generated string may include numeric characters

Returns:
	string - the random string
	error - an error stemming from an invalid parameter within underlying function, CryptoRandom(...)
*/
func CryptoRandomAlphaNumericCustom(count int, letters bool, numbers bool) (string, error) {
	return CryptoRandom(count, 0, 0, letters, numbers)
}

/*
CryptoRandom creates a random string based on a variety of options, using using golang's crypto/rand source of randomness.
If the parameters start and end are both 0, start and end are set to ' ' and 'z', the ASCII printable characters, will be used,
unless letters and numbers are both false, in which case, start and end are set to 0 and math.MaxInt32, respectively.
If chars is not nil, characters stored in chars that are between start and end are chosen.

Parameters:
	count - the length of random string to create
	start - the position in set of chars (ASCII/Unicode int) to start at
	end - the position in set of chars (ASCII/Unicode int) to end before
	letters - if true, generated string may include alphabetic characters
	numbers - if true, generated string may include numeric characters
	chars - the set of chars to choose randoms from. If nil, then it will use the set of all chars.

Returns:
	string - the random string
	error - an error stemming from invalid parameters: if count < 0; or the provided chars array is empty; or end <= start; or end > len(chars)
*/
func CryptoRandom(count int, start int, end int, letters bool, numbers bool, chars ...rune) (string, error) {
	if count == 0 {
		return "", nil
	} else if count < 0 {
		err := fmt.Errorf("randomstringutils illegal argument: Requested random string length %v is less than 0.", count) // equiv to err := errors.New("...")
		return "", err
	}
	if chars != nil && len(chars) == 0 {
		err := fmt.Errorf("randomstringutils illegal argument: The chars array must not be empty")
		return "", err
	}

	if start == 0 && end == 0 {
		if chars != nil {
			end = len(chars)
		} else {
			if !letters && !numbers {
				end = math.MaxInt32
			} else {
				end = 'z' + 1
				start = ' '
			}
		}
	} else {
		if end <= start {
			err := fmt.Errorf("randomstringutils illegal argument: Parameter end (%v) must be greater than start (%v)", end, start)
			return "", err
		}

		if chars != nil && end > len(chars) {
			err := fmt.Errorf("randomstringutils illegal argument: Parameter end (%v) cannot be greater than len(chars) (%v)", end, len(chars))
			return "", err
		}
	}

	buffer := make([]rune, count)
	gap := end - start

	// high-surrogates range, (\uD800-\uDBFF) = 55296 - 56319
	//  low-surrogates range, (\uDC00-\uDFFF) = 56320 - 57343

	for count != 0 {
		count--
		var ch rune
		if chars == nil {
			ch = rune(getCryptoRandomInt(gap) + int64(start))
		} else {
			ch = chars[getCryptoRandomInt(gap) + int64(start)]
		}

		if letters && unicode.IsLetter(ch) || numbers && unicode.IsDigit(ch) || !letters && !numbers {
			if ch >= 56320 && ch <= 57343 { // low surrogate range
				if count == 0 {
					count++
				} else {
					// Insert low surrogate
					buffer[count] = ch
					count--
					// Insert high surrogate
					buffer[count] = rune(55296 + getCryptoRandomInt(128))
				}
			} else if ch >= 55296 && ch <= 56191 { // High surrogates range (Partial)
				if count == 0 {
					count++
				} else {
					// Insert low surrogate
					buffer[count] = rune(56320 + getCryptoRandomInt(128))
					count--
					// Insert high surrogate
					buffer[count] = ch
				}
			} else if ch >= 56192 && ch <= 56319 {
				// private high surrogate, skip it
				count++
			} else {
				// not one of the surrogates*
				buffer[count] = ch
			}
		} else {
			count++
		}
	}
	return string(buffer), nil
}

func getCryptoRandomInt(count int) int64 {
	nBig, err := rand.Int(rand.Reader, big.NewInt(int64(count)))
	if err != nil {
		panic(err)
	}
	return nBig.Int64()
}
