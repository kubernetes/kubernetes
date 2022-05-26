This is a goLang port of python-zxcvbn and [zxcvbn](https://github.com/dropbox/zxcvbn), which are python and JavaScript password strength
generators. zxcvbn attempts to give sound password advice through pattern
matching and conservative entropy calculations. It finds 10k common passwords,
common American names and surnames, common English words, and common patterns
like dates, repeats (aaa), sequences (abcd), and QWERTY patterns.

Please refer to https://dropbox.tech/security/zxcvbn-realistic-password-strength-estimation for the full details and
motivation behind zxcbvn. The source code for the original JavaScript (well,
actually CoffeeScript) implementation can be found at:

https://github.com/lowe/zxcvbn

Python at:

https://github.com/dropbox/python-zxcvbn

For full motivation, see:

https://dropbox.tech/security/zxcvbn-realistic-password-strength-estimation

------------------------------------------------------------------------
Use
------------------------------------------------------------------------

The zxcvbn module has the public method PasswordStrength() function. Import zxcvbn, and
call PasswordStrength(password string, userInputs []string).  The function will return a
result dictionary with the following keys:

Entropy            # bits

CrackTime         # estimation of actual crack time, in seconds.

CrackTimeDisplay # same crack time, as a friendlier string:
                   # "instant", "6 minutes", "centuries", etc.

Score              # [0,1,2,3,4] if crack time is less than
                   # [10^2, 10^4, 10^6, 10^8, Infinity].
                   # (useful for implementing a strength bar.)

MatchSequence     # the list of patterns that zxcvbn based the
                   # entropy calculation on.

CalcTime   # how long it took to calculate an answer,
                   # in milliseconds. usually only a few ms.

The userInputs argument is an splice of strings that zxcvbn
will add to its internal dictionary. This can be whatever list of
strings you like, but is meant for user inputs from other fields of the
form, like name and email. That way a password that includes the user's
personal info can be heavily penalized. This list is also good for
site-specific vocabulary.

Bug reports and pull requests welcome!

------------------------------------------------------------------------
Project Status
------------------------------------------------------------------------

Use zxcvbn_test.go to check how close to feature parity the project is.

------------------------------------------------------------------------
Acknowledgment
------------------------------------------------------------------------

Thanks to Dan Wheeler (https://github.com/lowe) for the CoffeeScript implementation
(see above.) To repeat his outside acknowledgements (which remain useful, as always):

Many thanks to Mark Burnett for releasing his 10k top passwords list:
https://xato.net/passwords/more-top-worst-passwords
and for his 2006 book,
"Perfect Passwords: Selection, Protection, Authentication"

Huge thanks to Wiktionary contributors for building a frequency list
of English as used in television and movies:
https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists

Last but not least, big thanks to xkcd :)
https://xkcd.com/936/
