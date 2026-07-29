package govalidator

import "regexp"

// Basic regular expressions for validating strings
const (
	CreditCard string = "^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\\d{3})\\d{11})$"
	ISBN10     string = "^(?:[0-9]{9}X|[0-9]{10})$"
	ISBN13     string = "^(?:[0-9]{13})$"
	Hexcolor   string = "^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"
	RGBcolor   string = "^rgb\\(\\s*(0|[1-9]\\d?|1\\d\\d?|2[0-4]\\d|25[0-5])\\s*,\\s*(0|[1-9]\\d?|1\\d\\d?|2[0-4]\\d|25[0-5])\\s*,\\s*(0|[1-9]\\d?|1\\d\\d?|2[0-4]\\d|25[0-5])\\s*\\)$"
	Base64     string = "^(?:[A-Za-z0-9+\\/]{4})*(?:[A-Za-z0-9+\\/]{2}==|[A-Za-z0-9+\\/]{3}=|[A-Za-z0-9+\\/]{4})$"
	SSN        string = `^\d{3}[- ]?\d{2}[- ]?\d{4}$`
	Int        string = "^(?:[-+]?(?:0|[1-9][0-9]*))$"
)

var (
	rxCreditCard = regexp.MustCompile(CreditCard)
	rxInt        = regexp.MustCompile(Int)
	rxISBN10     = regexp.MustCompile(ISBN10)
	rxISBN13     = regexp.MustCompile(ISBN13)
	rxHexcolor   = regexp.MustCompile(Hexcolor)
	rxRGBcolor   = regexp.MustCompile(RGBcolor)
	rxBase64     = regexp.MustCompile(Base64)
	rxSSN        = regexp.MustCompile(SSN)
)
