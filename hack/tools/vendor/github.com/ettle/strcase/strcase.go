package strcase

// ToSnake returns words in snake_case (lower case words with underscores).
func ToSnake(s string) string {
	return convertWithoutInitialisms(s, '_', LowerCase)
}

// ToGoSnake returns words in snake_case (lower case words with underscores).
//
// Respects Go's common initialisms (e.g. http_response -> HTTP_response).
func ToGoSnake(s string) string {
	return convertWithGoInitialisms(s, '_', LowerCase)
}

// ToSNAKE returns words in SNAKE_CASE (upper case words with underscores).
// Also known as SCREAMING_SNAKE_CASE or UPPER_CASE.
func ToSNAKE(s string) string {
	return convertWithoutInitialisms(s, '_', UpperCase)
}

// ToKebab returns words in kebab-case (lower case words with dashes).
// Also known as dash-case.
func ToKebab(s string) string {
	return convertWithoutInitialisms(s, '-', LowerCase)
}

// ToGoKebab returns words in kebab-case (lower case words with dashes).
// Also known as dash-case.
//
// Respects Go's common initialisms (e.g. http-response -> HTTP-response).
func ToGoKebab(s string) string {
	return convertWithGoInitialisms(s, '-', LowerCase)
}

// ToKEBAB returns words in KEBAB-CASE (upper case words with dashes).
// Also known as SCREAMING-KEBAB-CASE or SCREAMING-DASH-CASE.
func ToKEBAB(s string) string {
	return convertWithoutInitialisms(s, '-', UpperCase)
}

// ToPascal returns words in PascalCase (capitalized words concatenated together).
// Also known as UpperPascalCase.
func ToPascal(s string) string {
	return convertWithoutInitialisms(s, 0, TitleCase)
}

// ToGoPascal returns words in PascalCase (capitalized words concatenated together).
// Also known as UpperPascalCase.
//
// Respects Go's common initialisms (e.g. HttpResponse -> HTTPResponse).
func ToGoPascal(s string) string {
	return convertWithGoInitialisms(s, 0, TitleCase)
}

// ToCamel returns words in camelCase (capitalized words concatenated together, with first word lower case).
// Also known as lowerCamelCase or mixedCase.
func ToCamel(s string) string {
	return convertWithoutInitialisms(s, 0, CamelCase)
}

// ToGoCamel returns words in camelCase (capitalized words concatenated together, with first word lower case).
// Also known as lowerCamelCase or mixedCase.
//
// Respects Go's common initialisms, but first word remains lowercased which is
// important for code generator use cases (e.g. toJson -> toJSON, httpResponse
// -> httpResponse).
func ToGoCamel(s string) string {
	return convertWithGoInitialisms(s, 0, CamelCase)
}

// ToCase returns words in given case and delimiter.
func ToCase(s string, wordCase WordCase, delimiter rune) string {
	return convertWithoutInitialisms(s, delimiter, wordCase)
}

// ToGoCase returns words in given case and delimiter.
//
// Respects Go's common initialisms (e.g. httpResponse -> HTTPResponse).
func ToGoCase(s string, wordCase WordCase, delimiter rune) string {
	return convertWithGoInitialisms(s, delimiter, wordCase)
}
