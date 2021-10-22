package ini

import (
	"testing"
)

const (
	section = `[default]
region = us-west-2
credential_source = Ec2InstanceMetadata
s3 =
	foo=bar
	bar=baz
output = json

[assumerole]
output = json
region = us-west-2
`
)

func BenchmarkINIParser(b *testing.B) {
	for i := 0; i < b.N; i++ {
		ParseBytes([]byte(section))
	}
}

func BenchmarkTokenize(b *testing.B) {
	lexer := iniLexer{}
	for i := 0; i < b.N; i++ {
		lexer.tokenize([]byte(section))
	}
}
