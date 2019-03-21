// Package ini is an LL(1) parser for configuration files.
//
//	Example:
//	sections, err := ini.OpenFile("/path/to/file")
//	if err != nil {
//		panic(err)
//	}
//
//	profile := "foo"
//	section, ok := sections.GetSection(profile)
//	if !ok {
//		fmt.Printf("section %q could not be found", profile)
//	}
//
// Below is the BNF that describes this parser
//	Grammar:
//	stmt -> value stmt'
//	stmt' -> epsilon | op stmt
//	value -> number | string | boolean | quoted_string
//
//	section -> [ section'
//	section' -> value section_close
//	section_close -> ]
//
//	SkipState will skip (NL WS)+
//
//	comment -> # comment' | ; comment'
//	comment' -> epsilon | value
package ini
