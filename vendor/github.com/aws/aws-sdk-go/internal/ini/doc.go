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
//  Grammar:
//  stmt -> section | stmt'
//  stmt' -> epsilon | expr
//  expr -> value (stmt)* | equal_expr (stmt)*
//  equal_expr -> value ( ':' | '=' ) equal_expr'
//  equal_expr' -> number | string | quoted_string
//  quoted_string -> " quoted_string'
//  quoted_string' -> string quoted_string_end
//  quoted_string_end -> "
//
//  section -> [ section'
//  section' -> section_value section_close
//  section_value -> number | string_subset | boolean | quoted_string_subset
//  quoted_string_subset -> " quoted_string_subset'
//  quoted_string_subset' -> string_subset quoted_string_end
//  quoted_string_subset -> "
//  section_close -> ]
//
//  value -> number | string_subset | boolean
//  string -> ? UTF-8 Code-Points except '\n' (U+000A) and '\r\n' (U+000D U+000A) ?
//  string_subset -> ? Code-points excepted by <string> grammar except ':' (U+003A), '=' (U+003D), '[' (U+005B), and ']' (U+005D) ?
//
//  SkipState will skip (NL WS)+
//
//  comment -> # comment' | ; comment'
//  comment' -> epsilon | value
package ini
