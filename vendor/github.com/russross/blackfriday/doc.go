// Package blackfriday is a markdown processor.
//
// It translates plain text with simple formatting rules into an AST, which can
// then be further processed to HTML (provided by Blackfriday itself) or other
// formats (provided by the community).
//
// The simplest way to invoke Blackfriday is to call the Run function. It will
// take a text input and produce a text output in HTML (or other format).
//
// A slightly more sophisticated way to use Blackfriday is to create a Markdown
// processor and to call Parse, which returns a syntax tree for the input
// document. You can leverage Blackfriday's parsing for content extraction from
// markdown documents. You can assign a custom renderer and set various options
// to the Markdown processor.
//
// If you're interested in calling Blackfriday from command line, see
// https://github.com/russross/blackfriday-tool.
package blackfriday
