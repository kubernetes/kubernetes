// tests for both the tokenizer and parser. Parser test results could be checked tighter.
// api: [input, token-output-count, ?regex-hints, desc]
// regex-hints are for tokenizer, will tell for each token whether it might parse regex or not (parser's job)
var Tests = [

["var abc;", 4, "Variable Declaration"],
["var abc = 5;", 8, "Variable Declaration, Assignment"],
["/* */", 1, "Block Comment"],
["/** **/", 1, "JSDoc-style Comment"],
["var f = function(){;};", 13, "Assignment, Function Expression"],
["hi; // moo", 4, "Trailing Line Comment"],
["hi; // moo\n;", 6, "Trailing Line Comment, Linefeed, `;`"],
["var varwithfunction;", 4, "Variable Declaration, Identifier Containing Reserved Words, `;`"],
["a + b;", 6, "Addition/Concatenation"],

["'a'", 1, "Single-Quoted String"],
["'a';", 2, "Single-Quoted String, `;`"], // Taken from the parser test suite.

["'a\\n'", 1, "Single-Quoted String With Escaped Linefeed"],
["'a\\n';", 2, "Single-Quoted String With Escaped Linefeed, `;`"], // Taken from the parser test suite.

["\"a\"", 1, "Double-Quoted String"],
["\"a\";", 2, "Double-Quoted String, `;`"], // Taken from the parser test suite.

["\"a\\n\"", 1, "Double-Quoted String With Escaped Linefeed"],
["\"a\\n\";", 2, "Double-Quoted String With Escaped Linefeed, `;`"], // Taken from the parser test suite.

["500", 1, "Integer"],
["500;", 2, "Integer, `;`"], // Taken from the parser test suite.

["500.", 1, "Double With Trailing Decimal Point"],
["500.;", 2, "Double With Trailing Decimal Point"], // Taken from the parser test suite.

["500.432", 1, "Double With Decimal Component"],
["500.432;", 2, "Double With Decimal Component, `;`"], // Taken from the parser test suite.

[".432432", 1, "Number, 0 < Double < 1"],
[".432432;", 2, "Number, 0 < Double < 1, `;`"], // Taken from the parser test suite.

["(a,b,c)", 7, "Parentheses, Comma-separated identifiers"],
["(a,b,c);", 8, "Parentheses, Comma-separated identifiers, `;`"], // Taken from the parser test suite.

["[1,2,abc]", 7, "Array literal"],
["[1,2,abc];", 8, "Array literal, `;`"], // Taken from the parser test suite.

["{a:1,\"b\":2,c:c}", 13, "Object literal"],
["var o = {a:1,\"b\":2,c:c};", 20, "Assignment, Object Literal, `;`"], // Taken from the parser test suite.

["var x;\nvar y;", 9, "2 Variable Declarations, Multiple lines"],
["var x;\nfunction n(){ }", 13, "Variable, Linefeed, Function Declaration"],
["var x;\nfunction n(abc){ }", 14, "Variable, Linefeed, Function Declaration With One Argument"],
["var x;\nfunction n(abc, def){ }", 17, "Variable, Linefeed, Function Declaration With Multiple Arguments"],
["function n(){ \"hello\"; }", 11, "Function Declaration, Body"],

["/a/;", 2, [true, false], "RegExp Literal, `;`"],
["/a/b;", 2, [true, true], "RegExp Literal, Flags, `;`"],
["++x;", 3, "Unary Increment, Prefix, `;`"],
[" / /;", 3, [true, true, false], "RegExp, Leading Whitespace, `;`"],
["/ / / / /", 5, [true, false, false, false, true], "RegExp Containing One Space, Space, Division, Space, RegExp Containing One Space"],

// Taken from the parser test suite.

["\"var\";", 2, "Keyword String, `;`"],
["\"variable\";", 2, "String Beginning With Keyword, `;`"],
["\"somevariable\";", 2, "String Containing Keyword, `;`"],
["\"somevar\";", 2, "String Ending With Keyword, `;`"],

["var varwithfunction;", 4, "Keywords should not be matched in identifiers"],

["var o = {a:1};", 12, "Object Literal With Unquoted Property"],
["var o = {\"b\":2};", 12, "Object Literal With Quoted Property"],
["var o = {c:c};", 12, "Object Literal With Equivalent Property Name and Identifier"],

["/a/ / /b/;", 6, [true, true, false, false, true, false], "RegExp, Division, RegExp, `;`"],
["a/b/c;", 6, "Triple Division (Identifier / Identifier / Identifier)"],

["+function(){/regex/;};", 9, [false, false, false, false, false, true, false, false, false], "Unary `+` Operator, Function Expression Containing RegExp and Semicolon, `;`"],

// Line Terminators.
["\r\n", 1, "CRLF Line Ending = 1 Linefeed"],
["\r", 1, "CR Line Ending = 1 Linefeed"],
["\n", 1, "LF Line Ending = 1 Linefeed"],
["\r\n\n\u2028\u2029\r", 5, "Various Line Terminators"],

// Whitespace.
["a \t\u000b\u000c\u00a0\uFFFFb", 8, "Whitespace"],

// Comments.
["//foo!@#^&$1234\nbar;", 4, "Line Comment, Linefeed, Identifier, `;`"],
["/* abcd!@#@$* { } && null*/;", 2, "Single-Line Block Comment, `;`"],
["/*foo\nbar*/;", 2, "Multi-Line Block Comment, `;`"],
["/*x*x*/;", 2, "Block Comment With Asterisks, `;`"],
["/**/;", 2, "Empty Comment, `;`"],

// Identifiers.
["x;", 2, "Single-Character Identifier, `;`"],
["_x;", 2, "Identifier With Leading `_`, `;`"],
["xyz;", 2, "Identifier With Letters Only, `;`"],
["$x;", 2, "Identifier With Leading `$`, `;`"],
["x5;", 2, "Identifier With Number As Second Character, `;`"],
["x_y;", 2, "Identifier Containing `_`, `;`"],
["x+5;", 4, "Identifier, Binary `+` Operator, Identifier, `;`"],
["xyz123;", 2, "Alphanumeric Identifier, `;`"],
["x1y1z1;", 2, "Alternating Alphanumeric Identifier, `;`"],
["foo\\u00d8bar;", 2, "Identifier With Unicode Escape Sequence (`\\uXXXX`), `;`"],
["f\u00d8\u00d8bar;", 2, "Identifier With Embedded Unicode Character"],

// Numbers.
["5;", 2, "Integer, `;`"],
["5.5;", 2, "Double, `;`"],
["0;", 2, "Integer Zero, `;`"],
["0.0;", 2, "Double Zero, `;`"],
["0.001;", 2, "0 < Decimalized Double < 1, `;`"],
["1.e2;", 2, "Integer With Decimal and Exponential Component (`e`), `;`"],
["1.e-2;", 2, "Integer With Decimal and Negative Exponential Component, `;`"],
["1.E2;", 2, "Integer With Decimal and Uppercase Exponential Component (`E`), `;`"],
["1.E-2;", 2, "Integer With Decimal and Uppercase Negative Exponential Component, `;`"],
[".5;", 2, "0 < Double < 1, `;`"],
[".5e3;", 2, "(0 < Double < 1) With Exponential Component"],
[".5e-3;", 2, "(0 < Double < 1) With Negative Exponential Component"],
["0.5e3;", 2, "(0 < Decimalized Double < 1) With Exponential Component"],
["55;", 2, "Two-Digit Integer, `;`"],
["123;", 2, "Three-Digit Integer, `;`"],
["55.55;", 2, "Two-Digit Double, `;`"],
["55.55e10;", 2, "Two-Digit Double With Exponential Component, `;`"],
["123.456;", 2, "Three-Digit Double, `;`"],
["1+e;", 4, "Additive Expression, `;`"],
["0x01;", 2, "Hexadecimal `1` With 1 Leading Zero, `;`"],
["0xcafe;", 2, "Hexadecimal `51966`, `;`"],
["0x12345678;", 2, "Hexadecimal `305419896`, `;`"],
["0x1234ABCD;", 2, "Hexadecimal `305441741` With Uppercase Letters, `;`"],
["0x0001;", 2, "Hexadecimal `1` with 3 Leading Zeros, `;`"],

// Strings.
["\"foo\";", 2, "Multi-Character Double-Quoted String, `;`"],
["\"a\\n\";", 2, "Double-Quoted String Containing Linefeed, `;`"],
["\'foo\';", 2, "Single-Quoted String, `;`"],
["'a\\n';", 2, "Single-Quoted String Containing Linefeed, `;`"],
["\"x\";", 2, "Single-Character Double-Quoted String, `;`"],
["'';", 2, "Empty Single-Quoted String, `;`"],
["\"foo\\tbar\";", 2, "Double-Quoted String With Tab Character, `;`"],
["\"!@#$%^&*()_+{}[]\";", 2, "Double-Quoted String Containing Punctuators, `;`"],
["\"/*test*/\";", 2, "Double-Quoted String Containing Block Comment, `;`"],
["\"//test\";", 2, "Double-Quoted String Containing Line Comment, `;`"],
["\"\\\\\";", 2, "Double-Quoted String Containing Reverse Solidus, `;`"],
["\"\\u0001\";", 2, "Double-Quoted String Containing Numeric Unicode Escape Sequence, `;`"],
["\"\\uFEFF\";", 2, "Double-Quoted String Containing Alphanumeric Unicode Escape Sequence, `;`"],
["\"\\u10002\";", 2, "Double-Quoted String Containing 5-Digit Unicode Escape Sequence, `;`"],
["\"\\x55\";", 2, "Double-Quoted String Containing Hex Escape Sequence, `;`"],
["\"\\x55a\";", 2, "Double-Quoted String Containing Hex Escape Sequence and Additional Character, `;`"],
["\"a\\\\nb\";", 2, "Double-Quoted String Containing Escaped Linefeed, `;`"],
["\";\"", 1, "Double-Quoted String Containing `;`"],
["\"a\\\nb\";", 2, "Double-Quoted String Containing Reverse Solidus and Linefeed, `;`"],
["'\\\\'+ ''", 4, "Single-Quoted String Containing Reverse Solidus, `+`, Empty Single-Quoted String"],

// `null`, `true`, and `false`.
["null;", 2, "`null`, `;`"],
["true;", 2, "`true`, `;`"],
["false;", 2, "`false`, `;`"],

// RegExps
["/a/;", 2, [true, true], "Single-Character RegExp, `;`"],
["/abc/;", 2, [true, true], "Multi-Character RegExp, `;`"],
["/abc[a-z]*def/g;", 2, [true, true], "RegExp Containing Character Range and Quantifier, `;`"],
["/\\b/;", 2, [true, true], "RegExp Containing Control Character, `;`"],
["/[a-zA-Z]/;", 2, [true, true], "RegExp Containing Extended Character Range, `;`"],
["/foo(.*)/g;", 2, [true, false], "RegExp Containing Capturing Group and Quantifier, `;`"],

// Array Literals.
["[];", 3, "Empty Array, `;`"],
["[\b\n\f\r\t\x20];", 9, "Array Containing Whitespace, `;`"],
["[1];", 4, "Array Containing 1 Element, `;`"],
["[1,2];", 6, "Array Containing 2 Elements, `;`"],
["[1,2,,];", 8, "Array Containing 2 Elisions, `;`"],
["[1,2,3];", 8, "Array Containing 3 Elements, `;`"],
["[1,2,3,,,];", 11, "Array Containing 3 Elisions, `;`"],

// Object Literals.
["({x:5});", 8, "Object Literal Containing 1 Member; `;`"],
["({x:5,y:6});", 12, "Object Literal Containing 2 Members, `;`"],
["({x:5,});", 9, "Object Literal Containing 1 Member and Trailing Comma, `;`"],
["({if:5});", 8, "Object Literal Containing Reserved Word Property Name, `;`"],
["({ get x() {42;} });", 17, "Object Literal Containing Getter, `;`"],
["({ set y(a) {1;} });", 18, "Object Literal Containing Setter, `;`"],

// Member Expressions.
["o.m;", 4, "Dot Member Accessor, `;`"],
["o['m'];", 5, "Square Bracket Member Accessor, `;`"],
["o['n']['m'];", 8, "Nested Square Bracket Member Accessor, `;`"],
["o.n.m;", 6, "Nested Dot Member Accessor, `;`"],
["o.if;", 4, "Dot Reserved Property Name Accessor, `;`"],

// Function Calls.
["f();", 4, "Function Call Operator, `;`"],
["f(x);", 5, "Function Call Operator With 1 Argument, `;`"],
["f(x,y);", 7, "Function Call Operator With Multiple Arguments, `;`"],
["o.m();", 6, "Dot Member Accessor, Function Call, `;`"],
["o['m']();", 7, "Square Bracket Member Accessor, Function Call, `;`"],
["o.m(x);", 7, "Dot Member Accessor, Function Call With 1 Argument, `;`"],
["o['m'](x);", 8, "Square Bracket Member Accessor, Function Call With 1 Argument, `;`"],
["o.m(x,y);", 9, "Dot Member Accessor, Function Call With 2 Arguments, `;`"],
["o['m'](x,y);", 10, "Square Bracket Member Accessor, Function Call With 2 Arguments, `;`"],
["f(x)(y);", 8, "Nested Function Call With 1 Argument Each, `;`"],
["f().x;", 6, "Function Call, Dot Member Accessor, `;`"],

// `eval` Function.
["eval('x');", 5, "`eval` Invocation With 1 Argument, `;`"],
["(eval)('x');", 7, "Direct `eval` Call Example, `;`"],
["(1,eval)('x');", 9, "Indirect `eval` Call Example, `;`"],
["eval(x,y);", 7, "`eval` Invocation With 2 Arguments, `;`"],

// `new` Operator.
["new f();", 6, "`new` Operator, Function Call, `;`"],
["new o;", 4, "`new` Operator, Identifier, `;`"],
["new o.m;", 6, "`new` Operator, Dot Member Accessor, `;`"],
["new o.m(x);", 9, "`new` Operator, Dot Member Accessor, Function Call With 1 Argument, `;`"],
["new o.m(x,y);", 11, "``new` Operator, Dot Member Accessor, Function Call With 2 Arguments , `;`"],

// Prefix and Postfix Increment.
["++x;", 3, "Prefix Increment, Identifier, `;`"],
["x++;", 3, "Identifier, Postfix Increment, `;`"],
["--x;", 3, "Prefix Decrement, Identifier, `;`"],
["x--;", 3, "Postfix Decrement, Identifier, `;`"],
["x ++;", 4, "Identifier, Space, Postfix Increment, `;`"],
["x /* comment */ ++;", 6, "Identifier, Block Comment, Postfix Increment, `;`"],
["++ /* comment */ x;", 6, "Prefix Increment, Block Comment, Identifier, `;`"],

// Unary Operators.
["delete x;", 4, "`delete` Operator, Space, Identifier, `;`"],
["void x;", 4, "`void` Operator, Space, Identifier, `;`"],
["typeof x;", 4, "`typeof` Operator, Space, Identifier, `;`"],
["+x;", 3, "Unary `+` Operator, Identifier, `;`"],
["-x;", 3, "Unary Negation Operator, Identifier, `;`"],
["~x;", 3, "Bitwise NOT Operator, Identifier, `;`"],
["!x;", 3, "Logical NOT Operator, Identifier, `;`"],

// Comma Operator.
["x, y;", 5, "Comma Operator"],

// Miscellaneous.
["new Date++;", 5, "`new` Operator, Identifier, Postfix Increment, `;`"],
["+x++;", 4, "Unary `+`, Identifier, Postfix Increment, `;`"],

// Expressions.
["1 * 2;", 6, "Integer, Multiplication, Integer, `;`"],
["1 / 2;", 6, "Integer, Division, Integer, `;`"],
["1 % 2;", 6, "Integer, Modulus, Integer, `;`"],
["1 + 2;", 6, "Integer, Addition, Integer, `;`"],
["1 - 2;", 6, "Integer, Subtraction, Integer, `;`"],
["1 << 2;", 6, "Integer, Bitwise Left Shift, Integer, `;`"],
["1 >>> 2;", 6, "Integer, Bitwise Zero-fill Right Shift, Integer, `;`"],
["1 >> 2;", 6, "Integer, Bitwise Sign-Propagating Right Shift, Integer, `;`"],
["1 * 2 + 3;", 10, "Order-of-Operations Expression, `;`"],
["(1+2)*3;", 8, "Parenthesized Additive Expression, Multiplication, `;`"],
["1*(2+3);", 8, "Multiplication, Parenthesized Additive Expression, `;`"],
["x<y;", 4, "Less-Than Relational Operator, `;`"],
["x>y;", 4, "Greater-Than Relational Operator, `;`"],
["x<=y;", 4, "Less-Than-or-Equal-To Relational Operator, `;`"],
["x>=y;", 4, "Greater-Than-or-Equal-To Relational Operator, `;`"],
["x instanceof y;", 6, "`instanceof` Operator, `;`"],
["x in y;", 6, "`in` Operator, `;`"],
["x&y;", 4, "Bitwise AND Operator, `;`"],
["x^y;", 4, "Bitwise XOR Operator, `;`"],
["x|y;", 4, "Bitwise OR Operator, `;`"],
["x+y<z;", 6, "Addition, Less-Than Relational, `;`"],
["x<y+z;", 6, "Less-Than Relational, Addition, `;`"],
["x+y+z;", 6, "Additive Expression With Three Identifiers, `;`"],
["x&y|z;", 6, "Bitwise AND-OR Expression With Three Identifiers, `;`"],
["x&&y;", 4, "Logical AND Operator, `;`"],
["x||y;", 4, "Logical OR Operator, `;`"],
["x&&y||z;", 6, "Logical AND-OR Expression With Three Identifiers, `;`"],
["x||y&&z;", 6, "Logical OR-AND Expression With Three Identifiers, `;`"],
["x<y?z:w;", 8, "Ternary Operator Expression With Four Identifiers, `;`"],

// Assignment Operators.
["x = y;", 6, "Assignment, `;`"],
["x >>>= y;", 6, "Bitwise Zero-Fill Right Shift Assignment, `;`"],
["x <<= y;", 6, "Bitwise Left Shift Assignment, `;`"],
["x += y;", 6, "Additive Assignment, `;`"],
["x -= y;", 6, "Subtractive Assignment, `;`"],
["x *= y;", 6, "Multiplicative Assignment, `;`"],
["x /= y;", 6, "Divisive Assignment, `;`"],
["x %= y;", 6, "Modulus Assignment, `;`"],
["x >>= y;", 6, "Bitwise Sign-Propagating Right Shift Assignment, `;`"],
["x &= y;", 6, "Bitwise AND Assignment, `;`"],
["x ^= y;", 6, "Bitwise XOR Assignment, `;`"],
["x |= y;", 6, "Bitwise OR Assignment, `;`"],

// Blocks.
["{};", 3, "Empty Block, `;`"],
["{x;};", 5, "Block Containing 1 Identifier, `;`"],
["{x;y;};", 7, "Block Containing 2 Identifiers, `;`"],

// Variable Declarations.
["var abc;", 4, "Variable Declaration"],
["var x,y;", 6, "Comma-Separated Variable Declarations, `;`"],
["var x=1,y=2;", 10, "Comma-Separated Variable Initializations, `;`"],
["var x,y=2;", 8, "Variable Declaration, Variable Initialization, `;`"],

// Empty Statements.
[";", 1, "Empty Statement"],
["\n;", 2, "Linefeed, `;`"],

// Expression Statements.
["x;", 2, "Identifier, `;`"],
["5;", 2, "Integer, `;`"],
["1+2;", 4, "Additive Statement, `;`"],

// `if...else` Statements.
["if (c) x; else y;", 13, "Space-Delimited `if...else` Statement"],
["if (c) x;", 8, "Space-Delimited `if` Statement, `;`"],
["if (c) {} else {};", 14, "Empty Block-Delimited `if...else` Statement"],
["if (c1) if (c2) s1; else s2;", 19, "Nested `if...else` Statement Without Dangling `else`"],

// `while` and `do...while` Loops.
["do s; while (e);", 11, "Space-Delimited `do...while` Loop"],
["do { s; } while (e);", 15, "Block-Delimited `do...while` Loop"],
["while (e) s;", 8, "Space-Delimited `while` Loop"],
["while (e) { s; };", 13, "Block-Delimited `while` Loop"],

// `for` and `for...in` Loops.
["for (;;) ;", 8, "Infinite Space-Delimited `for` Loop"],
["for (;c;x++) x;", 12, "`for` Loop: Empty Initialization Condition; Space-Delimited Body"],
["for (i;i<len;++i){};", 15, "Empty `for` Loop: Empty; Initialization, Test, and Increment Conditions Specified"],
["for (var i=0;i<len;++i) {};", 20, "Empty `for` Loop: Variable Declaration in Initialization Condition"],
["for (var i=0,j=0;;){};", 18, "`Empty for` Loop: Empty Test and Increment Conditions"],
["for ((x in b); c; u) {};", 21, "Empty `for` Loop: `in` Expression in Initialization Condition"],
["for (x in a);", 10, "Empty `for...in` Loop"],
["for (var x in a){};", 14, "Empty `for...in` Loop: Variable Declaration in Loop Header"],
["for (var x=5 in a) {};", 17, "Empty `for...in` Loop: Variable Initialization in Assignment Header"],
["for (var x = a in b in c) {};", 23, "Empty `for...in` Loop: Multiple `in` Expressions in Header"],
["for (var x=function(){a+b;}; a<b; ++i) some;", 29, "`for` Loop: Function Expression in Initialization Condition"],
["for (var x=function(){for (x=0; x<15; ++x) alert(foo); }; a<b; ++i) some;", 48, "for.in` Loop: Function Expression in Initialization Condition Containing `for` Loop"],
["for (x in a, b, c);", 16, "`for...in` With Multiple Comma-Separated Object References"],

// Flow of Control: `continue`, `break`, and `return` Statements.
["continue;", 2, "`continue` Statement"],
["continue label;", 4, "`continue` Statement With Identifier Label"],
["break;", 2, "`break` Statement"],
["break somewhere;", 4, "`break` Statement With Identifier Label"],
["continue /* comment */ ;", 5, "`continue` Statement, Block Comment, `;`"],
["continue \n;", 4, "`continue` Statement, Space, Linefeed, `;`"],
["return;", 2, "`return` Statement"],
["return 0;", 4, "`return` Statement, Integer, `;`"],
["return 0 + \n 1;", 10, "`return` Statement, Additive Expression Containing Linefeed, `;`"],

// `with` Statement.
["with (e) s;", 8, "`with` Statement, `;`"],

// `switch` Statement.
["switch (e) { case x: s; };", 18, "`switch` Statement With 1 `case`"],
["switch (e) { case x: s1;s2; default: s3; case y: s4; };", 34, "`switch` Statement: `case`, `default`, `case`"],
["switch (e) { default: s1; case x: s2; case y: s3; };", 32, "`switch` Statement: `default`, `case`, `case`"],
["switch (e) { default: s; };", 16, "`switch` Statement With `default` Case Only"],
["switch (e) { case x: s1; case y: s2; };", 26, "`switch` Statement With 2 `case`s"],

// Labels.
["foo : x;", 6, "Label (Identifier, Colon, Reference), `;`"],

// `throw` Statement.
["throw x;", 4, "Throw Statement, `;`"],
["throw x\n;", 5, "Throw Statement, Linefeed, `;`"],
["throw x", 3, "Throw Statement, No `;` (Safari 2 Case)"],

// `try...catch...finally` Statement.
["try { s1; } catch (e) { s2; };", 22, "`try...catch` Statement"],
["try { s1; } finally { s2; };", 18, "`try...finally` Statement"],
["try { s1; } catch (e) { s2; } finally { s3; };", 31, "`try...catch...finally` Statement"],

// `debugger` Statement.
["debugger;", 2, "`debugger` Statement"],

// Function Declarations.
["function f() { x; y; };", 16, "Named Function Declaration With Body"],
["function f(x) { e; return x; };", 19, "Named Function Declaration With Argument and `return`"],
["function f(x,y) { var z; return x; };", 23, "Named Function Declaration With 2 Arguments, Variable Declaration, and `return`"],

// Function Expressions.
["(function empty() {;});", 12, "Parenthesized Empty Named Function Expression"],
["(function (x) {; });", 13, "Parenthesized Empty Function Expression"],
["(function f(x) { return x; });", 18, "Named Function Expression"],

// ECMAScript Programs.
["var x; function f(){;}; null;", 17, "Variable Declaration, Function Declaration, `null`, `;`"],
[";;", 2, "Program: 2 Empty Statements"],
["{ x; y; z; }", 12, "Program: Block Comprising Semicolon-Delimited Identifiers"],
["function f(){ function g(){;}};", 17, "Program: Nested Function Declaration"],
["x;\n/*foo*/\n\t;", 7, "Program: Identifier, Linefeed, Block Comment, Linefeed"],

// Automatic Semicolon Insertion
["continue \n foo;", 6, "Restricted Production: `continue` Statement"],
["break \n foo;", 6, "Restricted Production: `break` Statement"],
["return\nfoo;", 4, "Restricted Production: `return` Statement"],
["throw\nfoo;", 4, "Restricted Production: `throw` Statement"],
["var x; { 1 \n 2 } 3", 16, "Classic Automatic Semicolon Insertion Case"],
["ab \t /* hi */\ncd", 7, "Automatic Semicolon Insertion: Block Comment"],
["ab/*\n*/cd", 3, "Automatic Semicolon Insertion Triggered by Multi-Line Block Comment"],
["continue /* wtf \n busta */ foo;", 6, "Automatic Semicolon Insertion: `continue` Statement Preceding Multi-Line Block Comment"],
["function f() { s }", 11, "Automatic Semicolon Insertion: Statement Within Function Declaration"],
["function f() { return }", 11, "Automatic Semicolon Insertion: `return` Statement Within Function Declaration"],

// Strict Mode.
["\"use strict\"; 'bla'\n; foo;", 9, "Double-Quoted Strict Mode Directive, Program"],
["'use strict'; \"bla\"\n; foo;", 9, "Single-Quoted Strict Mode Directive, Program"],
["(function() { \"use strict\"; 'bla';\n foo; });", 20, "Strict Mode Directive Within Function"],
["\"use\\n strict\";", 2, "Invalid Strict Mode Directive Containing Linefeed"],
["foo; \"use strict\";", 5, "Invalid Strict Mode Directive Within Program"],

// Taken from http://es5conform.codeplex.com.
["\"use strict\"; var o = { eval: 42};", 17, "Section 8.7.2: `eval` object property name is permitted in strict mode"],
["({foo:0,foo:1});", 12, "Duplicate object property name is permitted in non-strict mode"],
["function foo(a,a){}", 10, "Duplicate argument name is permitted in non-strict mode"],
["(function foo(eval){})", 10, "`eval` argument name is permitted in non-strict mode"],
["(function foo(arguments){})", 10, "`arguments` argument name is permitted in non-strict mode"],

// Empty Programs.
["", 0, "Empty Program"],
["// test", 1, "Line Comment"],
["//test\n", 2, "Line Comment, Linefeed"],
["\n// test", 2, "Linefeed, Line Comment"],
["\n// test\n", 3, "Linefeed, Line Comment, Linefeed"],
["/* */", 1, "Single-Line Block Comment"],
["/*\ns,fd\n*/", 1, "Multi-Line Block Comment"],
["/*\ns,fd\n*/\n", 2, "Block Comment Containing Linefeeds, Linefeed"],
["  \t", 3, "Spaces and Tabs"],
["  /*\nsmeh*/\t\n   ", 8, "Spaces, Block Comment, Linefeeds, and Tabs"],

// Trailing Whitespace.
["a  ", 3, "Trailing Space Characters"],
["a /* something */", 3, "Trailing Block Comment"],
["a\n\t// hah", 4, "Trailing Linefeed, Tab, and Line Comment"],
["/abc/de//f", 2, [true, true], "RegExp With Flags, Trailing Line Comment"],
["/abc/de/*f*/\n\t", 4, [true, true, true, true], "RegExp With Flags, Trailing Block Comment, Newline, Tab"],

// Regression Tests.
["for (x;function(){ a\nb };z) x;", 21, "`for` Loop: Test Condition Contains Function Body With No Terminating `;`"],
["c=function(){return;return};", 11, "Function Body: Two `return` Statements; No Terminating `;`"],
["d\nd()", 5, "Identifier, Newline, Function Call"],
["for(;;){x=function(){}}", 14, "Function Expression in `for` Loop Body"],
["for(var k;;){}", 10, "`for` Loop Header: Variable Declaration, Empty Test and Increment Conditions"],
["({get foo(){ }})", 12, "Empty Getter"],
["\nreturnr", 2, "Linefeed, Identifier Beginning With `return`"],
["/ // / /", 4, [true, false, false, true], "RegExp Containing One Space, Division Operator, Space, RegExp Containing One Space"],
["trimRight = /\\s+$/;", 6, [false, false, false, false, true, false], "Typical `trimRight` RegExp"],
["trimLeft = /^\\s+/;\n\ttrimRight = /\\s+$/;", 14, [false, false, false, false, true, false, false, false, false, false, false, false, true, false], "`trimLeft` and `trimRight` RegExps"],
["\n\t// Used for trimming whitespace\n\ttrimLeft = /^\\s+/;\n\ttrimRight = /\\s+$/;\t\n", 21, [false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, true, false, false], "Annotated `trimLeft` and `trimRight` RegExps"],

["/[\\/]/;", 2, [true, false], "RegExp: Escaped `/` Within Character Class"],
["/[/]/;", 2, [true, false], "RegExp: Escaped Trailing Character Class End (Valid in ES 5; Invalid in ES 3)"],

["({get:5});", 8, "`get` Used As Standard Property Name"],
["({set:5});", 8, "`get` Used As Standard Property Name"],
["l !== \"px\" && (d.style(h, c, (k || 1) + l), j = (k || 1) / f.cur() * j, d.style(h, c, j + l)), i[1] && (k = (i[1] === \"-=\" ? -1 : 1) * k + j), f.custom(j, k, l)", 131, "Regression Test: RegExp/Division"],

["(/\'/g, \'\\\\\\\'\') + \"'\";'", 14, "Regression Test: Confusing Escape Character Sequence"],
["/abc\//no_comment", 3, [true, false, false], "RegExp Followed By Line Comment"],
["a: b; c;", 8, "ASI Regression Test: Labeled Identifier, `;`, Identifier, `;`"],
["var x; function f(){ x; function g(){}}", 23, "Function Declaration Within Function Body"],
["if (x) { break }", 11, "ASI: `if` Statement, `break`"],
["x.hasOwnProperty()", 5, "Regression Test: Object Property Named `hasOwnProperty`"],
["(x) = 5", 7, "LHS of Expression Contains Grouping Operator"],
["(x,x) = 5", 9, "Syntactically Valid LHS Grouping Operator (Expression Will Produce A `ReferenceError` When Interpreted)"],
["switch(x){case 1:}", 10, "Single-`case` `switch` Statement Without Body"],
["while (x) { ++a\t}", 12, "Prefix Increment Operator, Tab Character Within `while` Loop"],

["{break}", 3, "ASI: `break`"],
["{continue}", 3, "ASI: `continue`"],
["{return}", 3, "ASI: `return`"],
["{continue a}", 5, "ASI: `continue`, Identifier"],
["{break b}", 5, "ASI: `break`, Identifier"],
["{return c}", 5, "ASI: `return`, Identifier"],

["this.charsX = Gui.getSize(this.textarea).w / this.fontSize.w;", 25, "Complex Division Not Treated as RegExp"],
["(x)/ (y);", 9, "Parenthesized Dividend, Division Operator, Space, Parenthesized Divisor"],
["/^(?:\\/(?![*\\n\\/])(?:\\[(?:\\\\.|[^\\]\\\\\\n])*\\]|\\\\.|[^\\[\\/\\\\\\n])+\\/[gim]*)$/", 1, [true], "Complex RegExp for Matching RegExps"],
["({a:b}[ohi].iets()++);", 16, "Object Literal With 1 Member, Square Bracket Member Accessor, Dot Member Accessor, Function Call, Postfix Increment"]

];