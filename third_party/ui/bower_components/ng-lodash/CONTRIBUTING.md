# Contributing

This document outlines general guidelines, best practices and coding style rules
for developers contributing to the project.

## Contents

- [Pull requests](#pull-requests)
- [Bugs](#bugs)
- [Features](#features)
- [Node dependencies](#node-dependencies)
- [Frontend dependencies](#frontend-dependencies)
- [Using grunt](#grunt)
- [JS Guidelines](#js)

## Pull requests

If you fixed or added something useful to the project, you can send a
pull request. It will be reviewed by maintainer and accepted, or commented for
rework, or declined.

## Bugs

If you found an error, typo or any other flaw in the project, please report
about it using GitHub Issues. The more details you provide, the easier it could
be reproduced and the faster it could be fixed. Unfortunately, sometimes the
bug can only be reproduced in your project or in your environment, so
maintainers cannot reproduce it. In this case we believe you can fix the bug
and send us the fix.

## Features

If you've got an idea about a new feature, it's most likely that you'll have to
implement it on your own. If you cannot implement the feature, but it is very
important, you can create an issue at GitHub, but expect it to be declined by
the maintainer.

## Node dependencies

This project's build process uses Node npm modules defined via the
`package.json` file.

To add the latest version of new npm package to the project and automatically
append it to `package.json` run:

```
npm install <pkg> --save-dev
```

If new packages have been added by other contributors then run:

```
npm update
```

To update all your local packages to the latest possible versions as constrained
by `package.json` then run:

```
npm install
```

### JS

#### General top level guidelines

- Don't skip semicolons
- 4 spaces for indentation
- Single quotes for strings
- Use `{}` instead of `new Object();`
- Use `[]` instead of `new Array();`
- Don't use single line `if` statements.
- 80 chars maximum line length.
- Use inline comments only where needed to document dense code, code should
otherwise be clearly comprehensible without redundant commenting
- Add a whitespace char when writing inline comments, i.e `// Good` vs `//Bad`.

#### Variable naming

- Always use `var` to declare variables.
- Use only one `var` for multiple variables.
- Define variables before use.
- Define constants in CAPS.

```js
// Good
var aVariable = 0,
    aSecondVariable = null,
    A_CONSTANT = 'constant';

// Bad
var a_variable = 0;
var a_second_variable = null;
```

#### Whitespace

Our JavaScript code is compressed so don't try and save on space when writing
code, focus on readability by trying to (however do not add unnecessary spaces).

 - Use one (and only one) blank line to separate logical sets of code.
 - Add a whitespace character after keywords like `if` and after right
parenthesis.
 - Add a whitespace character after commas.
 - Leave a line before a function or `if` statement.

```js
if (foo === 'foo') {
    var array = ['one', 'two', 'three'];
    bar(array);

    foo = 'bar';
}

function (arg1, arg2) {
    var foo = arg1;
    bar(foo);
}
```

- Don't add superfluos newlines when writing short/simple indented blocks:

```js
// Bad

if ( foo ) {

    bar();
}

function ( baz ) {

    bar();
}
```

- Do add newlines in blocks when the opening part of the block has wrapped onto
a newline due to the 80 char line length limit:

```js
// Good

angular.module('foo')
    .controller('FooCtrl', function ($scope, $someReallyLongServiceName,
        $anotherService, $somethingElse) {

        $scope.bar = 'baz';
        $scope.qux = 'quux';
    });
```

#### JSHint

This repo uses [JSHint](http://www.jshint.com/about/), defined in
[.jshintrc](../../.jshintrc). Files are checked on save, and errors are visible
in the console. Optionally this config file can also be used by your code
editor.

The following guidelines are checked by JSHint.

- Four spaces for indentation, never tabs.
- Lines no longer than 80 Characters (including comments).
- Use single `'` quotes for strings.
- Use `===` and `!==` over `==` and `!=`
- Use `camelCase` for variables.
- Don't leave trailing whitespace.

#### JS Code Style (JSCS)

This repo also uses [JSCS](https://github.com/mdevils/node-jscs), defined in
[.jscsrc](../../.jscsrc). Files are checked on save, and errors are visible in
the console. Optionally this config file can also be used by your code editor.

The following rules have been defined.

##### `requireCurlyBraces`

Use curly braces after statements.

```js
// Good
if (x) {
    x++;
}

// Bad
if (x) x++;
```

##### `requireSpaceAfterKeywords`

Use spaces after keywords:

```js
// Good
if (x) {
    x++;
}

// Bad
if(x) {
    x++;
}
```

##### `requireSpaceBeforeBlockStatements`

Use spaces after parenthesis, before curly brace:

```js
// Good
if (x) {
    x++;
}

// Bad
if (x){
    x++;
}
```

##### `requireSpacesInConditionalExpression`

Use spaces around conditional expressions:

```js
// Good
var a = b ? c : d;

// Bad
var a=b?c:d;
```

##### `requireSpacesInFunctionExpression`


##### `requireSpacesInAnonymousFunctionExpression`


##### `requireSpacesInNamedFunctionExpression`


##### `requireSpacesInFunctionDeclaration`

Use spaces before and after parenthesis:

```js
// Good
function a () {}

// Bad
function a() {}
function a (){}
```

##### `requireMultipleVarDecl`

Use one `var` statement for multiple variables:

```js
// Good
var x = 1,
    y = 2;

// Bad
var x = 1;
var y = 2;
```

##### `requireBlocksOnNewline`

Use new lines for multiple statements

```js
// Good
if (true) {
    doSomething();
    doSomethingElse();
}

// Bad
if (true) { doSomething(); doSomethingElse(); }
```

##### `disallowEmptyBlocks`

Don't use empty blocks

```js
// Good
if ( a === b ) {
    c = d;
}

// Bad
if ( a !== b ) {

} else {
    c = d;
}
```

##### `disallowSpacesInsideObjectBrackets`

Don't use a space after an opening or before a closing curly bracket

```js
// Good
var x = {a: 1};

// Bad
var x = { a: 1 };
```

##### `disallowSpacesInsideArrayBrackets`

Don't use a space after an opening or before a closing square bracket

```js
// Good
var x = [1];

// Bad
var x = [ 1 ];
```

##### `disallowSpacesInsideParentheses`

Don't use a space after an opening or before a closing parenthesis

```js
// Good
var x = (1 + 2) * 3;

// Bad
var x = ( 1 + 2 ) * 3;
```

##### `disallowSpaceAfterObjectKeys`

Don't use a space after a key in an object

```js
// Good
var x = {a: 1};

// Bad
var x = {a : 1};
```

##### `requireCommaBeforeLineBreak`

Don't put commas on the following line

```js
// Good
var x = {
    one: 1,
    two: 2
};

// Bad
var x = {
    one: 1
    , two: 2
};
```

##### `requireOperatorBeforeLineBreak`

Break lines after (not before) an operator

```js
// Good
x = y ?
    1 : 2;

// Bad
x = y
    ? 1 : 2;
```

##### `disallowLeftStickedOperators`

Put a space before an operator

```js
// Good
x = y ? 1 : 2;

// Bad
x = y? 1 : 2;
```

##### `disallowRightStickedOperators`

Put a space after an operator

```js
// Good
x = y + 1;

// Bad
x = y +1;
```

##### `requireLeftStickedOperators`

Don't put a space before a comma

```js
// Good
x = [1, 2];

// Bad
x = [1 , 2];
```

##### `disallowSpaceAfterPrefixUnaryOperators`

Don't put a space after a prefix unary operator

```js
// Good
x = !y;
y = ++z;
// Bad
x = ! y;
y = ++ z;
```

##### `disallowSpaceBeforePostfixUnaryOperators`

Don't put a space after a postfix unary operator

```js
// Good
x = y++;

// Bad
x = y ++;
```

##### `requireSpaceBeforeBinaryOperators`

Put a space before binary operators

```js
// Good
x !== y;

// Bad
x!== y;
```

##### `requireSpaceAfterBinaryOperators`

Put a space after binary operators

```js
// Good
x + y;

// Bad
x +y;
```

##### `disallowImplicitTypeConversion`

Don't use implicit type conversion

```js
// Good
x = Boolean(y);
x = Number(y);
x = String(y);
x = s.indexOf('.') !== -1;

// Bad
x = !!y;
x = +y;
x = '' + y;
x = ~s.indexOf('.');
```

##### `requireCamelCaseOrUpperCaseIdentifiers`

Use camelCase or
UPPERCASE_WITH_UNDERSCORES for variable names

```js
// Good
var camelCase = 0;
var UPPER_CASE = 4;

// Bad
var lower_case = 1;
var Mixed_case = 2;
var mixed_Case = 3;
```

##### `disallowKeywords`

Disallow certain keywords from anywhere in the code base.

```js
// Bad
with (foo) {
}
```

##### `disallowMultipleLineBreaks`

Don't use multiple blank lines in a row

```js
// Bad
var x = 1;
x++;
```

##### `validateLineBreaks`

Checks line break character consistency:

```js
// Good
foo();<LF>

// Bad
foo();<CRLF>
foo();<CR>
```
##### `validateQuoteMarks`

Check quote mark usage is single quotes.

```js
// Good
var foo = 'bar';

// Bad
var foo = "bar";
```

##### `validateIndentation`

Check indentation is at 4 spaces, no tabs.

```js
// Good
function foo (bar) {
    console.log(bar);
}

// Bad
function foo (bar) {
  console.log(bar);
}

```

##### `disallowMixedSpacesAndTabs`

Check that spaces and tabs are not mixed.

##### `disallowTrailingWhitespace`

Check that no line ends in a whitespace char.

```js
// Good
var foo = "blah blah";<LF>

// Bad
var foo = "blah blah"; <LF>
```

##### `disallowTrailingComma`

Don't use a comma at the end of an array or object

```js
// Good
var foo = [1, 2, 3];
var bar = {a: "a", b: "b"}

// Bad
var foo = [1, 2, 3,];
var bar = {a: "a", b: "b",}
```

##### `disallowKeywordsOnNewLine`

Don't place `else` statements on their own line

```js
// Good
if (x < 0) {
    x++;
} else {
    x--;
}

// Bad
if (x < 0) {
    x++;
}
else {
    x--;
}
```

##### `maximumLineLength`

Checks that no single line extends beyond 80 chars.

##### `requireCapitalizedConstructors`

Constructor variable names should be Capitalised.

```js
// Good
var foo = new Bar();

// Bad
var foo = new bar();
```

##### `safeContextKeyword`

When saving `this` context call the variable `$this`.

```js
// Good
var $this = this;

// Bad
var that = this;
var self = this;
```

##### `disallowYodaConditions`

The variable should be on the left in boolean comparisons.

```js
// Good
if (a === 1) {
    // ...
}

// Bad
if (1 === a) {
    // ...
}
```
