# simple-is.js
A maximally minimal type-testing library. Use it to make your code
more readable. Works in node and browsers.



## Usage
`var is = require("simple-is");`

Use `is.number(x)` instead of `typeof x === "number"` (also `is.boolean`, `is.string`, `is.fn`).

Use `is.nan(x)` instead of `typeof x === "number" && isNaN(x)`, `x !== x` or ES6 `Number.isNaN(x)`.

Use `is.object(x)` instead of `x !== null && typeof x === "object"`.

Use `is.primitive(x)` instead of `x === null || x === undefined || typeof x === "boolean" || typeof x === "number" || typeof x === "string"` (verbose on purpose).

Use `is.array(x)` instead of ES5 `Array.isArray`.

Use `is.finitenumber(x)` instead of `typeof x === "number" && isFinite(x)` or ES6 `Number.isFinite(x)`.

Use `is.someof(x, ["first", 2, obj])` instead of (usually) `x === "first" || x === 2 || x === obj` or (alternatively)  `["first", 2, obj].indexOf(x) >= 0`. Great for reducing copy and paste mistake in `if`-conditions and for making them more readable.

Use `is.noneof(x, ["first", 2, obj])` instead of (usually) `x !== "first" && x !== 2 && x !== obj` or (alternatively)  `["first", 2, obj].indexOf(x) === -1`.

Use `is.own(x, "name")` instead of `Object.prototype.hasOwnProperty.call(x, "name")`.

That's it.



## Installation

### Node
Install using npm

    npm install simple-is

```javascript
var is = require("simple-is");
```

### Browser
Clone the repo and include it in a script tag

    git clone https://github.com/olov/simple-is.git

```html
<script src="simple-is/simple-is.js"></script>
```
