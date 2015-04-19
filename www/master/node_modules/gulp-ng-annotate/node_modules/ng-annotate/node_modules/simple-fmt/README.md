# simple-fmt.js
A maximally minimal string formatting library. Use it to make your code more
readable compared to plain old string concatenation using `+`. The code is
shorter than the MIT license text so it doesn't hog you down and you can use
it everywhere. Works in node and browsers.



## Usage
```javascript
var fmt = require("simple-fmt");
console.log(fmt("hello {0} of age {1}", name, age));
```

instead of

```javascript
console.log("hello " + name + " of age " + age);
```

because string formatting with `+` makes your eyes bleed and fingers hurt.


There's also `fmt.obj(string, obj)` and `fmt.repeat(string, n)`:
```javascript
var o = {name: "xyz", age: 42};
fmt.obj("hello {name} of age {age}", obj);
fmt.repeat("*", 3); // "***"
```

That's it.



## Installation

### Node
Install using npm

    npm install simple-fmt

```javascript
var fmt = require("simple-fmt");
```

### Browser
Clone the repo and include it in a script tag

    git clone https://github.com/olov/simple-fmt.git

```html
<script src="simple-fmt/simple-fmt.js"></script>
```
