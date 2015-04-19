# alter.js
Alters a string by replacing multiple range fragments in one fast pass.
Works in node and browsers.



## Usage
```javascript
    var alter = require("alter");
    alter("0123456789", [
        {start: 1, end: 3, str: "first"},
        {start: 5, end: 9, str: "second"},
    ]); // => "0first34second9"
```

The fragments does not need to be sorted but must not overlap. More examples in `test/alter-tests.js`


## Installation

### Node
Install using npm

    npm install alter

```javascript
var alter = require("alter");
```

### Browser
Clone the repo and include it in a script tag

    git clone https://github.com/olov/alter.git

```html
<script src="alter/alter.js"></script>
```
