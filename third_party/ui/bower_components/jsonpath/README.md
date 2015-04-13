JSONPath [![build status](https://secure.travis-ci.org/s3u/JSONPath.png)](http://travis-ci.org/s3u/JSONPath)
========

Analyse, transform, and selectively extract data from JSON documents (and JavaScript objects).

Install
-------
    
    npm install JSONPath

Usage
-----

In node.js:

```js
var JSONPath = require('JSONPath');
JSONPath({json: obj, path: path});
```

For browser usage you can directly include `lib/jsonpath.js`, no browserify
magic necessary:

```html
<script src="lib/jsonpath.js"></script>
<script>
    JSONPath({json: obj, path: path});
</script>
```

Examples
--------

Given the following JSON, taken from http://goessner.net/articles/JsonPath/ :

```json
{
  "store": {
    "book": [
      {
        "category": "reference",
        "author": "Nigel Rees",
        "title": "Sayings of the Century",
        "price": 8.95
      },
      {
        "category": "fiction",
        "author": "Evelyn Waugh",
        "title": "Sword of Honour",
        "price": 12.99
      },
      {
        "category": "fiction",
        "author": "Herman Melville",
        "title": "Moby Dick",
        "isbn": "0-553-21311-3",
        "price": 8.99
      },
      {
        "category": "fiction",
        "author": "J. R. R. Tolkien",
        "title": "The Lord of the Rings",
        "isbn": "0-395-19395-8",
        "price": 22.99
      }
    ],
    "bicycle": {
      "color": "red",
      "price": 19.95
    }
  }
}
```


XPath               | JSONPath               | Result
------------------- | ---------------------- | -------------------------------------
/store/book/author	| $.store.book[*].author | the authors of all books in the store 
//author            | $..author              | all authors 
/store/*            | $.store.*              | all things in store, which are some books and a red bicycle.
/store//price       | $.store..price         | the price of everything in the store.
//book[3]           | $..book[2]             | the third book
//book[last()]      | $..book[(@.length-1)]  | the last book in order.
                    | $..book[-1:]           |
//book[position()<3]| $..book[0,1]           | the first two books
                    | $..book[:2]            | 
//book[isbn]        | $..book[?(@.isbn)]     | filter all books with isbn number
//book[price<10]    | $..book[?(@.price<10)] | filter all books cheapier than 10
//*[price>19]/..    | $..[?(@.price>19)]^    | categories with things more expensive than 19
//*                 | $..*                   | all Elements in XML document. All members of JSON structure.

Development
-----------

Running the tests on node: `npm test`. For in-browser tests:

* Ensure that nodeunit is browser-compiled: `cd node_modules/nodeunit; make browser;`
* Serve the js/html files:

```sh
    node -e "require('http').createServer(function(req,res) { \
        var s = require('fs').createReadStream('.' + req.url); \
        s.pipe(res); s.on('error', function() {}); }).listen(8082);"
```
* To run the tests visit [http://localhost:8082/test/test.html]().


License
-------

[MIT License](http://www.opensource.org/licenses/mit-license.php).
