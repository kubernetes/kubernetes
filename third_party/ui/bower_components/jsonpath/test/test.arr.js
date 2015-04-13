var JSONPath = require('../'),
    testCase = require('nodeunit').testCase

var json = {
    "store": {
        "book": { "category": "reference",
            "author": "Nigel Rees",
            "title": "Sayings of the Century",
            "price": [8.95, 8.94, 8.93]
        },
        "books": [
            { "category": "reference",
                "author": "Nigel Rees",
                "title": "Sayings of the Century",
                "price": [8.95, 8.94, 8.93]
            }
        ]
    }
};

module.exports = testCase({
    'get single': function (test) {
        var expected = json.store.book;
        var result = JSONPath({json: json, path: 'store.book', flatten: true, wrap: false});
        test.deepEqual(expected, result);
        test.done();
    },

    'get arr': function (test) {
        var expected = json.store.books;
        var result = JSONPath({json: json, path: 'store.books', flatten: true, wrap: false});
        test.deepEqual(expected, result);
        test.done();
    }
});
