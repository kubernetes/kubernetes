var JSONPath = require('../'),
    testCase = require('nodeunit').testCase

// tests based on examples at http://goessner.net/articles/JsonPath/

var json = {"store":{
    "book":[
        { "category":"reference",
            "author":"Nigel Rees",
            "title":"Sayings of the Century",
            "price":[8.95, 8.94, 8.93]
        },
        { "category":"fiction",
            "author":"Evelyn Waugh",
            "title":"Sword of Honour",
            "price":12.99
        },
        { "category":"fiction",
            "author":"Herman Melville",
            "title":"Moby Dick",
            "isbn":"0-553-21311-3",
            "price":8.99
        },
        { "category":"fiction",
            "author":"J. R. R. Tolkien",
            "title":"The Lord of the Rings",
            "isbn":"0-395-19395-8",
            "price":22.99
        }
    ],
    "bicycle":{
        "color":"red",
        "price":19.95
    }
}
};


module.exports = testCase({

    // ============================================================================
    'all sub properties, entire tree': function (test) {
        // ============================================================================
        test.expect(1);
        var books = json.store.book;
        var expected = [books[1].price, books[2].price, books[3].price, json.store.bicycle.price];
        expected = books[0].price.concat(expected);
        var result = JSONPath({json: json, path: '$.store..price', flatten: true});
        test.deepEqual(expected, result);

        test.done();
    }
});
