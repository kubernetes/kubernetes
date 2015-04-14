var JSONPath = require('../'),
    testCase = require('nodeunit').testCase

var json = {
    "name": "root",
    "children": [
        {"name": "child1", "children": [{"name": "child1_1"},{"name": "child1_2"}]},
        {"name": "child2", "children": [{"name": "child2_1"}]},
        {"name": "child3", "children": [{"name": "child3_1"}, {"name": "child3_2"}]}
    ]
};


module.exports = testCase({

    // ============================================================================
    'simple parent selection': function(test) {
    // ============================================================================
        test.expect(1);
        var result = JSONPath({json: json, path: '$.children[0]^', flatten: true});
        test.deepEqual(json.children, result);
        test.done();
    },

    // ============================================================================
    'parent selection with multiple matches': function(test) {
    // ============================================================================
        test.expect(1);
        var expected = [json.children,json.children];
        var result = JSONPath({json: json, path: '$.children[1:3]^'});
        test.deepEqual(expected, result);
        test.done();
    },

    // ============================================================================
    'select sibling via parent': function(test) {
    // ============================================================================
        test.expect(1);
        var expected = [{"name": "child3_2"}];
        var result = JSONPath({json: json, path: '$..[?(@.name && @.name.match(/3_1$/))]^[?(@.name.match(/_2$/))]'});
        test.deepEqual(expected, result);
        test.done();
    },

    // ============================================================================
    'parent parent parent': function(test) {
    // ============================================================================
        test.expect(1);
        var expected = json.children[0].children;
        var result = JSONPath({json: json, path: '$..[?(@.name && @.name.match(/1_1$/))].name^^', flatten: true});
        test.deepEqual(expected, result);
        test.done();
    },

    // ============================================================================
    'no such parent': function(test) {
    // ============================================================================
        test.expect(1);
        var result = JSONPath({json: json, path: 'name^^'});
        test.deepEqual([], result);
        test.done();
    }

});
