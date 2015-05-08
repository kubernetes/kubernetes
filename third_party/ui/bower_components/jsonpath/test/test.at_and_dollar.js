var JSONPath = require('../'),
    testCase = require('nodeunit').testCase


var t1 = {
  simpleString: "simpleString",
  "@" : "@asPropertyName",
  "$" : "$asPropertyName",
  "a$a": "$inPropertyName",
  "$": {
    "@": "withboth",
  },
  a: {
    b: {
      c: "food"
    }
  }
};
  

module.exports = testCase({
    

    // ============================================================================    
    'test undefined, null': function(test) {
    // ============================================================================    
        test.expect(5);
        test.equal(undefined, JSONPath({json: undefined, path: 'foo'}));
        test.equal(null, JSONPath({json: null, path: 'foo'}));
        test.equal(undefined, JSONPath({json: {}, path: 'foo'})[0]);
        test.equal(undefined, JSONPath({json: { a: 'b' }, path: 'foo'})[0]);
        test.equal(undefined, JSONPath({json: { a: 'b' }, path: 'foo'})[100]);
        test.done();
    },

    
    // ============================================================================    
    'test $ and @': function(test) {
    // ============================================================================    
        test.expect(7);
        test.equal(t1['$'],   JSONPath({json: t1, path: '\$'})[0]);
        test.equal(t1['$'],   JSONPath({json: t1, path: '$'})[0]);
        test.equal(t1['a$a'], JSONPath({json: t1, path: 'a$a'})[0]);
        test.equal(t1['@'],   JSONPath({json: t1, path: '\@'})[0]);
        test.equal(t1['@'],   JSONPath({json: t1, path: '@'})[0]);
        test.equal(t1['$']['@'], JSONPath({json: t1, path: '$.$.@'})[0]);
        test.equal(undefined, JSONPath({json: t1, path: '\@'})[1]);
        
        test.done();
    }
    
});


