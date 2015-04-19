if(module.parent === null) {
  console.log('Run this test file with nodeunit:');
  console.log('$ nodeunit test.js');
}


var clone = require('./');
var util = require('util');
var _ = require('underscore');



exports["clone string"] = function(test) {
  test.expect(2); // how many tests?

  var a = "foo";
  test.strictEqual(clone(a), a);
  a = "";
  test.strictEqual(clone(a), a);

  test.done();
};



exports["clone number"] = function(test) {
  test.expect(5); // how many tests?

  var a = 0;
  test.strictEqual(clone(a), a);
  a = 1;
  test.strictEqual(clone(a), a);
  a = -1000;
  test.strictEqual(clone(a), a);
  a = 3.1415927;
  test.strictEqual(clone(a), a);
  a = -3.1415927;
  test.strictEqual(clone(a), a);

  test.done();
};



exports["clone date"] = function(test) {
  test.expect(3); // how many tests?

  var a = new Date;
  var c = clone(a);
  test.ok(a instanceof Date);
  test.ok(c instanceof Date);
  test.equal(c.getTime(), a.getTime());

  test.done();
};



exports["clone object"] = function(test) {
  test.expect(2); // how many tests?

  var a = { foo: { bar: "baz" } };
  var b = clone(a);

  test.ok(_(a).isEqual(b), "underscore equal");
  test.deepEqual(b, a);

  test.done();
};



exports["clone array"] = function(test) {
  test.expect(2); // how many tests?

  var a = [
    { foo: "bar" },
    "baz"
  ];
  var b = clone(a);

  test.ok(_(a).isEqual(b), "underscore equal");
  test.deepEqual(b, a);

  test.done();
};

exports["clone buffer"] = function(test) {
  test.expect(1);

  var a = new Buffer("this is a test buffer");
  var b = clone(a);

  // no underscore equal since it has no concept of Buffers
  test.deepEqual(b, a);
  test.done();
};



exports["clone regexp"] = function(test) {
  test.expect(5);

  var a = /abc123/gi;
  var b = clone(a);

  test.deepEqual(b, a);

  var c = /a/g;
  test.ok(c.lastIndex === 0);

  c.exec('123a456a');
  test.ok(c.lastIndex === 4);

  var d = clone(c);
  test.ok(d.global);
  test.ok(d.lastIndex === 4);

  test.done();
};


exports["clone object containing array"] = function(test) {
  test.expect(2); // how many tests?

  var a = {
    arr1: [ { a: '1234', b: '2345' } ],
    arr2: [ { c: '345', d: '456' } ]
  };
  var b = clone(a);

  test.ok(_(a).isEqual(b), "underscore equal");
  test.deepEqual(b, a);

  test.done();
};



exports["clone object with circular reference"] = function(test) {
  test.expect(8); // how many tests?

  var _ = test.ok;
  var c = [1, "foo", {'hello': 'bar'}, function() {}, false, [2]];
  var b = [c, 2, 3, 4];
  var a = {'b': b, 'c': c};
  a.loop = a;
  a.loop2 = a;
  c.loop = c;
  c.aloop = a;
  var aCopy = clone(a);
  _(a != aCopy);
  _(a.c != aCopy.c);
  _(aCopy.c == aCopy.b[0]);
  _(aCopy.c.loop.loop.aloop == aCopy);
  _(aCopy.c[0] == a.c[0]);

  //console.log(util.inspect(aCopy, true, null) );
  //console.log("------------------------------------------------------------");
  //console.log(util.inspect(a, true, null) );
  _(eq(a, aCopy));
  aCopy.c[0] = 2;
  _(!eq(a, aCopy));
  aCopy.c = "2";
  _(!eq(a, aCopy));
  //console.log("------------------------------------------------------------");
  //console.log(util.inspect(aCopy, true, null) );

  function eq(x, y) {
    return util.inspect(x, true, null) === util.inspect(y, true, null);
  }

  test.done();
};



exports['clonePrototype'] = function(test) {
  test.expect(3); // how many tests?

  var a = {
    a: "aaa",
    x: 123,
    y: 45.65
  };
  var b = clone.clonePrototype(a);

  test.strictEqual(b.a, a.a);
  test.strictEqual(b.x, a.x);
  test.strictEqual(b.y, a.y);

  test.done();
}

exports['cloneWithinNewVMContext'] = function(test) {
  test.expect(3);
  var vm = require('vm');
  var ctx = vm.createContext({ clone: clone });
  var script = "clone( {array: [1, 2, 3], date: new Date(), regex: /^foo$/ig} );";
  var results = vm.runInContext(script, ctx);
  test.ok(results.array instanceof Array);
  test.ok(results.date instanceof Date);
  test.ok(results.regex instanceof RegExp);
  test.done();
}

exports['cloneObjectWithNoConstructor'] = function(test) {
  test.expect(3);
  var n = null;
  var a = { foo: 'bar' };
  a.__proto__ = n;
  test.ok(typeof a === 'object');
  test.ok(typeof a !== null);
  var b = clone(a);
  test.ok(a.foo, b.foo);
  test.done();
}

exports['clone object with depth argument'] = function (test) {
  test.expect(6);
  var a = {
    foo: {
      bar : {
        baz : 'qux'
      }
    }
  };
  var b = clone(a, false, 1);
  test.deepEqual(b, a);
  test.notEqual(b, a);
  test.strictEqual(b.foo, a.foo);

  b = clone(a, true, 2);
  test.deepEqual(b, a);
  test.notEqual(b.foo, a.foo);
  test.strictEqual(b.foo.bar, a.foo.bar);
  test.done();
}

exports['maintain prototype chain in clones'] = function (test) {
  test.expect(1);
  function Constructor() {}
  var a = new Constructor();
  var b = clone(a);
  test.strictEqual(Object.getPrototypeOf(a), Object.getPrototypeOf(b));
  test.done();
}

exports['parent prototype is overriden with prototype provided'] = function (test) {
  test.expect(1);
  function Constructor() {}
  var a = new Constructor();
  var b = clone(a, true, Infinity, null);
  test.strictEqual(b.__defineSetter__, undefined);
  test.done();
}

exports['clone object with null children'] = function(test) {
  test.expect(1);
  var a = {
    foo: {
      bar: null,
      baz: {
        qux: false
      }
    }
  };
  var b = clone(a);
  test.deepEqual(b, a);
  test.done();
}

exports['clone instance with getter'] = function(test) {
  test.expect(1);
  function Ctor() {};
  Object.defineProperty(Ctor.prototype, 'prop', {
    configurable: true,
    enumerable: true,
    get: function() {
      return 'value';
    }
  });

  var a = new Ctor();
  var b = clone(a);

  test.strictEqual(b.prop, 'value');
  test.done();
};