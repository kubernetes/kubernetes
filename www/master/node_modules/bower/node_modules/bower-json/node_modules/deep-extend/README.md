Node.JS module “Deep Extend”
============================

Recursive object extending.

Install
-----

    npm install deep-extend

Usage
-----

    var deepExtend = require('deep-extend');
    var obj1 = {
        a: 1,
        b: 2,
        d: {
            a: 1,
            b: [],
            c: { test1: 123, test2: 321 }
        },
        f: 5,
        g: 123
    };
    var obj2 = {
        b: 3,
        c: 5,
        d: {
            b: { first: 'one', second: 'two' },
            c: { test2: 222 }
        },
        e: { one: 1, two: 2 },
        f: [],
        g: (void 0)
    };

    deepExtend(obj1, obj2);

    console.log(obj1);
    /*
        { a: 1,
          b: 3,
          d:
           { a: 1,
             b: { first: 'one', second: 'two' },
             c: { test1: 123, test2: 222 } },
          f: [],
          c: 5,
          e: { one: 1, two: 2 },
          g: undefined }
    */
