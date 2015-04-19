// Load modules

var Lab = require('lab');
var Hoek = require('../lib');


// Declare internals

var internals = {};


// Test shortcuts

var expect = Lab.expect;
var before = Lab.before;
var after = Lab.after;
var describe = Lab.experiment;
var it = Lab.test;


describe('Hoek', function () {

    var nestedObj = {
        v: [7,8,9],
        w: /^something$/igm,
        x: {
            a: [1, 2, 3],
            b: 123456,
            c: new Date(),
            d: /hi/igm,
            e: /hello/
        },
        y: 'y',
        z: new Date()
    };

    var dupsArray = [nestedObj, { z: 'z' }, nestedObj];
    var reducedDupsArray = [nestedObj, { z: 'z' }];

    describe('#clone', function () {

        it('should clone a nested object', function (done) {

            var a = nestedObj;
            var b = Hoek.clone(a);

            expect(a).to.deep.equal(b);
            expect(a.z.getTime()).to.equal(b.z.getTime());
            done();
        });

        it('should clone a null object', function (done) {

            var b = Hoek.clone(null);

            expect(b).to.equal(null);
            done();
        });

        it('should not convert undefined properties to null', function (done) {

            var obj = { something: undefined };
            var b = Hoek.clone(obj);

            expect(typeof b.something).to.equal('undefined');
            done();
        });

        it('should not throw on circular reference', function (done) {

            var a = {};
            a.x = a;

            var test = (function () {

                var b = Hoek.clone(a);
            });

            expect(test).to.not.throw();
            done();
        });

        it('should properly clone circular reference', function (done) {

            var x = {
                'z': new Date()
            };
            x.y = x;

            var b = Hoek.clone(x);
            expect(Object.keys(b.y)).to.deep.equal(Object.keys(x))
            expect(b.z).to.not.equal(x.z);
            expect(b.y).to.not.equal(x.y);
            expect(b.y.z).to.not.equal(x.y.z);
            expect(b.y).to.equal(b);
            expect(b.y.y.y.y).to.equal(b);
            done();
        });

        it('should properly clone deeply nested object', function (done) {

            var a = {
                x: {
                    y: {
                        a: [1, 2, 3],
                        b: 123456,
                        c: new Date(),
                        d: /hi/igm,
                        e: /hello/
                    },
                }
            };

            var b = Hoek.clone(a);

            expect(a).to.deep.equal(b);
            expect(a.x.y.c.getTime()).to.equal(b.x.y.c.getTime());
            done();
        });

        it('should properly clone arrays', function (done) {

            var a = [1,2,3];

            var b = Hoek.clone(a);

            expect(a).to.deep.equal(b);
            done();
        });

        it('should perform actual copy for shallow keys (no pass by reference)', function (done) {

            var x = Hoek.clone(nestedObj);
            var y = Hoek.clone(nestedObj);

            // Date
            expect(x.z).to.not.equal(nestedObj.z);
            expect(x.z).to.not.equal(y.z);

            // Regex
            expect(x.w).to.not.equal(nestedObj.w);
            expect(x.w).to.not.equal(y.w);

            // Array
            expect(x.v).to.not.equal(nestedObj.v);
            expect(x.v).to.not.equal(y.v);

            // Immutable(s)
            x.y = 5;
            expect(x.y).to.not.equal(nestedObj.y);
            expect(x.y).to.not.equal(y.y);

            done();
        });

        it('should perform actual copy for deep keys (no pass by reference)', function (done) {

            var x = Hoek.clone(nestedObj);
            var y = Hoek.clone(nestedObj);

            expect(x.x.c).to.not.equal(nestedObj.x.c);
            expect(x.x.c).to.not.equal(y.x.c);

            expect(x.x.c.getTime()).to.equal(nestedObj.x.c.getTime());
            expect(x.x.c.getTime()).to.equal(y.x.c.getTime());
            done();
        });

        it('copies functions with properties', function (done) {

            var a = {
                x: function () { return 1; },
                y: {}
            };
            a.x.z = 'string in function';
            a.x.v = function () { return 2; };
            a.y.u = a.x;

            var b = Hoek.clone(a);
            expect(b.x()).to.equal(1);
            expect(b.x.v()).to.equal(2);
            expect(b.y.u).to.equal(b.x);
            expect(b.x.z).to.equal('string in function');
            done();
        });

        it('should copy a buffer', function(done){
            var tls = {
                key: new Buffer([1,2,3,4,5]),
                cert: new Buffer([1,2,3,4,5,6,10])
            }

            copiedTls = Hoek.clone(tls);
            expect(Buffer.isBuffer(copiedTls.key)).to.equal(true);
            expect(JSON.stringify(copiedTls.key)).to.equal(JSON.stringify(tls.key))
            expect(Buffer.isBuffer(copiedTls.cert)).to.equal(true);
            expect(JSON.stringify(copiedTls.cert)).to.equal(JSON.stringify(tls.cert))
            done();
        });
    });

    describe('#merge', function () {

        it('does not throw if source is null', function (done) {

            var a = {};
            var b = null;
            var c = null;

            expect(function () {

                c = Hoek.merge(a, b);
            }).to.not.throw();

            expect(c).to.equal(a);
            done();
        });

        it('does not throw if source is undefined', function (done) {

            var a = {};
            var b = undefined;
            var c = null;

            expect(function () {

                c = Hoek.merge(a, b);
            }).to.not.throw();

            expect(c).to.equal(a);
            done();
        });

        it('throws if source is not an object', function (done) {

            expect(function () {

                var a = {};
                var b = 0;

                Hoek.merge(a, b);
            }).to.throw('Invalid source value: must be null, undefined, or an object');
            done();
        });

        it('throws if target is not an object', function (done) {

            expect(function () {

                var a = 0;
                var b = {};

                Hoek.merge(a, b);
            }).to.throw('Invalid target value: must be an object');
            done();
        });

        it('throws if target is not an array and source is', function (done) {

            expect(function () {

                var a = {};
                var b = [1, 2];

                Hoek.merge(a, b);
            }).to.throw('Cannot merge array onto an object');
            done();
        });

        it('returns the same object when merging arrays', function (done) {

            var a = [];
            var b = [1, 2];

            expect(Hoek.merge(a, b)).to.equal(a);
            done();
        });

        it('should combine an empty object with a non-empty object', function (done) {

            var a = {};
            var b = nestedObj;

            var c = Hoek.merge(a, b);
            expect(a).to.deep.equal(b);
            expect(c).to.deep.equal(b);
            done();
        });

        it('should override values in target', function (done) {

            var a = { x: 1, y: 2, z: 3, v: 5, t: 'test', m: 'abc' };
            var b = { x: null, z: 4, v: 0, t: { u: 6 }, m: '123' };

            var c = Hoek.merge(a, b);
            expect(c.x).to.equal(null);
            expect(c.y).to.equal(2);
            expect(c.z).to.equal(4);
            expect(c.v).to.equal(0);
            expect(c.m).to.equal('123');
            expect(c.t).to.deep.equal({ u: 6 });
            done();
        });

        it('should override values in target (flip)', function (done) {

            var a = { x: 1, y: 2, z: 3, v: 5, t: 'test', m: 'abc' };
            var b = { x: null, z: 4, v: 0, t: { u: 6 }, m: '123' };

            var d = Hoek.merge(b, a);
            expect(d.x).to.equal(1);
            expect(d.y).to.equal(2);
            expect(d.z).to.equal(3);
            expect(d.v).to.equal(5);
            expect(d.m).to.equal('abc');
            expect(d.t).to.deep.equal('test');
            done();
        });
    });

    describe('#applyToDefaults', function () {

        var defaults = {
            a: 1,
            b: 2,
            c: {
                d: 3,
                e: [5, 6]
            },
            f: 6,
            g: 'test'
        };

        it('should return null if options is false', function (done) {

            var result = Hoek.applyToDefaults(defaults, false);
            expect(result).to.equal(null);
            done();
        });

        it('should return a copy of defaults if options is true', function (done) {

            var result = Hoek.applyToDefaults(defaults, true);
            expect(result).to.deep.equal(result);
            done();
        });

        it('should apply object to defaults', function (done) {

            var obj = {
                a: null,
                c: {
                    e: [4]
                },
                f: 0,
                g: {
                    h: 5
                }
            };

            var result = Hoek.applyToDefaults(defaults, obj);
            expect(result.c.e).to.deep.equal([4]);
            expect(result.a).to.equal(1);
            expect(result.b).to.equal(2);
            expect(result.f).to.equal(0);
            expect(result.g).to.deep.equal({ h: 5 });
            done();
        });
    });

    describe('#unique', function () {

        it('should ensure uniqueness within array of objects based on subkey', function (done) {

            var a = Hoek.unique(dupsArray, 'x');
            expect(a).to.deep.equal(reducedDupsArray);
            done();
        });

        it('removes duplicated without key', function (done) {

            expect(Hoek.unique([1, 2, 3, 4, 2, 1, 5])).to.deep.equal([1, 2, 3, 4, 5]);
            done();
        });
    });

    describe('#mapToObject', function () {

        it('should return null on null array', function (done) {

            var a = Hoek.mapToObject(null);
            expect(a).to.equal(null);
            done();
        });

        it('should convert basic array to existential object', function (done) {

            var keys = [1, 2, 3, 4];
            var a = Hoek.mapToObject(keys);
            for (var i in keys) {
                expect(a[keys[i]]).to.equal(true);
            }
            done();
        });

        it('should convert array of objects to existential object', function (done) {

            var keys = [{ x: 1 }, { x: 2 }, { x: 3 }];
            var subkey = 'x';
            var a = Hoek.mapToObject(keys, subkey);
            for (var i in keys) {
                expect(a[keys[i][subkey]]).to.equal(true);
            }
            done();
        });
    });

    describe('#intersect', function () {

        it('should return the common objects of two arrays', function (done) {

            var array1 = [1, 2, 3, 4, 4, 5, 5];
            var array2 = [5, 4, 5, 6, 7];
            var common = Hoek.intersect(array1, array2);
            expect(common.length).to.equal(2);
            done();
        });

        it('should return just the first common object of two arrays', function (done) {

            var array1 = [1, 2, 3, 4, 4, 5, 5];
            var array2 = [5, 4, 5, 6, 7];
            var common = Hoek.intersect(array1, array2, true);
            expect(common).to.equal(5);
            done();
        });

        it('should return an empty array if either input is null', function (done) {

            expect(Hoek.intersect([1], null).length).to.equal(0);
            expect(Hoek.intersect(null, [1]).length).to.equal(0);
            done();
        });

        it('should return the common objects of object and array', function (done) {

            var array1 = [1, 2, 3, 4, 4, 5, 5];
            var array2 = [5, 4, 5, 6, 7];
            var common = Hoek.intersect(Hoek.mapToObject(array1), array2);
            expect(common.length).to.equal(2);
            done();
        });
    });

    describe('#matchKeys', function () {

        it('should match the existing object keys', function (done) {

            var obj = {
                a: 1,
                b: 2,
                c: 3,
                d: null
            };

            expect(Hoek.matchKeys(obj, ['b', 'c', 'd', 'e'])).to.deep.equal(['b', 'c', 'd']);
            done();
        });
    });

    describe('#flatten', function () {

        it('should return a flat array', function (done) {

            var result = Hoek.flatten([1, 2, [3, 4, [5, 6], [7], 8], [9], [10, [11, 12]], 13]);
            expect(result.length).to.equal(13);
            expect(result).to.deep.equal([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
            done();
        });
    });

    describe('#removeKeys', function () {

        var objWithHiddenKeys = {
            location: {
                name: 'San Bruno'
            },
            company: {
                name: '@WalmartLabs'
            }
        };

        it('should delete params with definition\'s hide set to true', function (done) {

            var a = Hoek.removeKeys(objWithHiddenKeys, ['location']);
            expect(objWithHiddenKeys.location).to.not.exist;
            expect(objWithHiddenKeys.company).to.exist;
            done();
        });
    });

    describe('#reach', function () {

        var obj = {
            a: {
                b: {
                    c: {
                        d: 1,
                        e: 2
                    },
                    f: 'hello'
                },
                g: {
                    h: 3
                }
            },
            i: function () { }
        };

        it('returns a valid member', function (done) {

            expect(Hoek.reach(obj, 'a.b.c.d')).to.equal(1);
            done();
        });

        it('returns null on null object', function (done) {

            expect(Hoek.reach(null, 'a.b.c.d')).to.not.exist;
            done();
        });

        it('returns null on missing member', function (done) {

            expect(Hoek.reach(obj, 'a.b.c.d.x')).to.not.exist;
            done();
        });

        it('returns null on invalid member', function (done) {

            expect(Hoek.reach(obj, 'a.b.c.d-.x')).to.not.exist;
            done();
        });

        it('returns function member', function (done) {

            expect(typeof Hoek.reach(obj, 'i')).to.equal('function');
            done();
        });
    });

    describe('#inheritAsync', function () {

        it('should inherit selected methods and wrap in async call', function (done) {

            var proto = {
                a: function () {
                    return 'a!';
                },
                b: function () {
                    return 'b!';
                },
                c: function () {
                    throw new Error('c!');
                }
            };

            var targetFunc = function () { };
            targetFunc.prototype.c = function () {

                return 'oops';
            };

            Hoek.inheritAsync(targetFunc, proto, ['a', 'c']);
            var target = new targetFunc();

            expect(typeof target.a).to.equal('function');
            expect(typeof target.c).to.equal('function');
            expect(target.b).to.not.exist;

            target.a(function (err, result) {

                expect(err).to.not.exist;
                expect(result).to.equal('a!');

                target.c(function (err, result) {

                    expect(result).to.not.exist;
                    expect(err.message).to.equal('c!');
                    done();
                });
            });
        });
    });

    describe('#callStack', function () {

        it('should return the full call stack', function (done) {

            var stack = Hoek.callStack();
            expect(stack[0][0]).to.contain('index.js');
            expect(stack[0][2]).to.equal(30);
            done();
        });
    });

    describe('#displayStack ', function () {

        it('should return the full call stack for display', function (done) {

            var stack = Hoek.displayStack();
            expect(stack[0]).to.contain('test/index.js:');
            done();
        });

        it('should include constructor functions correctly', function (done) {

            var Something = function (next) {

                next();
            };

            var something = new Something(function () {

                var stack = Hoek.displayStack();
                expect(stack[1]).to.contain('new Something');
                done();
            });
        });
    });

    describe('#abort', function () {

        it('should exit process when not in test mode', function (done) {

            var env = process.env.NODE_ENV;
            var write = process.stdout.write;
            var exit = process.exit;

            process.env.NODE_ENV = 'nottatest';
            process.stdout.write = function () { };
            process.exit = function (state) {

                process.exit = exit;
                process.env.NODE_ENV = env;
                process.stdout.write = write;

                expect(state).to.equal(1);
                done();
            };

            Hoek.abort('Boom');
        });

        it('should throw when not in test mode and abortThrow is true', function (done) {

            var env = process.env.NODE_ENV;
            process.env.NODE_ENV = 'nottatest';
            Hoek.abortThrow = true;

            var fn = function () {

                Hoek.abort('my error message');
            };

            expect(fn).to.throw('my error message');
            Hoek.abortThrow = false;
            process.env.NODE_ENV = env;

            done();
        });


        it('should respect hideStack argument', function (done) {

            var env = process.env.NODE_ENV;
            var write = process.stdout.write;
            var exit = process.exit;
            var output = '';

            process.exit = function () { };
            process.env.NODE_ENV = '';
            process.stdout.write = function (message) {

                output = message;
            };

            Hoek.abort('my error message', true);

            process.env.NODE_ENV = env;
            process.stdout.write = write;
            process.exit = exit;

            expect(output).to.equal('ABORT: my error message\n\t\n');

            done();
        });

        it('should default to showing stack', function (done) {

            var env = process.env.NODE_ENV;
            var write = process.stdout.write;
            var exit = process.exit;
            var output = '';

            process.exit = function () { };
            process.env.NODE_ENV = '';
            process.stdout.write = function (message) {

                output = message;
            };

            Hoek.abort('my error message');

            process.env.NODE_ENV = env;
            process.stdout.write = write;
            process.exit = exit;

            expect(output).to.contain('index.js');

            done();
        });
    });

    describe('#assert', function () {

        it('should throw an Error when using assert in a test', function (done) {

            var fn = function () {

                Hoek.assert(false, 'my error message');
            };

            expect(fn).to.throw('my error message');
            done();
        });

        it('should throw an Error when using assert in a test with no message', function (done) {

            var fn = function () {

                Hoek.assert(false);
            };

            expect(fn).to.throw('Unknown error');
            done();
        });

        it('should throw an Error when using assert in a test with multipart message', function (done) {

            var fn = function () {

                Hoek.assert(false, 'This', 'is', 'my message');
            };

            expect(fn).to.throw('This is my message');
            done();
        });

        it('should throw an Error when using assert in a test with object message', function (done) {

            var fn = function () {

                Hoek.assert(false, 'This', 'is', { spinal: 'tap' });
            };

            expect(fn).to.throw('This is {"spinal":"tap"}');
            done();
        });

        it('should throw an Error when using assert in a test with error object message', function (done) {

            var fn = function () {

                Hoek.assert(false, new Error('This is spinal tap'));
            };

            expect(fn).to.throw('This is spinal tap');
            done();
        });
    });

    describe('#loadDirModules', function () {

        it('should load modules from directory', function (done) {

            var target = {};
            Hoek.loadDirModules(__dirname + '/modules', ['test2'], target);
            expect(target.Test1.x).to.equal(1);
            expect(target.Test2).to.not.exist;
            expect(target.Test3.z).to.equal(3);
            done();
        });

        it('should list modules from directory into function', function (done) {

            var target = {};
            Hoek.loadDirModules(__dirname + '/modules', ['test2'], function (path, name, capName) {

                target[name] = capName;
            });

            expect(target.test1).to.equal('Test1');
            expect(target.test2).to.not.exist;
            expect(target.test3).to.equal('Test3');
            done();
        });
    });

    describe('#rename', function () {

        it('should rename object key', function (done) {

            var a = { b: 'c' };
            Hoek.rename(a, 'b', 'x');
            expect(a.b).to.not.exist;
            expect(a.x).to.equal('c');
            done();
        });
    });

    describe('Timer', function () {

        it('should return time elapsed', function (done) {

            var timer = new Hoek.Timer();
            setTimeout(function () {

                expect(timer.elapsed()).to.be.above(9);
                done();
            }, 12);
        });
    });

    describe('#loadPackage', function () {

        it('should', function (done) {

            var pack = Hoek.loadPackage();
            expect(pack.name).to.equal('hoek');
            done();
        });
    });

    describe('#escapeRegex', function () {

        it('should escape all special regular expression characters', function (done) {

            var a = Hoek.escapeRegex('4^f$s.4*5+-_?%=#!:@|~\\/`"(>)[<]d{}s,');
            expect(a).to.equal('4\\^f\\$s\\.4\\*5\\+\\-_\\?%\\=#\\!\\:@\\|~\\\\\\/`"\\(>\\)\\[<\\]d\\{\\}s\\,');
            done();
        });
    });

    describe('#toss', function () {

        it('should call callback with new error', function (done) {

            var callback = function (err) {

                expect(err).to.exist;
                expect(err.message).to.equal('bug');
                done();
            };

            Hoek.toss(true, 'feature', callback);
            Hoek.toss(false, 'bug', callback);
        });

        it('should call callback with new error and no message', function (done) {

            Hoek.toss(false, function (err) {

                expect(err).to.exist;
                expect(err.message).to.equal('');
                done();
            });
        });

        it('should call callback with error condition', function (done) {

            Hoek.toss(new Error('boom'), function (err) {

                expect(err).to.exist;
                expect(err.message).to.equal('boom');
                done();
            });
        });

        it('should call callback with new error using message with error condition', function (done) {

            Hoek.toss(new Error('ka'), 'boom', function (err) {

                expect(err).to.exist;
                expect(err.message).to.equal('boom');
                done();
            });
        });

        it('should call callback with new error using passed error with error condition', function (done) {

            Hoek.toss(new Error('ka'), new Error('boom'), function (err) {

                expect(err).to.exist;
                expect(err.message).to.equal('boom');
                done();
            });
        });
    });

    describe('Base64Url', function () {

        var base64str = 'AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0-P0BBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWltcXV5fYGFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6e3x9fn-AgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq-wsbKztLW2t7i5uru8vb6_wMHCw8TFxsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t_g4eLj5OXm5-jp6uvs7e7v8PHy8_T19vf4-fr7_P3-_w';
        var str = unescape('%00%01%02%03%04%05%06%07%08%09%0A%0B%0C%0D%0E%0F%10%11%12%13%14%15%16%17%18%19%1A%1B%1C%1D%1E%1F%20%21%22%23%24%25%26%27%28%29*+%2C-./0123456789%3A%3B%3C%3D%3E%3F@ABCDEFGHIJKLMNOPQRSTUVWXYZ%5B%5C%5D%5E_%60abcdefghijklmnopqrstuvwxyz%7B%7C%7D%7E%7F%80%81%82%83%84%85%86%87%88%89%8A%8B%8C%8D%8E%8F%90%91%92%93%94%95%96%97%98%99%9A%9B%9C%9D%9E%9F%A0%A1%A2%A3%A4%A5%A6%A7%A8%A9%AA%AB%AC%AD%AE%AF%B0%B1%B2%B3%B4%B5%B6%B7%B8%B9%BA%BB%BC%BD%BE%BF%C0%C1%C2%C3%C4%C5%C6%C7%C8%C9%CA%CB%CC%CD%CE%CF%D0%D1%D2%D3%D4%D5%D6%D7%D8%D9%DA%DB%DC%DD%DE%DF%E0%E1%E2%E3%E4%E5%E6%E7%E8%E9%EA%EB%EC%ED%EE%EF%F0%F1%F2%F3%F4%F5%F6%F7%F8%F9%FA%FB%FC%FD%FE%FF');

        describe('#base64urlEncode', function () {

            it('should base64 URL-safe a string', function (done) {

                expect(Hoek.base64urlEncode(str)).to.equal(base64str);
                done();
            });
        });

        describe('#base64urlDecode', function () {

            it('should un-base64 URL-safe a string', function (done) {

                expect(Hoek.base64urlDecode(base64str)).to.equal(str);
                done();
            });

            it('should return error on undefined input', function (done) {

                expect(Hoek.base64urlDecode().message).to.exist;
                done();
            });

            it('should return error on invalid input', function (done) {

                expect(Hoek.base64urlDecode('*').message).to.exist;
                done();
            });
        });
    });

    describe('#escapeHeaderAttribute', function () {

        it('should not alter ascii values', function (done) {

            var a = Hoek.escapeHeaderAttribute('My Value');
            expect(a).to.equal('My Value');
            done();
        });

        it('should escape all special HTTP header attribute characters', function (done) {

            var a = Hoek.escapeHeaderAttribute('I said go!!!#"' + String.fromCharCode(92));
            expect(a).to.equal('I said go!!!#\\"\\\\');
            done();
        });

        it('should throw on large unicode characters', function (done) {

            var fn = function () {

                Hoek.escapeHeaderAttribute('this is a test' + String.fromCharCode(500) + String.fromCharCode(300));
            };

            expect(fn).to.throw(Error);
            done();
        });

        it('should throw on CRLF to prevent response splitting', function (done) {

            var fn = function () {

                Hoek.escapeHeaderAttribute('this is a test\r\n');
            };

            expect(fn).to.throw(Error);
            done();
        });
    });

    describe('#escapeHtml', function () {

        it('should escape all special HTML characters', function (done) {

            var a = Hoek.escapeHtml('&<>"\'`');
            expect(a).to.equal('&amp;&lt;&gt;&quot;&#x27;&#x60;');
            done();
        });

        it('should return empty string on falsy input', function (done) {

            var a = Hoek.escapeHtml('');
            expect(a).to.equal('');
            done();
        });

        it('should return unchanged string on no reserved input', function (done) {

            var a = Hoek.escapeHtml('abc');
            expect(a).to.equal('abc');
            done();
        });
    });

    describe('#printEvent', function () {

        it('outputs event as string', function (done) {

            var event = {
                timestamp: (new Date(2013, 1, 1, 6, 30, 45, 123)).getTime(),
                tags: ['a', 'b', 'c'],
                data: { some: 'data' }
            };

            Hoek.consoleFunc = function (string) {

                Hoek.consoleFunc = console.log;
                expect(string).to.equal('130201/063045.123, a, {"some":"data"}');
                done();
            };

            Hoek.printEvent(event);
        });

        it('outputs JSON error', function (done) {

            var event = {
                timestamp: (new Date(2013, 1, 1, 6, 30, 45, 123)).getTime(),
                tags: ['a', 'b', 'c'],
                data: { some: 'data' }
            };

            event.data.a = event.data;

            Hoek.consoleFunc = function (string) {

                Hoek.consoleFunc = console.log;
                expect(string).to.equal('130201/063045.123, a, JSON Error: Converting circular structure to JSON');
                done();
            };

            Hoek.printEvent(event);
        });
    });

    describe('#nextTick', function () {

        it('calls the provided callback on nextTick', function (done) {

            var a = 0;

            var inc = function (step, next) {

                a += step;
                next();
            };

            var ticked = Hoek.nextTick(inc);

            ticked(5, function () {

                expect(a).to.equal(6);
                done();
            });

            expect(a).to.equal(0);
            inc(1, function () {

                expect(a).to.equal(1);
            });
        });
    });
});

