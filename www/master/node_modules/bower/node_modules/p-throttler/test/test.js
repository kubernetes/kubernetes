'use strict';

var expect = require('expect.js');
var Q = require('q');
var PThtroller = require('../');

describe('PThtroller', function () {
    var timeout;

    afterEach(function () {
        if (timeout) {
            clearTimeout(timeout);
            timeout = null;
        }
    });

    describe('.enqueue', function () {
        it('return a promise', function () {
            var throttler = new PThtroller();
            var promise;

            promise = throttler.enqueue(function () { return Q.resolve('foo'); });

            expect(promise).to.be.an('object');
            expect(promise.then).to.be.a('function');
        });

        it('should call the function and fulfill the promise accordingly', function (next) {
            var throttler = new PThtroller();

            throttler.enqueue(function () { return Q.resolve('foo'); })
            .then(function (ret) {
                expect(ret).to.equal('foo');

                return throttler.enqueue(function () { return Q.reject(new Error('foo')); });
            })
           .fail(function (err) {
                expect(err).to.be.an(Error);
                expect(err.message).to.equal('foo');
                next();
            })
            .done();
        });

        it('should forward promise progress', function (next) {
            var progress;
            var throttler = new PThtroller();

            throttler.enqueue(function () {
                var deferred = Q.defer();

                setTimeout(function () {
                    deferred.notify(0.5);
                    deferred.resolve('foo');
                }, 200);

                return deferred.promise;
            })
            .progress(function (data) {
                progress = data;
            })
            .then(function (ret) {
                expect(ret).to.equal('foo');
                expect(progress).to.equal(0.5);
                next();
            })
            .done();

        });

        it('should work with functions that return values syncronously', function (next) {
            var throttler = new PThtroller();

            throttler.enqueue(function () { return 'foo'; })
            .then(function (ret) {
                expect(ret).to.equal('foo');
                next();
            })
            .done();
        });

        it('should assume the default concurrency when a type is not specified', function (next) {
            var throttler = new PThtroller(1);
            var calls = 0;

            throttler.enqueue(function () { calls++; return Q.defer().promise; });
            throttler.enqueue(function () { next(new Error('Should not be called!')); });

            timeout = setTimeout(function () {
                expect(calls).to.equal(1);
                next();
            }, 25);
        });

        it('should assume the default concurrency when a type is not known', function (next) {
            var throttler = new PThtroller(1);
            var calls = 0;

            throttler.enqueue(function () { calls++; return Q.defer().promise; }, 'foo_type');
            throttler.enqueue(function () { next(new Error('Should not be called!')); }, 'foo_type');

            timeout = setTimeout(function () {
                expect(calls).to.equal(1);
                next();
            }, 25);
        });

        it('should have different slots when type is not passed or is not known', function (next) {
            var throttler = new PThtroller(1);
            var calls = 0;

            throttler.enqueue(function () { calls++; return Q.defer().promise; });
            throttler.enqueue(function () { calls++; return Q.defer().promise; }, 'foo_type');
            throttler.enqueue(function () { next(new Error('Should not be called!')); });
            throttler.enqueue(function () { next(new Error('Should not be called!')); }, 'foo_type');

            timeout = setTimeout(function () {
                expect(calls).to.equal(2);
                next();
            }, 25);
        });

        it('should use the configured concurrency for the type', function (next) {
            var throttler = new PThtroller(1, {
                foo: 2,
                bar: 3
            });
            var calls = {
                def: 0,
                foo: 0,
                bar: 0
            };

            throttler.enqueue(function () { calls.def++; return Q.defer().promise; });
            throttler.enqueue(function () { next(new Error('Should not be called!')); });
            throttler.enqueue(function () { calls.foo++; return Q.defer().promise; }, 'foo');
            throttler.enqueue(function () { calls.foo++; return Q.defer().promise; }, 'foo');
            throttler.enqueue(function () { calls.bar++; return Q.defer().promise; }, 'bar');
            throttler.enqueue(function () { calls.bar++; return Q.defer().promise; }, 'bar');
            throttler.enqueue(function () { calls.bar++; return Q.defer().promise; }, 'bar');
            throttler.enqueue(function () { next(new Error('Should not be called!')); }, 'bar');

            timeout = setTimeout(function () {
                expect(calls.def).to.equal(1);
                expect(calls.foo).to.equal(2);
                expect(calls.bar).to.equal(3);
                next();
            }, 25);
        });
    });

    describe('.abort', function () {
        it('should clear the whole queue', function (next) {
            var throttler = new PThtroller(1, {
                foo: 2
            });
            var calls = 0;

            throttler.enqueue(function () { calls++; return Q.resolve(); });
            throttler.enqueue(function () { next(new Error('Should not be called!')); });
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');
            throttler.enqueue(function () { next(new Error('Should not be called!')); }, 'foo');

            throttler.abort();

            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');

            timeout = setTimeout(function () {
                expect(calls).to.equal(4);
                next();
            }, 25);
        });

        it('should wait for currently running functions to finish', function (next) {
            var throttler = new PThtroller(1, {
                foo: 2
            });
            var calls = [];

            throttler.enqueue(function () { calls.push(1); return Q.resolve(); });
            throttler.enqueue(function () { calls.push(2); return Q.resolve(); });
            throttler.enqueue(function () {
                var deferred = Q.defer();

                setTimeout(function () {
                    calls.push(3);
                    deferred.resolve();
                }, 25);

                return deferred.promise;
            }, 'foo');

            timeout = setTimeout(function () {
                throttler.abort().then(function () {
                    expect(calls).to.eql([1, 2, 3]);
                    next();
                });
            }, 30);
        });
    });


    describe('scheduler', function () {
        it('should start remaining tasks when one ends', function (next) {
            var throttler = new PThtroller(1, {
                foo: 2
            });
            var calls = 0;

            throttler.enqueue(function () { calls++; return Q.resolve(); });
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');
            throttler.enqueue(function () { calls++; return Q.resolve(); });
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');

            timeout = setTimeout(function () {
                expect(calls).to.equal(5);
                next();
            }, 25);
        });

        it('should respect the enqueue order', function (next) {
            var throttler = new PThtroller(1);
            var defCalls = [];
            var fooCalls = [];

            throttler.enqueue(function () {
                defCalls.push(1);
                return Q.resolve();
            });

            throttler.enqueue(function () {
                defCalls.push(2);
                return Q.resolve();
            });

            throttler.enqueue(function () {
                defCalls.push(3);
                return Q.resolve();
            });

            throttler.enqueue(function () {
                fooCalls.push(1);
                return Q.resolve();
            }, 'foo');

            throttler.enqueue(function () {
                fooCalls.push(2);
                return Q.resolve();
            }, 'foo');

            throttler.enqueue(function () {
                fooCalls.push(3);
                return Q.resolve();
            }, 'foo');

            timeout = setTimeout(function () {
                expect(defCalls).to.eql([1, 2, 3]);
                expect(fooCalls).to.eql([1, 2, 3]);
                next();
            }, 25);
        });

        it('should wait for one slot in every type on a multi-type function', function (next) {
            var throttler = new PThtroller(1, {
                foo: 1,
                bar: 2
            });
            var calls = 0;

            throttler.enqueue(function () { return Q.defer().promise; }, 'foo');
            throttler.enqueue(function () { return Q.defer().promise; }, 'bar');

            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'bar');
            throttler.enqueue(function () { next(new Error('Should not be called!')); }, ['foo', 'bar']);
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'bar');
            throttler.enqueue(function () { next(new Error('Should not be called!')); }, 'foo');

            timeout = setTimeout(function () {
                expect(calls).to.equal(2);
                next();
            }, 25);
        });

        it('should free all type slots when finished running a function', function (next) {
            var throttler = new PThtroller(1, {
                foo: 1,
                bar: 2
            });
            var calls = 0;

            throttler.enqueue(function () { return Q.defer().promise; }, 'bar');
            throttler.enqueue(function () { calls++; return Q.resolve(); }, ['foo', 'bar']);
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'foo');
            throttler.enqueue(function () { calls++; return Q.resolve(); }, 'bar');

            timeout = setTimeout(function () {
                expect(calls).to.equal(3);
                next();
            }, 25);
        });
    });
});
