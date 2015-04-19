var Cache = require('../../../lib/util/Cache');
var expect = require('expect.js');

describe('Cache', function () {
    beforeEach(function () {
        this.cache = new Cache();
    });

    describe('Constructor', function () {
        describe('instantiating cache', function () {
            it('should provide an instance of RegistryClient', function () {
                expect(this.cache instanceof Cache).to.be.ok;
            });

            it('should inherit LRU cache methods', function () {
                var self = this,
                    lruMethods = [
                    'max', 'lengthCalculator', 'length', 'itemCount', 'forEach',
                    'keys', 'values', 'reset', 'dump', 'dumpLru', 'set', 'has',
                    'get', 'peek', 'del'
                ];

                lruMethods.forEach(function (method) {
                    expect(self.cache._cache).to.have.property(method);
                });
            });
        });

        it('should have a get prototype method', function () {
            expect(Cache.prototype).to.have.property('get');
        });

        it('should have a set prototype method', function () {
            expect(Cache.prototype).to.have.property('set');
        });

        it('should have a del prototype method', function () {
            expect(Cache.prototype).to.have.property('del');
        });

        it('should have a clear prototype method', function () {
            expect(Cache.prototype).to.have.property('clear');
        });

        it('should have a reset prototype method', function () {
            expect(Cache.prototype).to.have.property('reset');
        });
    });
});
